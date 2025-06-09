"""
Evaluation metrics.
"""

import concurrent.futures
import os
import statistics
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy
from tqdm.auto import tqdm

from dataset import Dataset


def _run_single_evaluation(heuristic_code: str, dataset: Dataset) -> float:
    namespace: dict[str, Any] = {"Dataset": Dataset, "numpy": numpy, "np": numpy}

    try:
        exec(heuristic_code, namespace)
    except Exception:
        return float("nan")

    try:
        heuristic = namespace["heuristic"]
    except KeyError:
        return float("nan")

    n_ship = dataset.ship_types.shape[0]
    n_missle = len(dataset.missles)

    ship_health = [1.0] * n_ship
    assignment = []
    for i, _ in enumerate(dataset.missles):
        try:
            ship_id = heuristic(dataset, ship_health, i + 1)
        except Exception:
            return float("nan")

        if dataset.can_hit_ship(dataset.missles[i], ship_id):
            ship_health[ship_id - 1] -= dataset.missle_damages[
                dataset.ship_types[ship_id - 1] - 1, dataset.missles[i].type - 1
            ].item()
            assignment.append(ship_id)
        else:
            assignment.append(0)

    if len(assignment) != n_missle:
        return float("nan")

    ship_health = numpy.array(ship_health)
    ship_health[ship_health < 0] = 0

    destroyed_ship = numpy.sum(numpy.array(ship_health) <= 0, dtype=numpy.int64).item()

    return destroyed_ship / n_ship


def evaluate_heuristic(
    heuristic_code: str, datasets: Sequence[Dataset], timeout: float | None = None
) -> float:
    """
    Evaluates a heuristic over a range of datasets.

    Parameters
    ----------
    heuristic : str
        heuristic code to evaluate
    config_low : DatasetConfig
        lower bound dataset configuration
    config_high : DatasetConfig
        upper bound dataset configuration
    n_dataset : int, optional
        number of datasets to generate, by default 1000

    Returns
    -------
    float
        average ship destroy rate
    """

    evaluation_func = partial(_run_single_evaluation, heuristic_code)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        destroy_rates = list(
            tqdm(
                executor.map(
                    evaluation_func,
                    datasets,
                    chunksize=int(len(datasets) / (os.cpu_count() or 1)),
                    timeout=timeout,
                ),
                total=len(datasets),
            )
        )

    return statistics.fmean(destroy_rates)
