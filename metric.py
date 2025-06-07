"""
Evaluation metrics.
"""

import concurrent.futures
from collections.abc import Callable, Sequence

import numpy
from tqdm.auto import tqdm

from dataset import Dataset


def evaluate_heuristic(
    heuristic: Callable[[Dataset, list[float], int], int],
    datasets: Sequence[Dataset],
) -> float:
    """
    Evaluates a heuristic over a range of datasets.

    Parameters
    ----------
    heuristic : Callable[[Dataset, list[float], int], int]
        heuristic to evaluate
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

    def run_single_evaluation(dataset: Dataset) -> float:
        n_ship = dataset.ship_types.shape[0]
        n_missle = len(dataset.missles)

        ship_health = [1.0] * n_ship
        assignment = []
        for i, _ in enumerate(dataset.missles):
            ship_id = heuristic(dataset, ship_health, i + 1)
            if dataset.can_hit_ship(dataset.missles[i], ship_id):
                ship_health[ship_id - 1] -= dataset.missle_damages[
                    dataset.ship_types[ship_id - 1] - 1, dataset.missles[i].type - 1
                ].item()
                assignment.append(ship_id)
            else:
                assignment.append(0)

        if len(assignment) != n_missle:
            return 0.0

        ship_health = numpy.array(ship_health)
        ship_health[ship_health < 0] = 0

        destroyed_ship = numpy.sum(
            numpy.array(ship_health) <= 0, dtype=numpy.int64
        ).item()

        return destroyed_ship / n_ship

    with concurrent.futures.ProcessPoolExecutor() as executor:
        destroy_rates = list(
            tqdm(executor.map(run_single_evaluation, datasets), total=len(datasets))
        )

    avg_destroy_rate = float(numpy.mean(destroy_rates))

    return avg_destroy_rate
