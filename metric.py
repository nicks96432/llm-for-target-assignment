"""
Evaluation metrics.
"""

import concurrent.futures
from collections.abc import Callable

import numpy
from tqdm.auto import tqdm

from dataset import Config as DatasetConfig
from dataset import Dataset


def evaluate_heuristic(
    heuristic: Callable[[Dataset, list[float], int], int],
    config_low: DatasetConfig,
    config_high: DatasetConfig,
    n_dataset: int = 1000,
) -> tuple[float, float]:
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
    tuple[float, float]
        tuple of average destroyed ships and total damage
    """

    def run_single_evaluation():
        config = DatasetConfig(**{
            field: numpy.random.randint(
                getattr(config_low, field), getattr(config_high, field) + 1
            )
            for field in [
                "n_ship",
                "n_turret",
                "n_missle",
                "n_ship_type",
                "n_missle_type",
                "max_ship_coast_distance",
                "max_turret_coast_distance",
            ]
        })

        dataset = Dataset.generate(config)

        ship_health = [1.0] * len(dataset.ship_types)
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

        if len(assignment) != len(dataset.missles):
            return 0, 0.0

        ship_health = numpy.array(ship_health)
        ship_health[ship_health < 0] = 0

        destroyed_ship = numpy.sum(
            numpy.array(ship_health) <= 0, dtype=numpy.int64
        ).item()
        total_damage = len(ship_health) - numpy.sum(numpy.array(ship_health)).item()

        return destroyed_ship, total_damage

    destroyed_ships = []
    total_damages = []

    with (
        tqdm(total=n_dataset) as progress_bar,
        concurrent.futures.ProcessPoolExecutor() as executor,
    ):
        futures = []
        for _ in range(n_dataset):
            future = executor.submit(run_single_evaluation)
            future.add_done_callback(lambda _: progress_bar.update())
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            destroyed_ship, total_damage = future.result()
            destroyed_ships.append(destroyed_ship)
            total_damages.append(total_damage)

    avg_destroyed_ship = float(numpy.mean(destroyed_ships))
    avg_total_damage = float(numpy.mean(total_damages))

    return avg_destroyed_ship, avg_total_damage
