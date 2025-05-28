"""
Prompt construction.
"""

import dataclasses
import itertools
import textwrap
from collections.abc import Callable, Sequence
from functools import partial

import numpy

from dataset import Dataset


@dataclasses.dataclass(frozen=True, slots=True)
class PromptResult:
    prompt: str
    response: str | None
    step: int
    destroyed_ships: float
    total_damage: float


@dataclasses.dataclass(frozen=True, slots=True)
class OptimizePromptResult(PromptResult):
    assignment: list[int]


@dataclasses.dataclass(frozen=True, slots=True)
class HeuristicGeneratePromptResult(PromptResult):
    heuristic: str


def get_optimizer_prompt(
    sampled_results: Sequence[OptimizePromptResult],
    dataset: Dataset,
    method: str,
    include_damage: bool = True,
) -> str:
    """
    Parameters
    ----------
    sampled_results : `Sequence[PromptResult]`
        example previous target assignments with their metrics, sorted by metrics
    dataset : `Dataset`
        dataset used to generate the prompt

    """

    prompt = (
        f"There are {len(dataset.ships_xy)} battle ships invading the island, and "
        f"there are {dataset.require_missles.shape[1]} types of missles that can be "
        "used to destroy them.\n\n"
    )

    prompt += "Below is a list of missles with their types and available targets:\n"

    for i, missle in enumerate(dataset.missles):
        prompt += f"{i}: type: {missle.type}, targets: {missle.available_targets}\n"

    prompt += (
        "\n"
        "Here is the list of how many missles each ship requires to destroy it with "
        "each missle type, where -1 means you can't destroy it with the missle type:\n"
    )

    prompt += "\n".join(
        f"{i}: {' '.join(str(x) for x in dataset.require_missles[ship.type - 1])}"  # type: ignore
        for i, ship in enumerate(dataset.ship_types, start=1)
    )

    if sampled_results:
        prompt += (
            "\n\n"
            "Below are some previous target assignments with their number "
            "of destroyed ships, where higher values are better."
        )

        if include_damage:
            prompt += (
                " If the number of destroyed ships is the same, higher damage "
                "is better. If there are no available targets, assign the missle to 0."
            )

        prompt += "\n\n"

        old_value_pairs_substr = ""
        for result in sampled_results:
            targets_str = " ".join(map(str, result.assignment))

            total_damage_str = (
                f", total damage: {result.total_damage:.4f}" if include_damage else ""
            )
            old_value_pairs_substr += textwrap.dedent(
                f"""
                <target> {targets_str} </target>
                destroyed ships: {result.destroyed_ships}{total_damage_str}
                """
            )

        prompt += old_value_pairs_substr.strip()
        prompt += "\n\n"

    match method:
        case "direct":
            additional_prompt = (
                "No other text should be included. Output the assignment directly. "
                "Do not include any explanation."
            )
        case "chain":
            additional_prompt = "Think step by step and show your answer."
        case _:
            msg = f"Unknown prompt method: {method}"
            raise ValueError(msg)

    prompt += (
        "Give me a new assignment that is different from all assignments above, "
        "and has more destroyed ships than any of the above. The length of "
        f"assignment should be equal to the number of missles ({len(dataset.missles)}). "  # noqa: E501
        "Assign all missles exactly once. The assignment should start with "
        "'<target>' and end with '</target>'. Assignment should contain only "
        f"targets that are separated by spaces. {additional_prompt}\n"
    )

    return prompt


def get_heuristic_generator_prompt(
    sampled_results: Sequence[HeuristicGeneratePromptResult],
    dataset: Dataset,
    method: str,
    detailed_table: bool = False,
) -> str:
    """
    Parameters
    ----------
    sampled_results : `Sequence[HeuristicGeneratePromptResult]`
        example previous target assignments with their metrics, sorted by metrics
    dataset : `Dataset`
        dataset used to generate the prompt

    """

    prompt = (
        f"There are {len(dataset.ships_xy)} battle ships invading the island, and "
        f"{dataset.require_missles.shape[1]} types of missles are available for "
        "defense.\n\n"
    )

    prompt += (
        "Below is a list of available missiles, where each entry specifies the "
        "missile type and the IDs of the ships it can target:\n"
    )

    missles_iter = itertools.islice(dataset.missles, 3)
    if detailed_table:
        missles_iter = iter(dataset.missles)

    for i, missle in enumerate(missles_iter, start=1):
        prompt += f"{i}: type: {missle.type}, targets: {missle.available_targets}\n"

    if not detailed_table and len(dataset.missles) > 3:
        prompt += "...\n"
        prompt += (
            f"{len(dataset.missles)}: type: {dataset.missles[-1].type}, "
            f"targets: {dataset.missles[-1].available_targets}\n"
        )

    prompt += (
        "\n"
        "Each ship requires a certain number of missiles of each type to be destroyed. "
        "The table below shows the number of missiles required per ship, indexed by "
        "ship ID. A value of -1 indicates that a ship cannot be destroyed by that "
        "missile type:\n"
    )

    ship_types_iter = itertools.islice(dataset.ship_types, 3)
    if detailed_table:
        ship_types_iter = iter(dataset.ship_types)

    for i, ship in enumerate(ship_types_iter, start=1):
        prompt += (
            f"{i}: {' '.join(str(x) for x in dataset.require_missles[ship - 1])}\n"
        )

    if not detailed_table and len(dataset.ship_types) > 3:
        prompt += "...\n"
        prompt += (
            f"{len(dataset.ship_types)}: "
            f"{' '.join(str(x) for x in dataset.require_missles[-1])}\n"
        )

    prompt += textwrap.dedent(
        """
        To solve this problem in Python, the data is encapsulated in a `Dataset` object named `dataset`, which includes the following attributes:

        * `ships_xy`: A `numpy.ndarray` of shape `(n_ship, 2)` representing the x and y coordinates of each ship.
        * `ship_types`: A `numpy.ndarray` of shape `(n_ship)` indicating the type of each ship.
        * `turrets_xy`: A `numpy.ndarray` of shape `(n_turret, 2)` representing the x and y coordinates of each turret.
        * `turrets_missle_count`: A `numpy.ndarray` of shape `(n_turret, n_missle_type)` representing the x and y coordinates of each turret.
        * `missle_types`: A `DataFrame` with columns `type`, `min_range` and `max_range`.
        * `require_missles`: A `numpy.ndarray` of shape `(n_ship_type, n_missle_type)` indicating how many missiles are required to destroy each ship type with each missile type.
        * `distances`: A `numpy.ndarray` of shape `(n_turret, n_ship)` representing distances between each turret and each ship.
        * `in_range`: A `numpy.ndarray` of shape `(n_missle_type, n_turret, n_ship)` indicating whether a ship is in range of a turret for a given missile type.
        * `missle_damages`: A `numpy.ndarray` of shape `(n_ship_type, n_missle_type)` indicating the damage a missile type can inflict on each ship type.
        * `missles`: A list of Missle objects, where each has:
            * `type`: `int`
            * `turret_id`: `int`
            * `available_targets`: `list[int]` of ship IDs

        Note: IDs for ships, turrets, and missile types are indexed from 1 based on their order in the corresponding data structure.
        Your task is to design a heuristic function that assigns a missile to a ship. This function takes the following parameters:

        1. `dataset`: the current `Dataset` object representing the scenario
        2. `ship_health`: a list of the current health values for all ships
        3. `i`: the index of the missile to be assigned, starting from 1
        """  # noqa: E501
    )

    if sampled_results:
        prompt += (
            "\n"
            "Below is some examples of a previously implemented heuristic function for "
            "missile assignment, along with its performance metrics. The performance "
            "is measured by the number of ships destroyed—where a higher count is "
            "better. In the case of a tie, the total inflicted damage is used as a "
            "secondary criterion. If a missile has no available targets, it should be "
            "assigned to 0 (i.e., no assignment).\n\n"
        )

        old_value_pairs_substr = ""
        for result in sampled_results:
            old_value_pairs_substr += textwrap.dedent(
                f"""
```
{result.heuristic}```
destroyed ships: {result.destroyed_ships}, total damage: {result.total_damage:.4f}
"""
            )

        prompt += old_value_pairs_substr.strip()
        prompt += "\n\n"

    match method:
        case "direct":
            additional_prompt = (
                "No other text should be included. Complete the function directly. "
                "Do not include any explanation."
            )
        case "chain":
            additional_prompt = "Think step by step and show your answer."
        case _:
            msg = f"Unknown prompt method: {method}"
            raise ValueError(msg)

    prompt += (
        "Improve the heuristic function shown below. "
        f"Wrap your solution within a markdown code block. {additional_prompt}\n"
    )

    prompt += textwrap.dedent(
        """
        ```
        def heuristic(dataset: Dataset, ship_health: list[float], i: int) -> int:
            pass
        ```
        """
    )

    return prompt


def sample_best[T](ascending_items: Sequence[T], n_best=10) -> list[T]:
    """
    Gets the best `n_best` items from an ascending sequence.

    Parameters
    ----------
    ascending_items : Sequence[T]
        items to sample from
    n_best : int, optional
        number of items to sample, by default 10

    Returns
    -------
    list[T]
        sampled items
    """

    return list(ascending_items)[-n_best:]


def sample_uniform[T](items: Sequence[T], n_sample=10) -> list[T]:
    """
    Gets `n_sample` items randomly from a sequence.

    Parameters
    ----------
    items : Sequence[T]
        items to sample from
    n_sample : int, optional
        number of items to sample, by default 10

    Returns
    -------
    list[T]
        sampled items
    """

    n_sample = min(n_sample, len(items))

    return numpy.random.choice(items, n_sample, replace=False).tolist()  # type: ignore


def sample_gumbel_top_k[P: PromptResult](
    old_results: Sequence[P], k=10, tau=1.0, damage_factor=0.1
) -> list[P]:
    if k > len(old_results):
        return list(old_results)

    scores = numpy.array(
        [result.destroyed_ships for result in old_results],
        dtype=numpy.float64,
    ) + (numpy.array([result.total_damage for result in old_results]) * damage_factor)

    scores = (scores - scores.mean()) / (scores.std() + numpy.finfo(numpy.float32).eps)

    gumbel_noise = numpy.random.gumbel(0.0, 1.0, len(scores))
    noisy_scores = (scores / tau) + gumbel_noise

    top_k_indices = numpy.argpartition(noisy_scores, -k)[-k:]
    top_k_indices = top_k_indices[numpy.argsort(noisy_scores[top_k_indices])]

    return [old_results[i] for i in top_k_indices]


def get_sample_method(
    method: str, n_sample: int, gumbel_tau: float
) -> Callable[[Sequence[PromptResult]], list[PromptResult]]:
    match method:
        case "best":
            return partial(sample_best, n_best=n_sample)
        case "uniform":
            return partial(sample_uniform, n_sample=n_sample)
        case "gumbel_top_k":
            return partial(sample_gumbel_top_k, k=n_sample, tau=gumbel_tau)
        case _:
            msg = f"Unknown sample method: {method}"
            raise ValueError(msg)
