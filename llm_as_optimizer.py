import argparse
import dataclasses
import json
from collections.abc import Sequence
from functools import partial

import numpy
import openai
import wandb
from google import genai

from config import OptimizationConfig
from dataset import Dataset, Missle
from llm import count_tokens, get_ai_response
from prompt import OptimizePromptResult, get_optimizer_prompt, get_sample_method


def evaluate_destroyed_ships(
    assignment: Sequence[int],
    dataset: Dataset,
) -> tuple[int, float]:
    damage_on_ship = numpy.zeros(len(dataset.ships_xy), dtype=numpy.float64)
    for i, missle in enumerate(dataset.missles):
        if dataset.can_hit_ship(missle, assignment[i]):
            damage_on_ship[assignment[i] - 1] += dataset.missle_damages[
                dataset.ship_types[assignment[i] - 1] - 1, missle.type - 1
            ]

    damage_on_ship[damage_on_ship >= 1] = 1

    return numpy.sum(
        damage_on_ship[damage_on_ship >= 1], dtype=numpy.int64
    ).item(), numpy.sum(damage_on_ship).item()


def extract_assignment(input_string: str, n_ships: int):
    start_string = "<target>"
    end_string = "</target>"
    if start_string not in input_string or end_string not in input_string:
        return []

    input_string = input_string[
        input_string.rindex(start_string) + len(start_string) : input_string.rindex(
            end_string
        )
    ]

    parsed_list = []
    for s in input_string.split(" "):
        p = s.strip()
        try:
            p = int(p)
            if p not in range(n_ships + 1):
                return []
        except ValueError:
            continue
        parsed_list.append(p)

    return parsed_list


def get_random_assignment(missles: Sequence[Missle]) -> list[int]:
    assignment = []
    for missle in missles:
        if missle.available_targets:
            assignment.append(int(numpy.random.choice(missle.available_targets)))
        else:
            assignment.append(0)

    return assignment


def init_random(dataset: Dataset, n: int) -> list[OptimizePromptResult]:
    results = []
    for _ in range(n):
        assignment = get_random_assignment(dataset.missles)
        metrics = evaluate_destroyed_ships(assignment, dataset)

        result = OptimizePromptResult(
            prompt="",
            response="",
            step=0,
            assignment=assignment,
            destroyed_ships=metrics[0],
            total_damage=metrics[1],
        )
        results.append(result)

    return results


def init_nearest_first(dataset: Dataset) -> list[OptimizePromptResult]:
    assignment = []
    damages = [0.0] * len(dataset.ships_xy)
    for missle in dataset.missles:
        targets = missle.available_targets
        if not targets:
            assignment.append(0)
            continue

        targets = numpy.array(targets)
        target_distances = dataset.distances[missle.turret_id - 1, targets - 1]  # type: ignore
        targets = numpy.array(targets)[numpy.argsort(target_distances)]

        for target in targets:
            if damages[target - 1] < 1:
                assignment.append(target.item())
                damages[target - 1] += dataset.missle_damages[
                    dataset.ship_types[target - 1] - 1, missle.type - 1
                ]
                break
        else:
            assignment.append(0)

    metrics = evaluate_destroyed_ships(assignment, dataset)

    result = OptimizePromptResult(
        prompt="",
        response="",
        step=0,
        assignment=assignment,
        destroyed_ships=metrics[0],
        total_damage=metrics[1],
    )

    return [result]


def main(config: OptimizationConfig):
    dataset = Dataset.load_dir(config.prompt.example_dataset_dir)

    rng = numpy.random.default_rng(config.seed)

    match config.init_method:
        case "random":
            init_assignments = init_random(dataset, config.n_init_assignment)
        case "nearest_first":
            init_assignments = init_nearest_first(dataset)
        case _:
            msg = f"Unknown init method: {config.init_method}"
            raise ValueError(msg)

    old_results = init_assignments

    for result in old_results:
        print(
            f"initial destroyed ships: {result.destroyed_ships}, "
            f"total damage: {result.total_damage:.4f}"
        )

    sample_func = get_sample_method(
        rng, config.prompt.sample_method, config.prompt.n_example, config.gumbel_tau
    )

    get_prompt = partial(
        get_optimizer_prompt,
        method=config.prompt.method,
        include_damage=config.prompt.include_damage,
    )

    if config.debug:
        print("----------------------------prompt------------------------------")
        print(get_prompt(old_results, dataset))
        print("----------------------------------------------------------------")
        return

    run = wandb.init(
        project="target-assignment-llm-optimizer",
        config={
            "init_assignments": [result.assignment for result in old_results],
            "init_destroyed_ships": [result.destroyed_ships for result in old_results],
            "init_prompt": get_prompt(old_results, dataset),
            "n_missle": len(dataset.missles),
            "n_ship": len(dataset.ships_xy),
            **config.model_dump(),
        },
    )
    run.define_metric("destroyed_ships", summary="max")
    run.define_metric("total_damage", summary="max")

    step = 0

    match config.llm.provider:
        case "openai":
            client_cls = openai.OpenAI
        case "genai":
            client_cls = genai.Client
        case _:
            msg = f"Unknown model provider: {config.llm.provider}"
            raise ValueError(msg)

    client = client_cls(api_key=config.llm.api_key)

    api_called_count = 0
    total_input_tokens, total_output_tokens = 0, 0
    while step < config.n_step:
        examples: list[OptimizePromptResult] = sample_func(old_results)  # type: ignore
        prompt = get_prompt(examples, dataset)
        response = get_ai_response(
            client,
            prompt,
            config.llm,
            metadata={"step": str(step), "run-name": run.name},
        )

        input_tokens = count_tokens(prompt, client, config.llm.model)
        output_tokens = count_tokens(response, client, config.llm.model)
        api_called_count += 1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        assignment = extract_assignment(response, len(dataset.ships))  # type: ignore
        if not assignment:
            print("Invalid assignment.")
            continue
        if len(assignment) != len(dataset.missles):
            print(
                f"Assignment length {len(assignment)} != "
                f"missles count {len(dataset.missles)}"
            )
            continue

        metric = evaluate_destroyed_ships(assignment, dataset)

        for result in old_results:
            if assignment == result.assignment:
                print("Assignment already seen.")
                continue

        old_results.append(
            OptimizePromptResult(
                prompt, response, step, assignment, metric[0], metric[1]
            )
        )
        old_results.sort(key=lambda r: (r.destroyed_ships, r.total_damage))

        print(
            f"step {step:3d}: destroyed ships: {metric[0]}, "
            f"total damage: {metric[1]:.4f}"
        )
        run.log({"destroyed_ships": metric[0], "total_damage": metric[1]})

        run.log({
            "destroyed_ships": metric[0],
            "total_damage": metric[1],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        step += 1

    best_result = old_results[-1]
    print(
        f"Best assignment (found in step {best_result.step}):", best_result.assignment
    )
    print(
        f"destroyed ships: {best_result.destroyed_ships}, "
        f"total damage: {best_result.total_damage:.4f}"
    )

    print(f"API called count: {api_called_count}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    run.log({
        "best_assignment": best_result.assignment,
        "best_step": best_result.step,
        "api_called_count": api_called_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    })

    run.finish()

    old_results.sort(key=lambda r: r.step)
    old_results = [dataclasses.asdict(result) for result in old_results]
    with open(f"llm_as_optimizer_results_{run.name}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(old_results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", type=str, help="Config file path")
    config_file = parser.parse_args().config_file

    with open(config_file, encoding="utf-8") as f:
        config = OptimizationConfig.model_validate_json(f.read())

    main(config)
