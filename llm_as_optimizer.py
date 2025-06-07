import argparse
import dataclasses
import json
import os
from collections.abc import Sequence
from functools import partial

import numpy
import openai
import wandb
from google import genai

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


def main(args: dict):
    dataset = Dataset.load_dir(args["data_dir"])

    rng = numpy.random.default_rng(args["seed"])

    match args["init_method"]:
        case "random":
            init_assignments = init_random(dataset, args["n_init_assignment"])
        case "nearest_first":
            init_assignments = init_nearest_first(dataset)
        case _:
            msg = f"Unknown init method: {args['init_method']}"
            raise ValueError(msg)

    old_results = init_assignments

    for result in old_results:
        print(
            f"initial destroyed ships: {result.destroyed_ships}, "
            f"total damage: {result.total_damage:.4f}"
        )

    sample_func = get_sample_method(
        rng, args["sample_method"], args["n_example"], args["gumbel_tau"]
    )

    get_prompt = partial(
        get_optimizer_prompt,
        method=args["prompt_method"],
        include_damage=args["prompt_include_damage"],
    )

    if args["debug"]:
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
            **args,
        },
    )
    run.define_metric("destroyed_ships", summary="max")
    run.define_metric("total_damage", summary="max")

    step = 0

    match args["model_provider"]:
        case "openai":
            client = openai.OpenAI(api_key=args["openai_api_key"])
        case "google":
            client = genai.Client(api_key=args["google_api_key"])
        case _:
            msg = f"Unknown model provider: {args['model_provider']}"
            raise ValueError(msg)

    api_called_count = 0
    total_input_tokens, total_output_tokens = 0, 0
    while step < args["n_step"]:
        examples: list[OptimizePromptResult] = sample_func(old_results)  # type: ignore
        prompt = get_prompt(examples, dataset)
        response = get_ai_response(
            prompt,
            args["model"],
            client,
            args["temperature"],
            args["max_output_tokens"],
            metadata={"step": str(step), "run-name": run.name},
        )

        input_tokens = count_tokens(prompt, client, args["model"])
        output_tokens = count_tokens(response, client, args["model"])
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--debug", action="store_true", default=False, help="print init prompt"
    )
    argparser.add_argument(
        "--data-dir", type=str, required=True, help="path to data directory"
    )
    argparser.add_argument(
        "--n-init-assignment", type=int, default=3, help="number of initial assignments"
    )
    argparser.add_argument(
        "--init-method",
        choices=["random", "nearest_first"],
        default="random",
        help="method for initializing example assignments",
    )
    argparser.add_argument(
        "--n-step", type=int, default=250, help="number of optimization steps"
    )
    argparser.add_argument(
        "--n-example",
        type=int,
        default=6,
        help="number of example assignments in prompt",
    )
    argparser.add_argument(
        "--sample-method",
        choices=["best", "uniform", "gumbel_top_k"],
        default="best",
        help="method for sampling example assignments",
    )
    argparser.add_argument(
        "--gumbel-tau", type=float, default=1.0, help="temperature for gumbel sampling"
    )
    argparser.add_argument(
        "--damage-factor",
        type=float,
        default=0.1,
        help="how much to weigh damage contributions in gumbel sampling",
    )
    argparser.add_argument(
        "--temperature", type=float, default=1.5, help="temperature for LLM"
    )
    argparser.add_argument(
        "--seed", type=int, default=0, help="seed for init assignments and model"
    )
    argparser.add_argument(
        "--model", type=str, default="gpt-4.1-mini", help="model name"
    )
    argparser.add_argument(
        "--model-provider",
        choices=["openai", "google"],
        default="openai",
        help="model provider",
    )
    argparser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ["OPENAI_API_KEY"],
        help="OpenAI API key",
    )
    argparser.add_argument(
        "--google-api-key",
        type=str,
        default=os.environ["GOOGLE_API_KEY"],
        help="Google API key",
    )

    argparser.add_argument(
        "--prompt-method",
        choices=["direct", "chain"],
        default="direct",
        help="prompt method",
    )
    argparser.add_argument(
        "--prompt-include-damage",
        action="store_true",
        default=False,
        help="include total damage in prompt",
    )
    argparser.add_argument(
        "--max-output-tokens", type=int, default=256, help="max output tokens"
    )

    args = argparser.parse_args()

    main(vars(args))
