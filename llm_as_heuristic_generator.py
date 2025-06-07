import inspect
import traceback
from functools import partial
from typing import Any

import numpy
import openai
import tiktoken
import wandb
from google import genai

from config import DatasetConfig, HeuristicGenerationConfig
from dataset import Dataset
from llm import count_tokens, get_ai_response
from metric import evaluate_heuristic
from prompt import (
    HeuristicGeneratePromptResult,
    get_heuristic_generator_prompt,
    get_sample_method,
)


def extract_heuristic_code(string: str) -> str:
    """
    Removes the first line and the last line of the string.

    Parameters
    ----------
    string : str
        the string

    Returns
    -------
    str
        the extracted string
    """

    return string.partition("\n")[2].rpartition("\n")[0] + "\n"


def example_heuristic(dataset: Dataset, ship_health: list[float], i: int) -> int:
    missle = dataset.missles[i - 1]

    if not missle.available_targets:
        return 0

    return missle.available_targets[0]


def main(config: HeuristicGenerationConfig):
    rng = numpy.random.default_rng(config.seed)

    datasets = []
    for seed in range(config.n_dataset):
        dataset_config = DatasetConfig.model_validate({
            **{
                field: rng.integers(
                    getattr(config.dataset_gen_low, field),
                    getattr(config.dataset_gen_high, field) + 1,
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
            },
            "seed": seed,
        })

        datasets.append(Dataset.generate(dataset_config))

    example_dataset = Dataset.load_dir(config.prompt.example_dataset_dir)

    sample_func = get_sample_method(
        rng, config.prompt.sample_method, config.prompt.n_example, config.gumbel_tau
    )

    get_prompt = partial(
        get_heuristic_generator_prompt,
        method=config.prompt.method,
        detailed_table=config.prompt.detailed_table,
    )

    old_results: list[HeuristicGeneratePromptResult] = []

    if config.debug:
        metric = evaluate_heuristic(example_heuristic, datasets)
        old_results.append(
            HeuristicGeneratePromptResult(
                prompt="",
                response="",
                step=0,
                heuristic=inspect.getsource(example_heuristic),
                average_destroy_rate=metric,
            )
        )
        print(
            f"""----------------------------prompt------------------------------
{get_prompt(old_results, example_dataset)}
----------------------------------------------------------------
average destroy rate: {metric}"""
        )
        return

    run = wandb.init(
        project="target-assignment-llm-algorithm-generator",
        config={
            "init_heuristics": [result.heuristic for result in old_results],
            "init_average_destroy_rate": [
                result.average_destroy_rate for result in old_results
            ],
            "init_prompt": get_prompt(old_results, example_dataset),
            **config.model_dump(),
        },
    )
    run.define_metric("average_destroy_rate", summary="max")
    run.define_metric("total_damage", summary="max")

    step = 0

    match config.llm.provider:
        case "openai":
            client_cls = openai.OpenAI
        case "google":
            client_cls = genai.Client
        case _:
            msg = f"Unknown model provider: {config.llm.provider}"
            raise ValueError(msg)

    client = client_cls(api_key=config.llm.api_key)

    api_called_count = 0
    total_input_tokens, total_output_tokens = 0, 0
    while step < config.n_step:
        examples: list[HeuristicGeneratePromptResult] = sample_func(old_results)  # type: ignore
        prompt = get_prompt(examples, example_dataset)
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

        heuristic_code = extract_heuristic_code(response)  # type: ignore
        if not heuristic_code:
            print("Invalid heuristic.")
            continue

        namespace: dict[str, Any] = {"Dataset": Dataset}
        try:
            exec(heuristic_code, namespace)
            metric = evaluate_heuristic(namespace["heuristic"], datasets)
            old_results.append(
                HeuristicGeneratePromptResult(
                    prompt=prompt,
                    response=response,
                    step=step,
                    heuristic=heuristic_code,
                    average_destroy_rate=metric,
                )
            )
        except Exception as e:
            print(f"Invalid heuristic: {e}")
            print(traceback.format_exc())
            continue

        old_results.sort(key=lambda r: (r.average_destroy_rate))

        print(f"step {step:3d}: average ship destroy rate: {metric}")

        encoding = tiktoken.get_encoding("o200k_base")
        input_tokens = len(encoding.encode(prompt))
        output_tokens = len(encoding.encode(response))

        run.log({
            "average_destroy_rate": metric,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        step += 1

    best_result = old_results[-1]
    print(f"Best heuristic (found in step {best_result.step}):")
    print(best_result.heuristic)
    print(f"average ship destroy rate: {best_result.average_destroy_rate} ")

    print(f"API called count: {api_called_count}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    run.log({
        "best_heuristic": best_result.heuristic,
        "best_step": best_result.step,
        "api_called_count": api_called_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    })

    run.finish()


if __name__ == "__main__":
    with open("config.json", encoding="utf-8") as f:
        config = HeuristicGenerationConfig.model_validate_json(f.read())

    main(config)
