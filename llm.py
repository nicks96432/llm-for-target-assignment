import time
from typing import Any

import google.genai.errors
import google.genai.types
import httpx
import openai
import tiktoken
from google import genai

from config import LLMConfig


def get_ai_response(
    client: openai.OpenAI | genai.Client,
    prompt: str,
    config: LLMConfig,
    metadata: dict[str, Any] | None = None,
) -> str:
    if metadata is None:
        metadata = {}

    if isinstance(client, openai.OpenAI):
        while True:
            try:
                response = client.responses.create(
                    model=config.model,
                    input=[{"role": "user", "content": prompt}],
                    metadata=metadata,  # type: ignore
                    max_output_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                    store=True,
                    timeout=config.timeout,
                )
            except openai.APIError as e:
                print(getattr(e, "message", str(e)))
                time.sleep(10)
            else:
                return response.output_text
    elif isinstance(client, genai.Client):
        match config.model:
            case "gemini-2.5-pro-preview-06-05":
                thinking_config = google.genai.types.ThinkingConfig(thinking_budget=128)
            case "gemini-2.5-flash-preview-04-17":
                thinking_config = google.genai.types.ThinkingConfig(thinking_budget=0)
            case _:
                thinking_config = None

        while True:
            try:
                response = client.models.generate_content(
                    model=config.model,
                    contents=prompt,
                    config=google.genai.types.GenerateContentConfig(
                        http_options=google.genai.types.HttpOptions(
                            timeout=config.timeout * 1000
                        ),
                        max_output_tokens=config.max_output_tokens,
                        temperature=config.temperature,
                        thinking_config=thinking_config,
                    ),
                )
            except google.genai.errors.APIError as e:
                print(getattr(e, "message", str(e)))
                time.sleep(10)
            except httpx.HTTPError:
                time.sleep(10)
            else:
                return response.text or ""

    msg = f"Unknown client type: {type(client)}"
    raise ValueError(msg)


def count_tokens(text: str, client: Any, model: str) -> int:
    if isinstance(client, openai.OpenAI):
        if model.startswith("gpt-4.1"):
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    elif isinstance(client, genai.Client):
        try:
            return (
                client.models.count_tokens(model=model, contents=text).total_tokens or 0
            )
        except google.genai.errors.APIError as e:
            print(getattr(e, "message", str(e)))
            return 0

    msg = f"Unknown client type: {type(client)}"
    raise ValueError(msg)
