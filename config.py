from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LLMProvider(StrEnum):
    OPENAI = "openai"
    GENAI = "genai"


class PromptMethod(StrEnum):
    CHAIN = "chain"
    DIRECT = "direct"


class SampleMethod(StrEnum):
    BEST = "best"
    UNIFORM = "uniform"
    GUMBEL_TOP_K = "gumbel_top_k"


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = Field(default="gpt-4.1-mini", title="LLM model name")
    temperature: float = Field(default=1.0, title="LLM temperature")
    max_output_tokens: int = Field(default=8192, title="LLM max output tokens")
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, title="LLM provider")
    api_key: str = Field(title="LLM API key")
    timeout: int = Field(default=180, title="LLM API timeout in seconds")


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_ship: int = Field(default=20, title="ship count", ge=10, le=400)
    n_turret: int = Field(default=20, title="turret count")
    n_missle: int = Field(default=100, title="missle count", ge=10, le=1000)
    n_ship_type: int = Field(default=10, title="ship type count", ge=5, le=30)
    n_missle_type: int = Field(default=2, title="missle type count")
    max_ship_coast_distance: int = Field(
        default=50000, title="maximum distance between ships and coast"
    )
    max_turret_coast_distance: int = Field(
        default=3000, title="maximum distance between turrets and coast"
    )
    seed: int = Field(
        default=0,
        title="random seed",
        description="this is not used for seed selection, but for dataset generation",
    )
    twd97: bool = Field(default=True, title="use TWD97 coordinate system")


class PromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_example: int = Field(default=2, title="number of examples heuristics in prompt")
    example_dataset_dir: str = Field(title="example dataset directory")
    method: PromptMethod = Field(default=PromptMethod.CHAIN, title="prompt method")
    sample_method: SampleMethod = Field(
        default=SampleMethod.GUMBEL_TOP_K,
        title="method for sampling few-shot example heuristics",
    )
    detailed_table: bool = Field(default=False, title="use detailed table prompt")


DATASET_GEN_DEFAULT_LOW = DatasetConfig.model_construct(
    n_ship=10, n_turret=10, n_missle=50, n_ship_type=10, n_missle_type=2
)
DATASET_GEN_DEFAULT_HIGH = DatasetConfig.model_construct(
    n_ship=400, n_turret=100, n_missle=1000, n_ship_type=30, n_missle_type=5
)


class HeuristicGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    debug: bool = Field(default=False, title="print init prompt and exit")

    seed: int = Field(default=0, title="seed for init datasets and model")
    n_step: int = Field(default=25, title="number of steps")
    n_dataset: int = Field(
        default=1000, title="number of evaluation datasets to generate"
    )
    timeout: float = Field(
        default=30.0, title="timeout for one evaluation step in seconds"
    )

    gumbel_tau: float = Field(default=1.0, title="gumbel sampling temperature")
    damage_factor: float = Field(
        default=0.1, title="how much to weigh damage contributions in gumbel sampling"
    )

    llm: LLMConfig
    dataset_gen_low: DatasetConfig = Field(
        default=DATASET_GEN_DEFAULT_LOW, title="dataset generation lower bound"
    )
    dataset_gen_high: DatasetConfig = Field(
        default=DATASET_GEN_DEFAULT_HIGH, title="dataset generation upper bound"
    )

    prompt: PromptConfig
