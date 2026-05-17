from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Language = Literal["en", "zh"]
Side = Literal["pro", "con"]


class Topic(BaseModel):
    id: str
    domain: str
    topic_zh: str
    topic_en: str
    pro_label: str
    con_label: str
    source_note: str


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str = Field(min_length=1)
    target_side: Side = "pro"
    language: Language = "en"
    use_cache: bool = True
    side_claim: str | None = Field(
        default=None,
        description="Explicit claim for target_side on this motion (e.g. topic pro_label/con_label).",
    )


class DebateArgument(BaseModel):
    reason: str = ""
    logic_chain: str = ""


class AgentOutput(BaseModel):
    role: str
    side: Side | Literal["neutral"]
    content: str
    argument: DebateArgument = Field(default_factory=DebateArgument)
    duration_sec: float = 0.0
    token_estimate: int = 0


class TranscriptItem(BaseModel):
    stage: str
    agent: str
    side: Side | Literal["neutral"]
    content: str
    duration_sec: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Metrics(BaseModel):
    total_duration_sec: float
    single_duration_sec: float
    adversarial_duration_sec: float
    token_estimate: int
    single_argument_count: int = 1
    adversarial_argument_count: int = 1
    candidate_count: int = 0
    eliminated_count: int = 0
    optimized_count: int = 0
    scoring_agent_count: int = 0
    final_average_score: float = 0.0
    latency_cost_ratio: float = 0.0


class GenerationResult(BaseModel):
    topic: str
    target_side: Side
    language: Language
    single_agent: AgentOutput
    adversarial: AgentOutput
    transcript: list[TranscriptItem]
    metrics: Metrics


class StreamEvent(BaseModel):
    event: str
    payload: dict[str, Any]


class EvaluationRequest(BaseModel):
    topic: str = Field(min_length=1)
    target_side: Side = "pro"
    language: Language = "en"
    single_content: str = Field(min_length=1)
    adversarial_content: str = Field(min_length=1)


class EvaluationResult(BaseModel):
    single_scores: dict[str, int]
    adversarial_scores: dict[str, int]
    single_total: float
    adversarial_total: float
    dimension_reasoning: dict[str, dict[str, str]]
    winner: str
    winner_reasoning: str
    total_duration_sec: float
    token_usage: dict[str, int]
