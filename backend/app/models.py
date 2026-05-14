from typing import Any, Literal

from pydantic import BaseModel, Field


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
    topic: str = Field(min_length=2)
    target_side: Side = "pro"
    language: Language = "en"
    use_search: bool = True
    use_cache: bool = True


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""


class AgentOutput(BaseModel):
    role: str
    side: Side | Literal["neutral"]
    content: str
    sources: list[Source] = Field(default_factory=list)
    duration_sec: float = 0.0
    token_estimate: int = 0


class TranscriptItem(BaseModel):
    stage: str
    agent: str
    side: Side | Literal["neutral"]
    content: str
    duration_sec: float


class Metrics(BaseModel):
    total_duration_sec: float
    single_duration_sec: float
    adversarial_duration_sec: float
    token_estimate: int
    source_count: int
    single_argument_count: int
    adversarial_argument_count: int
    single_diversity: float
    adversarial_diversity: float


class AdvantageAnnotation(BaseModel):
    sentence: str
    rebuttal_note: str = ""
    advantage_note: str = ""


class AdvantageAnalysis(BaseModel):
    annotations: list[AdvantageAnnotation] = Field(default_factory=list)


class GenerationResult(BaseModel):
    topic: str
    target_side: Side
    language: Language
    single_agent: AgentOutput
    adversarial: AgentOutput
    transcript: list[TranscriptItem]
    sources: list[Source]
    metrics: Metrics
    advantage_analysis: AdvantageAnalysis = Field(default_factory=AdvantageAnalysis)


class StreamEvent(BaseModel):
    event: str
    payload: dict[str, Any]
