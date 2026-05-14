import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from backend.app.agents.prompts import advantage_prompt, debater_prompt, rebuttal_prompt, single_prompt, synthesis_prompt
from backend.app.core.metrics import (
    count_arguments,
    estimate_tokens,
    section_diversity,
    split_argument_sections,
)
from backend.app.core.ollama import OllamaClient
from backend.app.models import (
    AdvantageAnalysis,
    AdvantageAnnotation,
    AgentOutput,
    GenerationRequest,
    GenerationResult,
    Metrics,
    Source,
    TranscriptItem,
)
from backend.app.tools.search import SearchError, TavilySearch


EventCallback = Callable[[str, dict], Awaitable[None]]


CACHE_VERSION = "v7-advantage-annotations"
ROLE_ORDER = ["pro_logic", "pro_data", "con_logic", "con_data"]
ROLE_SIDE = {
    "pro_logic": "pro",
    "pro_data": "pro",
    "con_logic": "con",
    "con_data": "con",
}


class DebateOrchestrator:
    def __init__(self, llm: OllamaClient, search: TavilySearch):
        self.llm = llm
        self.search = search
        self.result_cache_path = search.cache_dir / "generation_cache.json"

    async def generate(self, request: GenerationRequest, emit: EventCallback | None = None) -> GenerationResult:
        started = time.perf_counter()
        emit = emit or _noop_emit
        cached = self._read_generation_cache(request) if request.use_cache else None
        if cached:
            await emit("stage", {"stage": "cache", "message": "Loaded cached generation result"})
            result = GenerationResult(**cached)
            await self._emit_cached_result(result, emit)
            return result

        await emit("stage", {"stage": "search", "message": "Collecting search context"})
        sources = await self._safe_search(request.topic, request.use_search, request.use_cache)

        single_task = asyncio.create_task(self._run_single(request, sources, emit))
        adversarial_task = asyncio.create_task(self._run_adversarial(request, sources, emit))
        single_agent, adversarial, transcript, adversarial_duration = await _gather_strategies(
            single_task,
            adversarial_task,
        )
        await emit("stage", {"stage": "advantage", "message": "Annotating multi-agent advantages"})
        advantage_analysis, advantage_tokens = await self._run_advantage_analysis(
            request,
            single_agent,
            adversarial,
            transcript,
        )

        total_duration = round(time.perf_counter() - started, 3)
        metrics = Metrics(
            total_duration_sec=total_duration,
            single_duration_sec=single_agent.duration_sec,
            adversarial_duration_sec=adversarial_duration,
            token_estimate=single_agent.token_estimate
            + adversarial.token_estimate
            + advantage_tokens
            + sum(estimate_tokens(item.content) for item in transcript),
            source_count=len({source.url for source in sources}),
            single_argument_count=count_arguments(single_agent.content),
            adversarial_argument_count=count_arguments(adversarial.content),
            single_diversity=section_diversity(split_argument_sections(single_agent.content)),
            adversarial_diversity=section_diversity(split_argument_sections(adversarial.content)),
        )
        result = GenerationResult(
            topic=request.topic,
            target_side=request.target_side,
            language=request.language,
            single_agent=single_agent,
            adversarial=adversarial,
            transcript=transcript,
            sources=sources,
            metrics=metrics,
            advantage_analysis=advantage_analysis,
        )
        if request.use_cache:
            self._write_generation_cache(request, result)
        await emit("final", result.model_dump())
        return result

    async def _safe_search(self, topic: str, use_search: bool, use_cache: bool) -> list[Source]:
        if not use_search:
            return []
        try:
            return await self.search.search(topic, max_results=5, use_cache=use_cache)
        except SearchError:
            return []

    async def _run_advantage_analysis(
        self,
        request: GenerationRequest,
        single_agent: AgentOutput,
        adversarial: AgentOutput,
        transcript: list[TranscriptItem],
    ) -> tuple[AdvantageAnalysis, int]:
        transcript_text = "\n\n".join(
            f"{item.stage} | {item.agent} | {item.side}\n{item.content}" for item in transcript
        )
        sentences = _split_sentences(adversarial.content)
        if not sentences:
            return AdvantageAnalysis(), 0

        try:
            response = await self.llm.chat(
                advantage_prompt(
                    request.topic,
                    request.target_side,
                    request.language,
                    single_agent.content,
                    adversarial.content,
                    transcript_text,
                ),
                temperature=0.25,
                max_tokens=520,
                thinking=False,
            )
            notes = _parse_advantage_notes(response.content)
            return _build_advantage_analysis(sentences, notes, request, transcript), response.token_estimate
        except Exception:
            return _build_advantage_analysis(sentences, {}, request, transcript), 0

    async def _emit_cached_result(self, result: GenerationResult, emit: EventCallback) -> None:
        cached_sources = [source.model_dump() for source in result.sources]
        await emit("stage", {"stage": "single_agent", "message": "Loaded cached single-agent strategy"})
        await emit(
            "panel_append",
            {
                "panel": "single",
                "role": "single_agent",
                "stage": "single_agent",
                "content": result.single_agent.content,
            },
        )
        await emit("agent", result.single_agent.model_dump())

        stage_messages = {
            "round_1": "Loaded cached Round 1 constructive arguments",
            "round_2": "Loaded cached Round 2 targeted rebuttals",
            "synthesis": "Loaded cached synthesis",
        }
        emitted_stages: set[str] = set()
        for item in result.transcript:
            if item.stage not in emitted_stages:
                await emit("stage", {"stage": item.stage, "message": stage_messages.get(item.stage, f"Loaded cached {item.stage}")})
                emitted_stages.add(item.stage)

            role = "adversarial_synthesis" if item.stage == "synthesis" else item.agent
            await emit(
                "panel_append",
                {
                    "panel": "adversarial",
                    "role": role,
                    "stage": item.stage,
                    "content": item.content,
                },
            )
            await emit(
                "agent",
                {
                    "stage": item.stage,
                    "role": role,
                    "side": item.side,
                    "content": item.content,
                    "sources": cached_sources if item.stage in {"round_1", "synthesis"} else [],
                    "duration_sec": item.duration_sec,
                    "token_estimate": estimate_tokens(item.content),
                },
            )

        if not any(item.stage == "synthesis" for item in result.transcript):
            await emit("stage", {"stage": "synthesis", "message": "Loaded cached synthesis"})
            await emit(
                "panel_append",
                {
                    "panel": "adversarial",
                    "role": "adversarial_synthesis",
                    "stage": "synthesis",
                    "content": result.adversarial.content,
                },
            )
            await emit("agent", {"stage": "synthesis", **result.adversarial.model_dump()})

        await emit("final", result.model_dump())

    async def _run_single(self, request: GenerationRequest, sources: list[Source], emit: EventCallback) -> AgentOutput:
        await emit("stage", {"stage": "single_agent", "message": "Running direct single-agent strategy"})
        await emit("agent_start", {"panel": "single", "stage": "single_agent", "role": "single_agent", "side": request.target_side})

        async def on_token(token: str) -> None:
            await emit("token", {"panel": "single", "stage": "single_agent", "role": "single_agent", "token": token})

        response = await self.llm.chat(
            single_prompt(request.topic, request.target_side, request.language, sources),
            temperature=0.55,
            max_tokens=780,
            thinking=False,
            on_token=on_token,
        )
        output = AgentOutput(
            role="single_agent",
            side=request.target_side,
            content=response.content,
            sources=sources,
            duration_sec=response.duration_sec,
            token_estimate=response.token_estimate,
        )
        await emit("agent_done", {"panel": "single", "stage": "single_agent", **output.model_dump()})
        await emit("agent", output.model_dump())
        return output

    async def _run_adversarial(
        self, request: GenerationRequest, sources: list[Source], emit: EventCallback
    ) -> tuple[AgentOutput, list[TranscriptItem], float]:
        started = time.perf_counter()
        transcript: list[TranscriptItem] = []
        await emit("stage", {"stage": "round_1", "message": "Round 1: constructive arguments"})

        round1 = await asyncio.gather(
            *(self._run_debater(role, request, sources, emit, "round_1") for role in ROLE_ORDER)
        )
        constructive_by_role = {role: output for role, output in zip(ROLE_ORDER, round1, strict=True)}
        transcript.extend(_to_transcript("round_1", ROLE_ORDER, round1))

        await emit("stage", {"stage": "round_2", "message": "Round 2: targeted rebuttals"})
        round2 = await asyncio.gather(
            *(
                self._run_rebuttal(role, request, constructive_by_role, emit)
                for role in ROLE_ORDER
            )
        )
        transcript.extend(_to_transcript("round_2", ROLE_ORDER, round2))

        await emit("stage", {"stage": "synthesis", "message": "Synthesising strongest target-side arguments"})
        transcript_text = "\n\n".join(
            f"{item.stage} | {item.agent} | {item.side}\n{item.content}" for item in transcript
        )
        await emit("agent_start", {"panel": "adversarial", "stage": "synthesis", "role": "adversarial_synthesis", "side": "neutral"})
        response = await self.llm.chat(
            synthesis_prompt(request.topic, request.target_side, request.language, transcript_text),
            temperature=0.45,
            max_tokens=900,
            thinking=True,
            on_token=self._token_emitter(emit, "adversarial", "synthesis", "adversarial_synthesis"),
        )
        output = AgentOutput(
            role="adversarial_synthesis",
            side=request.target_side,
            content=response.content,
            sources=sources,
            duration_sec=response.duration_sec,
            token_estimate=response.token_estimate,
        )
        transcript.append(
            TranscriptItem(
                stage="synthesis",
                agent="synthesis",
                side="neutral",
                content=response.content,
                duration_sec=response.duration_sec,
            )
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "synthesis", **output.model_dump()})
        await emit("agent", output.model_dump())
        return output, transcript, round(time.perf_counter() - started, 3)

    async def _run_debater(
        self,
        role: str,
        request: GenerationRequest,
        sources: list[Source],
        emit: EventCallback,
        stage: str,
    ) -> AgentOutput:
        await emit("agent_start", {"panel": "adversarial", "stage": stage, "role": role, "side": ROLE_SIDE[role]})
        response = await self.llm.chat(
            debater_prompt(role, request.topic, request.language, sources),
            temperature=0.7,
            max_tokens=360,
            thinking=True,
            on_token=self._token_emitter(emit, "adversarial", stage, role),
        )
        output = AgentOutput(
            role=role,
            side=ROLE_SIDE[role],  # type: ignore[arg-type]
            content=response.content,
            sources=sources,
            duration_sec=response.duration_sec,
            token_estimate=response.token_estimate,
        )
        await emit("agent_done", {"panel": "adversarial", "stage": stage, **output.model_dump()})
        await emit("agent", {"stage": stage, **output.model_dump()})
        return output

    async def _run_rebuttal(
        self,
        role: str,
        request: GenerationRequest,
        constructive_by_role: dict[str, AgentOutput],
        emit: EventCallback,
    ) -> AgentOutput:
        own = constructive_by_role[role].content
        opposing = "\n\n".join(
            constructive_by_role[other].content for other in ROLE_ORDER if ROLE_SIDE[other] != ROLE_SIDE[role]
        )
        await emit("agent_start", {"panel": "adversarial", "stage": "round_2", "role": role, "side": ROLE_SIDE[role]})
        response = await self.llm.chat(
            rebuttal_prompt(role, request.topic, request.language, own, opposing),
            temperature=0.65,
            max_tokens=360,
            thinking=True,
            on_token=self._token_emitter(emit, "adversarial", "round_2", role),
        )
        output = AgentOutput(
            role=role,
            side=ROLE_SIDE[role],  # type: ignore[arg-type]
            content=response.content,
            sources=[],
            duration_sec=response.duration_sec,
            token_estimate=response.token_estimate,
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "round_2", **output.model_dump()})
        await emit("agent", {"stage": "round_2", **output.model_dump()})
        return output

    @staticmethod
    def _token_emitter(emit: EventCallback, panel: str, stage: str, role: str):
        async def on_token(token: str) -> None:
            await emit("token", {"panel": panel, "stage": stage, "role": role, "token": token})

        return on_token

    def _read_generation_cache(self, request: GenerationRequest) -> dict | None:
        cache = self._load_generation_cache()
        return cache.get(self._cache_key(request))

    def _write_generation_cache(self, request: GenerationRequest, result: GenerationResult) -> None:
        cache = self._load_generation_cache()
        cache[self._cache_key(request)] = result.model_dump()
        self.result_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_generation_cache(self) -> dict[str, dict]:
        if not self.result_cache_path.exists():
            return {}
        try:
            return json.loads(self.result_cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _cache_key(request: GenerationRequest) -> str:
        payload = {"version": CACHE_VERSION, **request.model_dump(exclude={"use_cache"})}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _split_sentences(text: str) -> list[str]:
    matches = re.findall(r"[^。！？!?\.]+[。！？!?\.]?", text or "")
    sentences = [match.strip() for match in matches if match.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _parse_advantage_notes(content: str) -> dict[str, list[str]]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return {
        "rebuttal_notes": _string_list(data.get("rebuttal_notes")),
        "advantage_notes": _string_list(data.get("advantage_notes")),
    }


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_advantage_analysis(
    sentences: list[str],
    notes: dict[str, list[str]],
    request: GenerationRequest,
    transcript: list[TranscriptItem],
) -> AdvantageAnalysis:
    rebuttal_notes = notes.get("rebuttal_notes", [])
    advantage_notes = notes.get("advantage_notes", [])
    fallback_rebuttals = _fallback_rebuttal_notes(request, transcript)
    fallback_advantages = _fallback_advantage_notes(request.language)

    annotations = []
    for index, sentence in enumerate(sentences):
        annotations.append(
            AdvantageAnnotation(
                sentence=sentence,
                rebuttal_note=_pick_note(rebuttal_notes, index, fallback_rebuttals),
                advantage_note=_pick_note(advantage_notes, index, fallback_advantages),
            )
        )
    return AdvantageAnalysis(annotations=annotations)


def _pick_note(primary: list[str], index: int, fallback: list[str]) -> str:
    if index < len(primary) and primary[index]:
        return primary[index]
    if fallback:
        return fallback[index % len(fallback)]
    return ""


def _fallback_rebuttal_notes(request: GenerationRequest, transcript: list[TranscriptItem]) -> list[str]:
    opposing = [
        item.content
        for item in transcript
        if item.stage == "round_2" and item.side not in {request.target_side, "neutral"}
    ]
    if request.language == "zh":
        if not opposing:
            return ["回应潜在反方质疑"]
        return [f"回应反方：{_short_note(text, 18)}" for text in opposing]
    if not opposing:
        return ["answers a likely objection"]
    return [f"answers: {_short_note(text, 42)}" for text in opposing]


def _fallback_advantage_notes(language: str) -> list[str]:
    if language == "zh":
        return ["主张更精确", "因果链更完整", "例子更贴题", "反驳意识更强"]
    return ["sharper claim", "fuller causal chain", "better-fitted example", "stronger rebuttal work"]


def _short_note(text: str, limit: int) -> str:
    sentence = _split_sentences(text)[0] if _split_sentences(text) else text
    compact = re.sub(r"\s+", " ", sentence).strip()
    return compact[:limit].rstrip("，,。.")


async def _noop_emit(_event: str, _payload: dict) -> None:
    return None


async def _gather_strategies(single_task, adversarial_task):
    tasks = [single_task, adversarial_task]
    try:
        single_agent, adversarial_bundle = await asyncio.gather(*tasks)
        adversarial, transcript, adversarial_duration = adversarial_bundle
        return single_agent, adversarial, transcript, adversarial_duration
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _to_transcript(stage: str, roles: list[str], outputs: list[AgentOutput]) -> list[TranscriptItem]:
    return [
        TranscriptItem(
            stage=stage,
            agent=role,
            side=output.side,
            content=output.content,
            duration_sec=output.duration_sec,
        )
        for role, output in zip(roles, outputs, strict=True)
    ]
