import asyncio
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from backend.app.agents.prompts import (
    ZH_LOGIC_CHAIN_MAX_CHARS,
    ZH_REASON_MAX_CHARS,
    candidate_generation_prompt,
    challenge_prompt,
    optimization_prompt,
    scoring_prompt,
    single_prompt,
)
from backend.app.core.metrics import estimate_tokens
from backend.app.core.ollama import OllamaClient
from backend.app.models import (
    AgentOutput,
    DebateArgument,
    GenerationRequest,
    GenerationResult,
    Metrics,
    Side,
    TranscriptItem,
)


EventCallback = Callable[[str, dict], Awaitable[None]]


CACHE_VERSION = "v14-single-aligned-prompts"
OPPOSITION_AGENT_COUNT = 5
SCORING_AGENT_COUNT = 5


class DebateOrchestrator:
    def __init__(self, llm: OllamaClient, cache_dir: Path):
        self.llm = llm
        self.result_cache_path = cache_dir / "generation_cache.json"

    async def generate(self, request: GenerationRequest, emit: EventCallback | None = None) -> GenerationResult:
        started = time.perf_counter()
        emit = emit or _noop_emit
        cached = self._read_generation_cache(request) if request.use_cache else None
        if cached:
            await emit("stage", {"stage": "cache", "message": "Loaded cached generation result"})
            result = GenerationResult(**cached)
            await self._emit_cached_result(result, emit)
            return result

        single_task = asyncio.create_task(self._run_single(request, emit))
        multi_task = asyncio.create_task(self._run_multi_agent(request, emit))
        single_agent, multi_agent, transcript, multi_duration = await _gather_strategies(single_task, multi_task)

        metadata = _multi_metadata(transcript)
        total_duration = round(time.perf_counter() - started, 3)
        metrics = Metrics(
            total_duration_sec=total_duration,
            single_duration_sec=single_agent.duration_sec,
            adversarial_duration_sec=multi_duration,
            token_estimate=single_agent.token_estimate
            + multi_agent.token_estimate
            + sum(estimate_tokens(item.content) for item in transcript),
            single_argument_count=1,
            adversarial_argument_count=1,
            candidate_count=len(metadata.get("candidates", [])),
            eliminated_count=1 if metadata.get("eliminated_id") else 0,
            optimized_count=len(metadata.get("optimized_ids", [])),
            scoring_agent_count=len(metadata.get("scorecards", [])),
            final_average_score=round(float(metadata.get("final_average_score") or 0.0), 3),
            latency_cost_ratio=_latency_cost_ratio(single_agent.duration_sec, multi_duration),
        )
        result = GenerationResult(
            topic=request.topic,
            target_side=request.target_side,
            language=request.language,
            single_agent=single_agent,
            adversarial=multi_agent,
            transcript=transcript,
            metrics=metrics,
        )
        if request.use_cache:
            self._write_generation_cache(request, result)
        await emit("final", result.model_dump())
        return result

    async def _emit_cached_result(self, result: GenerationResult, emit: EventCallback) -> None:
        await emit("stage", {"stage": "single_agent", "message": "Loaded cached single-agent result"})
        await emit("agent", {"stage": "single_agent", **result.single_agent.model_dump()})

        emitted_stages: set[str] = set()
        for item in result.transcript:
            if item.stage not in emitted_stages:
                await emit("stage", {"stage": item.stage, "message": _stage_message(item.stage, cached=True)})
                emitted_stages.add(item.stage)
            await emit(
                "panel_append",
                {
                    "panel": "adversarial",
                    "role": item.agent,
                    "stage": item.stage,
                    "content": item.content,
                },
            )
            await emit(
                "agent",
                {
                    "stage": item.stage,
                    "role": item.agent,
                    "side": item.side,
                    "content": item.content,
                    "duration_sec": item.duration_sec,
                    "token_estimate": estimate_tokens(item.content),
                    "metadata": item.metadata,
                },
            )

        await emit("stage", {"stage": "final", "message": "Loaded cached multi-agent final result"})
        await emit("agent", {"stage": "final", **result.adversarial.model_dump()})
        await emit("final", result.model_dump())

    async def _run_single(self, request: GenerationRequest, emit: EventCallback) -> AgentOutput:
        await emit("stage", {"stage": "single_agent", "message": "Running direct single-agent logic"})
        await emit(
            "agent_start",
            {"panel": "single", "stage": "single_agent", "role": "single_agent", "side": request.target_side},
        )
        response = await self.llm.chat(
            single_prompt(request.topic, request.target_side, request.language, request.side_claim),
            temperature=0.45,
            max_tokens=700,
            thinking=False,
            on_token=self._token_emitter(emit, "single", "single_agent", "single_agent"),
        )
        try:
            argument = _parse_argument(response.content, request.language)
            parse_warning = ""
        except Exception as exc:
            argument = _fallback_argument(response.content, request.language)
            parse_warning = str(exc)
        content = _format_argument(argument, request.language)
        output = AgentOutput(
            role="single_agent",
            side=request.target_side,
            content=content,
            argument=argument,
            duration_sec=response.duration_sec,
            token_estimate=response.token_estimate,
        )
        await emit("agent_done", {"panel": "single", "stage": "single_agent", **output.model_dump()})
        await emit("agent", {"stage": "single_agent", **output.model_dump()})
        return output

    async def _run_multi_agent(
        self, request: GenerationRequest, emit: EventCallback
    ) -> tuple[AgentOutput, list[TranscriptItem], float]:
        started = time.perf_counter()
        transcript: list[TranscriptItem] = []

        candidates_item = await self._run_candidate_generation(request, emit)
        transcript.append(candidates_item)
        candidates = candidates_item.metadata["candidates"]

        challenge_items = await self._run_challenges(request, candidates, emit)
        transcript.extend(challenge_items)
        challenges = [item.metadata["challenge"] for item in challenge_items]

        elimination = _select_elimination(challenges)
        elimination_content = _format_elimination(elimination, request.language)
        elimination_item = TranscriptItem(
            stage="elimination",
            agent="vote_counter",
            side="neutral",
            content=elimination_content,
            duration_sec=0.0,
            metadata=elimination,
        )
        transcript.append(elimination_item)
        await emit("stage", {"stage": "elimination", "message": _stage_message("elimination")})
        await self._emit_panel_item(elimination_item, emit)

        current_candidates, optimization_items = await self._run_optimizers(
            request,
            candidates,
            challenges,
            elimination.get("eliminated_id"),
            emit,
        )
        transcript.extend(optimization_items)

        scoring_items = await self._run_scorers(request, current_candidates, emit)
        transcript.extend(scoring_items)
        scorecards = [item.metadata["scores"] for item in scoring_items]

        selection = _select_final_candidate(current_candidates, scorecards)
        final_argument = DebateArgument(**selection["argument"])
        final_content = _format_argument(final_argument, request.language)
        selection_content = _format_final_selection(selection, request.language)
        selection_item = TranscriptItem(
            stage="final_selection",
            agent="score_aggregator",
            side="neutral",
            content=selection_content,
            duration_sec=0.0,
            metadata={
                **selection,
                "candidates": candidates,
                "optimized_ids": [
                    item.metadata["candidate_id"]
                    for item in optimization_items
                    if "candidate_id" in item.metadata
                ],
                "scorecards": scorecards,
                "eliminated_id": elimination.get("eliminated_id"),
            },
        )
        transcript.append(selection_item)
        await emit("stage", {"stage": "final_selection", "message": _stage_message("final_selection")})
        await self._emit_panel_item(selection_item, emit)

        output = AgentOutput(
            role="multi_agent_final",
            side=request.target_side,
            content=final_content,
            argument=final_argument,
            duration_sec=round(time.perf_counter() - started, 3),
            token_estimate=estimate_tokens(final_content),
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "final", **output.model_dump()})
        await emit("agent", {"stage": "final", **output.model_dump()})
        return output, transcript, output.duration_sec

    async def _run_candidate_generation(self, request: GenerationRequest, emit: EventCallback) -> TranscriptItem:
        await emit("stage", {"stage": "round_1", "message": _stage_message("round_1")})
        await emit(
            "agent_start",
            {"panel": "adversarial", "stage": "round_1", "role": "argument_generator", "side": request.target_side},
        )
        response = await self.llm.chat(
            candidate_generation_prompt(request.topic, request.target_side, request.language, request.side_claim),
            temperature=0.7,
            max_tokens=1500,
            thinking=False,
            on_token=self._token_emitter(emit, "adversarial", "round_1", "argument_generator"),
        )
        try:
            candidates = _parse_candidates(response.content, request.language)
            parse_warning = ""
        except Exception as exc:
            candidates = _fallback_candidates(response.content, request.language)
            parse_warning = str(exc)
        content = _format_candidates(candidates, request.language)
        item = TranscriptItem(
            stage="round_1",
            agent="argument_generator",
            side=request.target_side,
            content=content,
            duration_sec=response.duration_sec,
            metadata={"candidates": candidates, "token_estimate": response.token_estimate, "parse_warning": parse_warning},
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "round_1", "role": item.agent, "side": item.side, "content": content})
        await self._emit_agent_item(item, emit)
        return item

    async def _run_challenges(
        self, request: GenerationRequest, candidates: list[dict[str, str]], emit: EventCallback
    ) -> list[TranscriptItem]:
        await emit("stage", {"stage": "round_2", "message": _stage_message("round_2")})
        tasks = [
            self._run_challenge_agent(index, request, candidates, emit, random.Random(index))
            for index in range(1, OPPOSITION_AGENT_COUNT + 1)
        ]
        return await asyncio.gather(*tasks)

    async def _run_challenge_agent(
        self,
        agent_index: int,
        request: GenerationRequest,
        candidates: list[dict[str, str]],
        emit: EventCallback,
        rng: random.Random | None = None,
    ) -> TranscriptItem:
        rng = rng or random.Random()
        shuffled = candidates.copy()
        rng.shuffle(shuffled)

        role = f"opposition_{agent_index}"
        side = _opposite_side(request.target_side)
        await emit("agent_start", {"panel": "adversarial", "stage": "round_2", "role": role, "side": side})
        response = await self.llm.chat(
            challenge_prompt(agent_index, request.topic, request.target_side, request.language, shuffled),
            temperature=0.65,
            max_tokens=650,
            thinking=False,
            on_token=self._token_emitter(emit, "adversarial", "round_2", role),
        )
        try:
            challenge = _parse_challenge(response.content, {candidate["id"] for candidate in candidates})
            parse_warning = ""
        except Exception as exc:
            challenge = _fallback_challenge(agent_index, candidates, response.content, request.language)
            parse_warning = str(exc)
        content = _format_challenge(challenge, request.language)
        item = TranscriptItem(
            stage="round_2",
            agent=role,
            side=side,
            content=content,
            duration_sec=response.duration_sec,
            metadata={"challenge": challenge, "token_estimate": response.token_estimate, "parse_warning": parse_warning},
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "round_2", "role": role, "side": side, "content": content})
        await self._emit_agent_item(item, emit)
        return item

    async def _run_optimizers(
        self,
        request: GenerationRequest,
        candidates: list[dict[str, str]],
        challenges: list[dict[str, str]],
        eliminated_id: str | None,
        emit: EventCallback,
    ) -> tuple[list[dict[str, str]], list[TranscriptItem]]:
        await emit("stage", {"stage": "round_3", "message": _stage_message("round_3")})
        challenge_map: dict[str, list[dict[str, str]]] = defaultdict(list)
        for challenge in challenges:
            challenge_map[challenge["target_id"]].append(challenge)

        survivors = [candidate for candidate in candidates if candidate["id"] != eliminated_id]
        optimized_tasks = [
            self._run_optimizer(candidate, challenge_map[candidate["id"]], request, emit)
            for candidate in survivors
            if challenge_map[candidate["id"]]
        ]
        optimized_items = await asyncio.gather(*optimized_tasks) if optimized_tasks else []
        optimized_by_id = {
            item.metadata["candidate_id"]: item.metadata["candidate"]
            for item in optimized_items
        }
        current_candidates = [optimized_by_id.get(candidate["id"], candidate) for candidate in survivors]
        pool_item = TranscriptItem(
            stage="round_3",
            agent="optimized_pool",
            side=request.target_side,
            content=_format_candidates(current_candidates, request.language),
            duration_sec=0.0,
            metadata={"candidates": current_candidates},
        )
        await self._emit_panel_item(pool_item, emit)
        return current_candidates, [*optimized_items, pool_item]

    async def _run_optimizer(
        self,
        candidate: dict[str, str],
        challenges: list[dict[str, str]],
        request: GenerationRequest,
        emit: EventCallback,
    ) -> TranscriptItem:
        role = f"optimizer_{candidate['id']}"
        await emit("agent_start", {"panel": "adversarial", "stage": "round_3", "role": role, "side": request.target_side})
        response = await self.llm.chat(
            optimization_prompt(
                request.topic,
                request.target_side,
                request.language,
                candidate,
                challenges,
                request.side_claim,
            ),
            temperature=0.55,
            max_tokens=900,
            thinking=False,
            on_token=self._token_emitter(emit, "adversarial", "round_3", role),
        )
        try:
            argument = _parse_argument(response.content, request.language)
            parse_warning = ""
        except Exception as exc:
            argument = _fallback_optimized_argument(candidate, challenges, response.content, request.language)
            parse_warning = str(exc)
        optimized = {"id": candidate["id"], **argument.model_dump()}
        content = _format_candidate(optimized, request.language)
        item = TranscriptItem(
            stage="round_3",
            agent=role,
            side=request.target_side,
            content=content,
            duration_sec=response.duration_sec,
            metadata={
                "candidate_id": candidate["id"],
                "candidate": optimized,
                "challenge_count": len(challenges),
                "token_estimate": response.token_estimate,
                "parse_warning": parse_warning,
            },
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "round_3", "role": role, "side": request.target_side, "content": content})
        await self._emit_agent_item(item, emit)
        return item

    async def _run_scorers(
        self, request: GenerationRequest, candidates: list[dict[str, str]], emit: EventCallback
    ) -> list[TranscriptItem]:
        await emit("stage", {"stage": "round_4", "message": _stage_message("round_4")})
        tasks = [
            self._run_scoring_agent(index, request, candidates, emit, random.Random(index + 100))
            for index in range(1, SCORING_AGENT_COUNT + 1)
        ]
        return await asyncio.gather(*tasks)

    async def _run_scoring_agent(
        self,
        agent_index: int,
        request: GenerationRequest,
        candidates: list[dict[str, str]],
        emit: EventCallback,
        rng: random.Random | None = None,
    ) -> TranscriptItem:
        rng = rng or random.Random()
        shuffled = candidates.copy()
        rng.shuffle(shuffled)

        role = f"scoring_{agent_index}"
        await emit("agent_start", {"panel": "adversarial", "stage": "round_4", "role": role, "side": "neutral"})
        response = await self.llm.chat(
            scoring_prompt(agent_index, request.topic, request.target_side, request.language, shuffled),
            temperature=0.35,
            max_tokens=900,
            thinking=False,
            on_token=self._token_emitter(emit, "adversarial", "round_4", role),
        )
        try:
            scores = _parse_scores(response.content, {candidate["id"] for candidate in candidates})
            parse_warning = ""
        except Exception as exc:
            scores = _fallback_scores(candidates, agent_index, response.content)
            parse_warning = str(exc)
        content = _format_scores(scores, request.language)
        item = TranscriptItem(
            stage="round_4",
            agent=role,
            side="neutral",
            content=content,
            duration_sec=response.duration_sec,
            metadata={"scores": scores, "token_estimate": response.token_estimate, "parse_warning": parse_warning},
        )
        await emit("agent_done", {"panel": "adversarial", "stage": "round_4", "role": role, "side": "neutral", "content": content})
        await self._emit_agent_item(item, emit)
        return item

    async def _emit_panel_item(self, item: TranscriptItem, emit: EventCallback) -> None:
        await emit(
            "panel_append",
            {
                "panel": "adversarial",
                "role": item.agent,
                "stage": item.stage,
                "content": item.content,
            },
        )
        await self._emit_agent_item(item, emit)

    async def _emit_agent_item(self, item: TranscriptItem, emit: EventCallback) -> None:
        await emit(
            "agent",
            {
                "stage": item.stage,
                "role": item.agent,
                "side": item.side,
                "content": item.content,
                "duration_sec": item.duration_sec,
                "token_estimate": item.metadata.get("token_estimate", estimate_tokens(item.content)),
                "metadata": item.metadata,
            },
        )

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


def _parse_argument(content: str, language: str = "en") -> DebateArgument:
    data = _extract_json(content)
    if not isinstance(data, dict):
        raise ValueError("Model output must be a JSON object.")
    argument = _argument_from_mapping(data)
    if not argument.reason.strip() or not argument.logic_chain.strip():
        raise ValueError("Model output must include non-empty reason and logic_chain.")
    return _normalize_argument(argument, language)


def _parse_candidates(content: str, language: str = "en") -> list[dict[str, str]]:
    try:
        data = _extract_json(content)
    except json.JSONDecodeError:
        return _parse_candidates_from_text(content, language)
    candidates = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        raise ValueError("Round 1 output must include a candidates list.")
    normalized = []
    for index, item in enumerate(candidates[:6], start=1):
        if not isinstance(item, dict):
            continue
        argument = _argument_from_mapping(item)
        candidate_id = _normalize_id(str(item.get("id") or f"R{index}"))
        if candidate_id not in {f"R{number}" for number in range(1, 7)}:
            candidate_id = f"R{index}"
        normalized.append(_normalize_candidate({"id": candidate_id, **argument.model_dump()}, language))
    normalized = _dedupe_by_id(normalized)
    if len(normalized) != 6:
        raise ValueError("Round 1 must return exactly 6 valid candidates.")
    return sorted(normalized, key=lambda item: _candidate_order(item["id"]))


def _parse_challenge(content: str, valid_ids: set[str]) -> dict[str, str]:
    try:
        data = _extract_json(content)
    except json.JSONDecodeError:
        data = _parse_object_fields(content, ["target_id", "question", "weakness_reason", "opposing_reason"])
    if not isinstance(data, dict):
        raise ValueError("Challenge output must be a JSON object.")
    target_id = _normalize_id(str(data.get("target_id") or ""))
    if target_id not in valid_ids:
        raise ValueError(f"Challenge target_id must be one of {sorted(valid_ids)}.")
    return {
        "target_id": target_id,
        "question": str(data.get("question") or "").strip(),
        "weakness_reason": str(data.get("weakness_reason") or "").strip(),
        "opposing_reason": str(data.get("opposing_reason") or "").strip(),
    }


def _parse_scores(content: str, valid_ids: set[str]) -> list[dict[str, Any]]:
    try:
        data = _extract_json(content)
    except json.JSONDecodeError:
        return _parse_scores_from_text(content, valid_ids)
    scores = data.get("scores") if isinstance(data, dict) else data
    if not isinstance(scores, list):
        raise ValueError("Scoring output must include a scores list.")
    by_id: dict[str, dict[str, Any]] = {}
    for item in scores:
        if not isinstance(item, dict):
            continue
        candidate_id = _normalize_id(str(item.get("id") or ""))
        if candidate_id not in valid_ids:
            continue
        score = _clamp_score(item.get("score"))
        by_id[candidate_id] = {
            "id": candidate_id,
            "score": score,
            "rationale": str(item.get("rationale") or "").strip(),
        }
    missing = valid_ids - set(by_id)
    if missing:
        raise ValueError(f"Scoring output missing ids: {sorted(missing)}.")
    return [by_id[candidate_id] for candidate_id in sorted(valid_ids, key=_candidate_order)]


def _extract_json(content: str) -> Any:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        starts = [index for index, char in enumerate(cleaned) if char in "[{"]
        last_error: json.JSONDecodeError | None = None
        for start in starts:
            try:
                value, _end = decoder.raw_decode(cleaned[start:])
                return value
            except json.JSONDecodeError as exc:
                last_error = exc
        candidate = _balanced_json_slice(cleaned)
        if candidate:
            return json.loads(candidate)
        if last_error:
            raise last_error
        raise


def _normalize_argument(argument: DebateArgument, language: str) -> DebateArgument:
    if language != "zh":
        return argument
    return DebateArgument(
        reason=_truncate_zh_text(argument.reason, ZH_REASON_MAX_CHARS),
        logic_chain=_truncate_zh_text(argument.logic_chain, ZH_LOGIC_CHAIN_MAX_CHARS),
    )


def _normalize_candidate(candidate: dict[str, str], language: str) -> dict[str, str]:
    normalized = _normalize_argument(
        DebateArgument(reason=candidate.get("reason", ""), logic_chain=candidate.get("logic_chain", "")),
        language,
    )
    return {**candidate, **normalized.model_dump()}


def _truncate_zh_text(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    for separator in ("。", "！", "？", "；", "，", ".", "!", "?", ";", ","):
        index = truncated.rfind(separator)
        if index >= int(max_chars * 0.6):
            return truncated[: index + 1].strip()
    return truncated.rstrip("，,、；; ")


def _argument_from_mapping(data: dict[str, Any]) -> DebateArgument:
    reason = data.get("reason") or data.get("理由") or data.get("claim") or data.get("Claim") or ""
    logic_chain = (
        data.get("logic_chain")
        or data.get("logicChain")
        or data.get("逻辑链条")
        or data.get("logic")
        or data.get("Logic chain")
        or ""
    )
    return DebateArgument(reason=str(reason).strip(), logic_chain=str(logic_chain).strip())


def _parse_object_fields(content: str, fields: list[str]) -> dict[str, str]:
    data: dict[str, str] = {}
    for field in fields:
        match = re.search(rf'"?{re.escape(field)}"?\s*[:：]\s*"?(.*?)(?=",?\s*"?(?:{"|".join(map(re.escape, fields))})"?\s*[:：]|\n\s*"?(?:{"|".join(map(re.escape, fields))})"?\s*[:：]|\Z)', content, flags=re.DOTALL)
        if match:
            data[field] = match.group(1).strip().strip('",， \n')
    return data


def _parse_candidates_from_text(content: str, language: str = "en") -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    sections = re.split(r"(?=R[1-6]\b|\"id\"\s*:\s*\"?R[1-6])", content or "")
    for section in sections:
        id_match = re.search(r"R([1-6])", section)
        if not id_match:
            continue
        candidate_id = f"R{id_match.group(1)}"
        reason = _extract_labeled_value(section, ["reason", "理由"])
        logic_chain = _extract_labeled_value(section, ["logic_chain", "logicChain", "逻辑链条"])
        if reason and logic_chain:
            candidates.append({"id": candidate_id, "reason": reason, "logic_chain": logic_chain})
    candidates = _dedupe_by_id(candidates)
    if len(candidates) != 6:
        raise ValueError("Could not recover exactly 6 candidates from model output.")
    return [
        _normalize_candidate(candidate, language)
        for candidate in sorted(candidates, key=lambda item: _candidate_order(item["id"]))
    ]


def _parse_scores_from_text(content: str, valid_ids: set[str]) -> list[dict[str, Any]]:
    scores = []
    for candidate_id in sorted(valid_ids, key=_candidate_order):
        match = re.search(rf"{re.escape(candidate_id)}\D*([0-5])", content or "")
        if match:
            scores.append({"id": candidate_id, "score": int(match.group(1)), "rationale": "Recovered from non-strict model output."})
    if len(scores) != len(valid_ids):
        raise ValueError("Could not recover scores from model output.")
    return scores


def _extract_labeled_value(text: str, labels: list[str]) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    stop_labels = "reason|理由|logic_chain|logicChain|逻辑链条|id"
    match = re.search(
        rf"(?:{label_pattern})\s*[\"'：:]*\s*(.*?)(?=\n\s*(?:{stop_labels})\s*[\"'：:]|\n\s*R[1-6]\b|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return match.group(1).strip().strip('",， \n')


def _balanced_json_slice(text: str) -> str:
    start = min([index for index in (text.find("{"), text.find("[")) if index >= 0], default=-1)
    if start < 0:
        return ""
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _fallback_argument(content: str, language: str) -> DebateArgument:
    reason = _extract_labeled_value(content, ["reason", "理由"]) or _first_sentence(content)
    logic_chain = _extract_labeled_value(content, ["logic_chain", "logicChain", "逻辑链条"])
    if not logic_chain:
        logic_chain = content.strip() or (
            "模型输出格式不稳定；该结果保留原始输出作为逻辑链条。" if language == "zh" else "The model output format was unstable; the raw output is preserved as the logic chain."
        )
    return _normalize_argument(DebateArgument(reason=reason, logic_chain=logic_chain), language)


def _fallback_candidates(content: str, language: str) -> list[dict[str, str]]:
    try:
        return _parse_candidates_from_text(content, language)
    except Exception:
        base = _fallback_argument(content, language)
        candidates = []
        for index in range(1, 7):
            suffix = f"（候选 {index}）" if language == "zh" else f" (candidate {index})"
            candidates.append(
                {
                    "id": f"R{index}",
                    "reason": f"{base.reason}{suffix}",
                    "logic_chain": base.logic_chain,
                }
            )
        return [_normalize_candidate(candidate, language) for candidate in candidates]


def _fallback_challenge(agent_index: int, candidates: list[dict[str, str]], content: str, language: str) -> dict[str, str]:
    target = min(candidates, key=lambda candidate: (len(candidate.get("logic_chain", "")), _candidate_order(candidate["id"])))
    if language == "zh":
        return {
            "target_id": target["id"],
            "question": "该逻辑链条的成立条件是否过强，是否把应被证明的前提直接当成结论？",
            "weakness_reason": "模型质询输出格式不稳定，系统改用最短逻辑链条作为保守质询对象。",
            "opposing_reason": content.strip()[:300] or "反方认为该理由尚未充分证明主体资格与责任能力之间的必然关系。",
        }
    return {
        "target_id": target["id"],
        "question": "Does this logic chain assume the very condition it needs to prove?",
        "weakness_reason": "The model challenge output was not strict JSON, so the system conservatively challenged the shortest logic chain.",
        "opposing_reason": content.strip()[:300] or "The opposing side says the link between legal personhood and responsibility remains under-proven.",
    }


def _fallback_optimized_argument(candidate: dict[str, str], challenges: list[dict[str, str]], content: str, language: str) -> DebateArgument:
    challenge_text = "；".join(challenge.get("weakness_reason", "") for challenge in challenges if challenge.get("weakness_reason"))
    if language == "zh":
        logic_chain = (
            f"{candidate['logic_chain']} 在吸收质询意见的同时，仍需以一段话连贯论证我方观点："
            f"该理由的成立需要相应边界条件，并回应对方质疑后仍支持我方立场。{content.strip()[:120]}"
        )
    else:
        logic_chain = (
            f"{candidate['logic_chain']} While absorbing the challenges, still argue in one continuous paragraph "
            f"for your side: clarify scope, answer objections, and show the claim still supports your position. "
            f"{content.strip()[:120]}"
        )
    if challenge_text:
        logic_chain += f" {challenge_text[:80]}"
    return _normalize_argument(DebateArgument(reason=candidate["reason"], logic_chain=logic_chain), language)


def _fallback_scores(candidates: list[dict[str, str]], agent_index: int, content: str) -> list[dict[str, Any]]:
    scores = []
    for candidate in candidates:
        length_bonus = 1 if len(candidate.get("logic_chain", "")) > 120 else 0
        score = min(5, 3 + length_bonus + ((agent_index + _candidate_order(candidate["id"])) % 2))
        scores.append(
            {
                "id": candidate["id"],
                "score": score,
                "rationale": (content.strip()[:80] or "Fallback score after non-strict model output."),
            }
        )
    return scores


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[。！？.!?])", cleaned)
    return (parts[0] if parts and parts[0].strip() else cleaned[:120]).strip()


def _select_elimination(challenges: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(challenge["target_id"] for challenge in challenges)
    if not counts:
        return {"vote_counts": {}, "eliminated_id": None, "tie": False}
    max_votes = max(counts.values())
    leaders = sorted(candidate_id for candidate_id, votes in counts.items() if votes == max_votes)
    eliminated_id = leaders[0] if len(leaders) == 1 else None
    return {
        "vote_counts": dict(sorted(counts.items(), key=lambda item: _candidate_order(item[0]))),
        "eliminated_id": eliminated_id,
        "tie": len(leaders) > 1,
        "leaders": leaders,
    }


def _select_final_candidate(candidates: list[dict[str, str]], scorecards: list[list[dict[str, Any]]]) -> dict[str, Any]:
    totals: dict[str, float] = {candidate["id"]: 0.0 for candidate in candidates}
    counts: dict[str, int] = {candidate["id"]: 0 for candidate in candidates}
    for scorecard in scorecards:
        for item in scorecard:
            candidate_id = item["id"]
            if candidate_id in totals:
                totals[candidate_id] += float(item["score"])
                counts[candidate_id] += 1
    averages = {
        candidate_id: round(totals[candidate_id] / counts[candidate_id], 4) if counts[candidate_id] else 0.0
        for candidate_id in totals
    }
    max_average = max(averages.values()) if averages else 0.0
    leaders = [candidate_id for candidate_id, average in averages.items() if average == max_average]
    selected_id = sorted(leaders, key=_candidate_order)[0]
    selected = next(candidate for candidate in candidates if candidate["id"] == selected_id)
    return {
        "selected_id": selected_id,
        "argument": {"reason": selected["reason"], "logic_chain": selected["logic_chain"]},
        "averages": dict(sorted(averages.items(), key=lambda item: _candidate_order(item[0]))),
        "final_average_score": max_average,
        "tie": len(leaders) > 1,
        "leaders": sorted(leaders, key=_candidate_order),
    }


def _format_argument(argument: DebateArgument, language: str) -> str:
    if language == "zh":
        return f"理由：{argument.reason}\n逻辑链条：{argument.logic_chain}"
    return f"Reason: {argument.reason}\nLogic chain: {argument.logic_chain}"


def _format_candidate(candidate: dict[str, str], language: str) -> str:
    argument = DebateArgument(reason=candidate["reason"], logic_chain=candidate["logic_chain"])
    if language == "zh":
        return f"{candidate['id']}\n{_format_argument(argument, language)}"
    return f"{candidate['id']}\n{_format_argument(argument, language)}"


def _format_candidates(candidates: list[dict[str, str]], language: str) -> str:
    return "\n\n".join(_format_candidate(candidate, language) for candidate in candidates)


def _format_challenge(challenge: dict[str, str], language: str) -> str:
    if language == "zh":
        return (
            f"投票对象：{challenge['target_id']}\n"
            f"质询：{challenge['question']}\n"
            f"薄弱原因：{challenge['weakness_reason']}\n"
            f"反方理由：{challenge['opposing_reason']}"
        )
    return (
        f"Target: {challenge['target_id']}\n"
        f"Question: {challenge['question']}\n"
        f"Weakness: {challenge['weakness_reason']}\n"
        f"Opposing reason: {challenge['opposing_reason']}"
    )


def _format_elimination(elimination: dict[str, Any], language: str) -> str:
    counts = ", ".join(f"{candidate_id}: {votes}" for candidate_id, votes in elimination.get("vote_counts", {}).items())
    eliminated_id = elimination.get("eliminated_id")
    if language == "zh":
        result = f"淘汰：{eliminated_id}" if eliminated_id else "最高票平局，不淘汰观点"
        return f"投票统计：{counts or '无'}\n{result}"
    result = f"Eliminated: {eliminated_id}" if eliminated_id else "Top vote tied, no candidate eliminated"
    return f"Vote counts: {counts or 'none'}\n{result}"


def _format_scores(scores: list[dict[str, Any]], language: str) -> str:
    if language == "zh":
        return "\n".join(f"{item['id']}：{item['score']} 分，{item['rationale']}" for item in scores)
    return "\n".join(f"{item['id']}: {item['score']} points, {item['rationale']}" for item in scores)


def _format_final_selection(selection: dict[str, Any], language: str) -> str:
    averages = ", ".join(f"{candidate_id}: {score}" for candidate_id, score in selection["averages"].items())
    selected_id = selection["selected_id"]
    if language == "zh":
        tie_note = "；最高均分平局，按原始编号稳定选择" if selection.get("tie") else ""
        return f"平均分：{averages}\n最终选择：{selected_id}{tie_note}"
    tie_note = "; top average tied, selected by original order" if selection.get("tie") else ""
    return f"Average scores: {averages}\nFinal choice: {selected_id}{tie_note}"


def _multi_metadata(transcript: list[TranscriptItem]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for item in transcript:
        if item.stage == "round_1":
            metadata["candidates"] = item.metadata.get("candidates", [])
        elif item.stage == "elimination":
            metadata["eliminated_id"] = item.metadata.get("eliminated_id")
        elif item.stage == "final_selection":
            metadata.update(item.metadata)
    return metadata


def _dedupe_by_id(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate["id"] in seen:
            continue
        seen.add(candidate["id"])
        unique.append(candidate)
    return unique


def _normalize_id(value: str) -> str:
    cleaned = value.strip().upper()
    if cleaned.isdigit():
        return f"R{cleaned}"
    match = re.search(r"R?\s*([1-6])", cleaned)
    return f"R{match.group(1)}" if match else cleaned


def _candidate_order(candidate_id: str) -> int:
    match = re.search(r"(\d+)", candidate_id)
    return int(match.group(1)) if match else 999


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(5, score))


def _opposite_side(side: Side) -> Side:
    return "con" if side == "pro" else "pro"


def _stage_message(stage: str, cached: bool = False) -> str:
    prefix = "Loaded cached " if cached else ""
    messages = {
        "round_1": "Round 1: generating six candidate reasons",
        "round_2": "Round 2: five opposition agents challenge the weakest logic",
        "elimination": "Vote aggregation: eliminate only a unique highest-vote candidate",
        "round_3": "Round 3: optimize challenged survivors",
        "round_4": "Round 4: five local scoring agents rate each survivor",
        "final_selection": "Selecting the highest average score",
    }
    return prefix + messages.get(stage, stage)


def _latency_cost_ratio(single_duration: float, multi_duration: float) -> float:
    if single_duration <= 0:
        return 0.0
    return round(multi_duration / single_duration, 4)


async def _noop_emit(_event: str, _payload: dict) -> None:
    return None


async def _gather_strategies(single_task, multi_task):
    tasks = [single_task, multi_task]
    try:
        single_agent, multi_bundle = await asyncio.gather(*tasks)
        multi_agent, transcript, multi_duration = multi_bundle
        return single_agent, multi_agent, transcript, multi_duration
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
