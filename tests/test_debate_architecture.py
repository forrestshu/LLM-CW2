from dataclasses import dataclass

import pytest

from backend.app.agents.debate import (
    DebateOrchestrator,
    _latency_cost_ratio,
    _normalize_argument,
    _parse_argument,
    _parse_candidates,
    _select_elimination,
    _select_final_candidate,
)
from backend.app.agents.prompts import (
    ZH_LOGIC_CHAIN_MAX_CHARS,
    ZH_REASON_MAX_CHARS,
    candidate_generation_prompt,
    optimization_prompt,
    single_prompt,
)
from backend.app.models import DebateArgument, GenerationRequest


def test_single_prompt_requires_structured_logic_only_output():
    messages = single_prompt("AI should be regulated", "pro", "en")
    combined = "\n".join(message["content"] for message in messages)

    assert '"reason"' in combined
    assert '"logic_chain"' in combined
    assert "URLs" in combined
    assert "internal debate" not in combined
    assert "Motion (the full proposition" in combined
    assert "Self-check" in combined


def test_zh_con_prompt_clarifies_motion_and_side_with_claim():
    messages = single_prompt(
        "中小学应全面禁止手机进校园",
        "con",
        "zh",
        side_claim="反对全面禁止，主张有限度允许携带手机",
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "中小学应全面禁止手机进校园" in combined
    assert "反方" in combined
    assert "反对全面禁止" in combined
    assert "立场错误" in combined
    assert "全面禁止" in combined


def test_candidate_prompt_matches_single_argument_style():
    single_messages = single_prompt("人工智能应被监管", "pro", "zh", "应加强监管")
    multi_messages = candidate_generation_prompt("人工智能应被监管", "pro", "zh", "应加强监管")
    single = "\n".join(message["content"] for message in single_messages)
    multi = "\n".join(message["content"] for message in multi_messages)

    assert single_messages[0]["content"] == multi_messages[0]["content"]
    assert "只依赖模型内部的概念分析" in single
    assert "只依赖模型内部的概念分析" in multi
    assert "正确示例" in single
    assert "正确示例" in multi
    assert "效率提升源于减少倦怠" in single
    assert "效率提升源于减少倦怠" in multi
    assert "禁止分段标题" not in single
    assert "禁止分段标题" not in multi
    assert '"candidates"' in multi
    assert "与单条立论完全相同" in multi


def test_optimization_prompt_uses_same_constructive_system_as_single():
    single_system = single_prompt("人工智能应被监管", "pro", "zh", "应加强监管")[0]["content"]
    optimize_system = optimization_prompt(
        "人工智能应被监管",
        "pro",
        "zh",
        {"id": "R1", "reason": "短理由", "logic_chain": "短链条"},
        [{"target_id": "R1", "question": "q", "weakness_reason": "w", "opposing_reason": "o"}],
        "应加强监管",
    )[0]["content"]

    assert single_system == optimize_system
    assert "第三轮逻辑优化" not in optimize_system


def test_zh_prompts_include_argument_length_limits():
    for prompt_builder in (
        lambda: single_prompt("人工智能应被监管", "pro", "zh", "应加强监管"),
        lambda: candidate_generation_prompt("人工智能应被监管", "pro", "zh", "应加强监管"),
        lambda: optimization_prompt(
            "人工智能应被监管",
            "pro",
            "zh",
            {"id": "R1", "reason": "短理由", "logic_chain": "短链条"},
            [{"target_id": "R1", "question": "q", "weakness_reason": "w", "opposing_reason": "o"}],
            "应加强监管",
        ),
    ):
        combined = "\n".join(message["content"] for message in prompt_builder())
        assert f"不超过 {ZH_REASON_MAX_CHARS} 个汉字" in combined
        assert f"不超过 {ZH_LOGIC_CHAIN_MAX_CHARS} 个汉字" in combined


def test_zh_argument_normalization_enforces_length_limits():
    long_reason = "中" * 50
    long_logic = "文" * 200
    normalized = _normalize_argument(
        DebateArgument(reason=long_reason, logic_chain=long_logic),
        "zh",
    )

    assert len(normalized.reason) <= ZH_REASON_MAX_CHARS
    assert len(normalized.logic_chain) <= ZH_LOGIC_CHAIN_MAX_CHARS


def test_parse_argument_applies_zh_limits():
    parsed = _parse_argument(
        '{"reason":"' + ("辩" * 50) + '","logic_chain":"' + ("论" * 200) + '"}',
        "zh",
    )

    assert len(parsed.reason) <= ZH_REASON_MAX_CHARS
    assert len(parsed.logic_chain) <= ZH_LOGIC_CHAIN_MAX_CHARS


def test_argument_parser_accepts_fenced_json():
    parsed = _parse_argument(
        '```json\n{"reason":"Rules create trust.","logic_chain":"If systems affect rights, then accountable rules reduce harm while still allowing proportionate innovation."}\n```'
    )

    assert parsed.reason == "Rules create trust."
    assert "accountable rules" in parsed.logic_chain


def test_candidate_parser_requires_six_items():
    content = {
        "candidates": [
            {"id": f"R{index}", "reason": f"Reason {index}.", "logic_chain": f"Logic {index}."}
            for index in range(1, 7)
        ]
    }

    parsed = _parse_candidates(__import__("json").dumps(content))

    assert [item["id"] for item in parsed] == ["R1", "R2", "R3", "R4", "R5", "R6"]


def test_unique_highest_vote_is_eliminated():
    result = _select_elimination(
        [
            {"target_id": "R1"},
            {"target_id": "R1"},
            {"target_id": "R2"},
            {"target_id": "R3"},
            {"target_id": "R1"},
        ]
    )

    assert result["eliminated_id"] == "R1"
    assert result["tie"] is False


def test_tied_highest_vote_eliminates_nothing():
    result = _select_elimination(
        [
            {"target_id": "R1"},
            {"target_id": "R2"},
            {"target_id": "R1"},
            {"target_id": "R2"},
            {"target_id": "R3"},
        ]
    )

    assert result["eliminated_id"] is None
    assert result["tie"] is True


def test_final_selection_uses_average_then_original_order_tie_break():
    candidates = [
        {"id": "R1", "reason": "A.", "logic_chain": "A chain."},
        {"id": "R2", "reason": "B.", "logic_chain": "B chain."},
    ]
    scorecards = [
        [{"id": "R1", "score": 4}, {"id": "R2", "score": 5}],
        [{"id": "R1", "score": 5}, {"id": "R2", "score": 4}],
    ]

    selection = _select_final_candidate(candidates, scorecards)

    assert selection["selected_id"] == "R1"
    assert selection["tie"] is True
    assert selection["final_average_score"] == 4.5


@dataclass
class FakeResponse:
    content: str
    duration_sec: float = 0.01
    token_estimate: int = 10


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse('{"reason":"Improved reason.","logic_chain":"Improved logic answers the challenge while preserving the original claim."}')


@pytest.mark.asyncio
async def test_only_challenged_non_eliminated_candidates_are_optimized(tmp_path):
    request = GenerationRequest(topic="AI should be regulated", target_side="pro", language="en", use_cache=False)
    candidates = [
        {"id": "R1", "reason": "A.", "logic_chain": "A chain."},
        {"id": "R2", "reason": "B.", "logic_chain": "B chain."},
        {"id": "R3", "reason": "C.", "logic_chain": "C chain."},
    ]
    challenges = [
        {"target_id": "R1", "question": "q", "weakness_reason": "w", "opposing_reason": "o"},
        {"target_id": "R2", "question": "q", "weakness_reason": "w", "opposing_reason": "o"},
    ]
    fake_llm = FakeLLM()
    orchestrator = DebateOrchestrator(fake_llm, tmp_path)

    current, transcript = await orchestrator._run_optimizers(
        request,
        candidates,
        challenges,
        eliminated_id="R2",
        emit=_noop_emit,
    )

    assert fake_llm.calls == 1
    assert [item.metadata.get("candidate_id") for item in transcript if "candidate_id" in item.metadata] == ["R1"]
    assert [candidate["id"] for candidate in current] == ["R1", "R3"]
    assert current[0]["reason"] == "Improved reason."
    assert _latency_cost_ratio(2.0, 8.0) == 4.0


async def _noop_emit(_event: str, _payload: dict) -> None:
    return None
