import json

from backend.app.models import DebateArgument, Language, Side


ROLE_LABELS = {
    "en": {
        "pro": "supporting side",
        "con": "opposing side",
    },
    "zh": {
        "pro": "正方",
        "con": "反方",
    },
}

ZH_REASON_MAX_CHARS = 35
ZH_LOGIC_CHAIN_MAX_CHARS = 120

ZH_ARGUMENT_LENGTH_RULES = f"""- reason 不超过 {ZH_REASON_MAX_CHARS} 个汉字（含标点，按字符计）。
- logic_chain 不超过 {ZH_LOGIC_CHAIN_MAX_CHARS} 个汉字（含标点，按字符计）。
- 超出上限视为无效输出，必须在上限内写完。"""

ZH_REASON_FIELD_DESC = "一句话论证理由"
ZH_LOGIC_CHAIN_FIELD_DESC = (
    "一段话，具有辩证性和说服力的论证我方观点，并说明该观点对我方立场的支持"
)
ZH_LOGIC_CHAIN_OPTIMIZE_DESC = (
    "通过已有内容和反驳内容优化原 logic_chain 并生成新的一段话，"
    "具有辩证性和说服力的论证我方观点，并说明该观点对我方立场的支持"
)

ZH_GENERATION_GROUND_RULES = """只依赖模型内部的概念分析、价值权衡、因果推理和常识判断；不要引用网址、实时材料、机构报告标题或具体统计数字。"""

ZH_LOGIC_CHAIN_STYLE_EXAMPLE = (
    "正确示例（logic_chain 写法，勿照抄论点）："
    "效率提升源于减少倦怠与专注度增加，虽有人质疑产出下降，但弹性排班可灵活补足工时，故整体福祉与效能更优。"
)

EN_GENERATION_GROUND_RULES = (
    "Rely only on internal conceptual analysis, value weighing, causal reasoning, "
    "and common-sense judgment. Do not cite URLs, live material, report titles, or specific statistics."
)

EN_LOGIC_CHAIN_STYLE_EXAMPLE = (
    "Correct example (logic_chain style only, do not copy the claim): "
    '"Higher efficiency comes from less burnout and better focus; although some worry output may drop, '
    "flexible scheduling can make up hours, so overall wellbeing and productivity improve."
)

EN_REASON_FIELD_DESC = "one-sentence argumentative reason for your side"
EN_LOGIC_CHAIN_FIELD_DESC = (
    "one paragraph that is dialectical and persuasive, arguing your side's viewpoint "
    "and how that viewpoint supports your side's position on the motion"
)
EN_LOGIC_CHAIN_OPTIMIZE_DESC = (
    "optimize the original logic_chain using the existing content and rebuttal content, "
    "and produce a new paragraph that is dialectical and persuasive, arguing your side's "
    "viewpoint and how that viewpoint supports your side's position on the motion"
)


def opposite_side(side: Side) -> Side:
    return "con" if side == "pro" else "pro"


def _motion_context_block(
    topic: str,
    target_side: Side,
    language: Language,
    side_claim: str | None = None,
) -> str:
    challenge_side = opposite_side(target_side)
    side_name = ROLE_LABELS[language][target_side]
    challenge_name = ROLE_LABELS[language][challenge_side]
    claim = (side_claim or "").strip()

    if language == "zh":
        your_claim = claim or (
            "支持辩题命题（同意辩题所描述的政策或立场）"
            if target_side == "pro"
            else "反对辩题命题（不同意辩题所描述的政策或立场，采取与之相反的政策立场）"
        )
        challenge_claim = (
            "支持辩题命题"
            if challenge_side == "pro"
            else "反对辩题命题"
        )
        return f"""【动笔前必读：辩题与立场】
辩题（待裁决的完整命题）：{topic}
你方：{side_name}
你方必须论证的主张：{your_claim}
{challenge_name}的主张方向：{challenge_claim}（不要写成这一方）

立场规则（违反则整段论证无效）：
1. 辩题是一句完整命题；正方=支持该命题，反方=反对该命题，不是随意选边。
2. reason 与 logic_chain 必须直接服务「你方必须论证的主张」，不得论证对方主张。
3. 生成前自检：你的 reason 是在帮正方还是帮反方？若与「你方」不符，必须重写。

立场对照示例（只学对应关系，勿照抄论点）：
辩题「中小学应全面禁止手机进校园」→ 正方=支持全面禁止；反方=反对全面禁止（可主张有限度允许携带/使用等）。
若你方是反方，却写「不该带手机」「应该全面禁止」等，即立场错误（那是在帮正方）。"""

    your_claim = claim or (
        "Affirm the motion as stated (support the policy or position described in the motion)."
        if target_side == "pro"
        else "Negate the motion (oppose the policy or position described in the motion and argue the opposite policy stance)."
    )
    challenge_claim = (
        "Affirms the motion as stated"
        if challenge_side == "pro"
        else "Negates the motion as stated"
    )
    return f"""[Read before writing: motion and side]
Motion (the full proposition to adjudicate): {topic}
Your side: {side_name}
You must argue for: {your_claim}
{challenge_name}'s direction: {challenge_claim} (do NOT argue this)

Side rules (violations invalidate the argument):
1. The motion is one complete proposition; Pro affirms it, Con negates it.
2. reason and logic_chain must directly support YOUR claim, not the other side's claim.
3. Self-check: are you helping Pro or Con? If it does not match your assigned side, rewrite.

Worked example (learn the mapping only, do not copy claims):
Motion "K-12 schools should ban student smartphones on campus" → Pro supports a ban; Con opposes a ban (e.g. regulated allowance).
If you are Con, arguing "phones should be banned" is a side error (that helps Pro)."""


def _constructive_system_prompt(language: Language) -> str:
    if language == "zh":
        return """/no_think
你是一位经验丰富的专业辩手。请只输出合法 JSON，不要输出 Markdown、解释或思考过程。"""
    return """/no_think
You are an experienced competitive debater. Output valid JSON only, with no Markdown, explanation, or hidden reasoning."""


def _logic_chain_style_rules(language: Language) -> str:
    if language == "zh":
        return f"""- logic_chain 必须是连贯的一段话（一个自然段），用自然语句完成论证，不要分段、不要列小标题。
{ZH_LOGIC_CHAIN_STYLE_EXAMPLE}"""
    return f"""- logic_chain must be one continuous paragraph written in natural prose. Do not split into sections or headings.
{EN_LOGIC_CHAIN_STYLE_EXAMPLE}"""


def _single_argument_field_requirements(language: Language) -> str:
    style_rules = _logic_chain_style_rules(language)
    if language == "zh":
        return f"""要求：
- reason：{ZH_REASON_FIELD_DESC}。
- logic_chain：{ZH_LOGIC_CHAIN_FIELD_DESC}。
{style_rules}
{ZH_ARGUMENT_LENGTH_RULES}"""
    return f"""Requirements:
- reason: {EN_REASON_FIELD_DESC}.
- logic_chain: {EN_LOGIC_CHAIN_FIELD_DESC}.
{style_rules}"""


def _single_argument_json_template(language: Language) -> str:
    if language == "zh":
        return f"""输出 JSON 对象，格式必须完全一致：
{{"reason":"{ZH_REASON_FIELD_DESC}","logic_chain":"{ZH_LOGIC_CHAIN_FIELD_DESC}"}}"""
    return f"""Return exactly this JSON object:
{{"reason":"{EN_REASON_FIELD_DESC}","logic_chain":"{EN_LOGIC_CHAIN_FIELD_DESC}"}}"""


def _single_argument_user_block(language: Language) -> str:
    return f"{_single_argument_json_template(language)}\n\n{_single_argument_field_requirements(language)}"


def _constructive_messages_base(
    topic: str,
    target_side: Side,
    language: Language,
    side_claim: str | None = None,
) -> tuple[str, str]:
    motion_context = _motion_context_block(topic, target_side, language, side_claim)
    ground_rules = ZH_GENERATION_GROUND_RULES if language == "zh" else EN_GENERATION_GROUND_RULES
    system = _constructive_system_prompt(language)
    user_prefix = f"""{motion_context}

{ground_rules}

"""
    return system, user_prefix


def _candidate_generation_task_block(language: Language) -> str:
    if language == "zh":
        return f"""请生成 6 个彼此有明显区分的立论，每条都必须论证「你方必须论证的主张」，id 为 R1 到 R6。

输出 JSON 对象：
{{"candidates":[{{"id":"R1","reason":"...","logic_chain":"..."}},{{"id":"R2","reason":"...","logic_chain":"..."}},{{"id":"R3","reason":"...","logic_chain":"..."}},{{"id":"R4","reason":"...","logic_chain":"..."}},{{"id":"R5","reason":"...","logic_chain":"..."}},{{"id":"R6","reason":"...","logic_chain":"..."}}]}}

要求：
- 必须正好 6 条，id 必须是 R1 到 R6。
每一条候选均须满足下列要求（与单条立论完全相同）：
"""
    return f"""Generate 6 distinct arguments, each arguing YOUR assigned claim on the motion, with ids R1 through R6.

Return this JSON object:
{{"candidates":[{{"id":"R1","reason":"...","logic_chain":"..."}},{{"id":"R2","reason":"...","logic_chain":"..."}},{{"id":"R3","reason":"...","logic_chain":"..."}},{{"id":"R4","reason":"...","logic_chain":"..."}},{{"id":"R5","reason":"...","logic_chain":"..."}},{{"id":"R6","reason":"...","logic_chain":"..."}}]}}

Requirements:
- Return exactly 6 items, with ids R1 through R6.
Every candidate must satisfy the following requirements (identical to single-argument generation):
"""


def single_prompt(
    topic: str,
    target_side: Side,
    language: Language,
    side_claim: str | None = None,
) -> list[dict[str, str]]:
    system, user_prefix = _constructive_messages_base(topic, target_side, language, side_claim)
    user = f"{user_prefix}{_single_argument_user_block(language)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def candidate_generation_prompt(
    topic: str,
    target_side: Side,
    language: Language,
    side_claim: str | None = None,
) -> list[dict[str, str]]:
    system, user_prefix = _constructive_messages_base(topic, target_side, language, side_claim)
    user = f"{user_prefix}{_candidate_generation_task_block(language)}{_single_argument_field_requirements(language)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def challenge_prompt(
    agent_index: int,
    topic: str,
    target_side: Side,
    language: Language,
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    candidate_block = json.dumps(candidates, ensure_ascii=False, indent=2)
    challenge_side = opposite_side(target_side)
    if language == "zh":
        system = f"""/no_think
你是第二轮反对方 agent {agent_index}。请只输出合法 JSON，不要输出 Markdown、解释或思考过程。"""
        user = f"""辩题：{topic}
你代表的立场：{ROLE_LABELS[language][challenge_side]}
你要质询的对象：{ROLE_LABELS[language][target_side]}第一轮生成的 6 个理由

全部候选理由：
{candidate_block}

请选择你认为最站不住脚的一条，只能选一个 id。

输出 JSON 对象：
{{"target_id":"R1","question":"对该逻辑的质询","weakness_reason":"为什么它最站不住脚","opposing_reason":"反对方基于自身立场给出的反对理由"}}

要求：
- target_id 必须来自 R1 到 R6。
- question、weakness_reason、opposing_reason 都必须具体指向被选中的逻辑链条。"""
    else:
        system = f"""/no_think
You are opposition agent {agent_index} in round two. Output valid JSON only, with no Markdown, explanation, or hidden reasoning."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][challenge_side]}
You are challenging the 6 reasons generated for: {ROLE_LABELS[language][target_side]}

All candidate reasons:
{candidate_block}

Choose the one logic chain you consider weakest. Pick exactly one id.

Return this JSON object:
{{"target_id":"R1","question":"the challenge question","weakness_reason":"why this logic is weakest","opposing_reason":"the opposing side's reason against it"}}

Requirements:
- target_id must be one of R1 through R6.
- question, weakness_reason, and opposing_reason must directly address the chosen logic chain."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def optimization_prompt(
    topic: str,
    target_side: Side,
    language: Language,
    candidate: dict[str, str],
    challenges: list[dict[str, str]],
    side_claim: str | None = None,
) -> list[dict[str, str]]:
    challenge_block = json.dumps(challenges, ensure_ascii=False, indent=2)
    candidate_block = json.dumps(candidate, ensure_ascii=False, indent=2)
    system, user_prefix = _constructive_messages_base(topic, target_side, language, side_claim)
    if language == "zh":
        optimize_context = f"""原观点：
{candidate_block}

第二轮针对该观点的全部质询与反对理由：
{challenge_block}

请保留原 reason 的核心方向；优化后仍必须论证「你方必须论证的主张」，不得滑向对方立场。
logic_chain：{ZH_LOGIC_CHAIN_OPTIMIZE_DESC}；必须吸收所有质询与反驳内容。

"""
    else:
        optimize_context = f"""Original candidate:
{candidate_block}

All challenges and opposing reasons aimed at this candidate:
{challenge_block}

Keep the core direction of the original reason. The optimized text must still argue YOUR assigned claim, not the other side.
logic_chain: {EN_LOGIC_CHAIN_OPTIMIZE_DESC}; absorb every challenge and rebuttal direction.

"""
    user = f"{user_prefix}{optimize_context}{_single_argument_user_block(language)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def scoring_prompt(
    agent_index: int,
    topic: str,
    target_side: Side,
    language: Language,
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    candidate_block = json.dumps(candidates, ensure_ascii=False, indent=2)
    if language == "zh":
        system = f"""/no_think
你是第四轮打分 agent {agent_index}。请只输出合法 JSON，不要输出 Markdown、解释或思考过程。"""
        user = f"""辩题：{topic}
目标立场：{ROLE_LABELS[language][target_side]}

待评分观点：
{candidate_block}

请根据相关性、逻辑完整性、辩证性、抗质询能力，对每个观点打 0 到 5 的整数分。

输出 JSON 对象：
{{"scores":[{{"id":"R1","score":0,"rationale":"简短评分理由"}}]}}

要求：
- 必须给每个待评分观点一个分数。
- score 必须是 0、1、2、3、4、5 之一。"""
    else:
        system = f"""/no_think
You are scoring agent {agent_index} in round four. Output valid JSON only, with no Markdown, explanation, or hidden reasoning."""
        user = f"""Motion: {topic}
Target side: {ROLE_LABELS[language][target_side]}

Candidates to score:
{candidate_block}

Score every candidate from 0 to 5 as an integer, using relevance, logical completeness, dialectical strength, and resistance to challenge.

Return this JSON object:
{{"scores":[{{"id":"R1","score":0,"rationale":"short scoring reason"}}]}}

Requirements:
- Every candidate must receive one score.
- score must be one of 0, 1, 2, 3, 4, 5."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def argument_to_dict(argument: DebateArgument, candidate_id: str | None = None) -> dict[str, str]:
    data = argument.model_dump()
    if candidate_id:
        return {"id": candidate_id, **data}
    return data
