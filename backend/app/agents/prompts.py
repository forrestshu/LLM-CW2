from backend.app.models import Language, Side, Source


ROLE_LABELS = {
    "en": {
        "pro": "supporting side",
        "con": "opposing side",
        "single": "Single Agent",
        "pro_logic": "Pro A, logic debater",
        "pro_data": "Pro B, evidence debater",
        "con_logic": "Con A, logic debater",
        "con_data": "Con B, evidence debater",
        "synthesis": "Neutral synthesis judge",
    },
    "zh": {
        "pro": "正方",
        "con": "反方",
        "single": "单 Agent",
        "pro_logic": "正方A，逻辑型辩手",
        "pro_data": "正方B，数据型辩手",
        "con_logic": "反方A，逻辑型辩手",
        "con_data": "反方B，数据型辩手",
        "synthesis": "中立综合裁判",
    },
}


def source_block(sources: list[Source], language: Language) -> str:
    if not sources:
        return "No live search sources are available. Be explicit that claims need verification." if language == "en" else "没有可用的实时搜索来源。涉及事实时请明确需要核验。"
    lines = []
    for idx, source in enumerate(sources, start=1):
        lines.append(f"[{idx}] {source.title}\nURL: {source.url}\nSnippet: {source.snippet[:700]}")
    heading = "Search sources" if language == "en" else "搜索资料"
    return f"{heading}:\n" + "\n\n".join(lines)


def single_prompt(topic: str, target_side: Side, language: Language, sources: list[Source]) -> list[dict[str, str]]:
    if language == "zh":
        system = """/no_think
你是一位经验丰富的专业辩手。请只输出最终答案，不要输出思考过程。

任务：基于辩题和搜索资料，为用户指定立场生成 3 条有力论点。
每条必须包含：标题、核心主张、逻辑支撑、具体证据、可能反驳应对。
每条控制在 120 字以内。
语言：中文。风格：适合课堂演示和实际辩论准备，清晰、克制、有证据意识。"""
        user = f"""辩题：{topic}
目标立场：{ROLE_LABELS[language][target_side]}

{source_block(sources, language)}

请严格输出 3 条论点，格式如下：
论点一：[标题]
- 核心主张：...
- 逻辑支撑：...
- 具体证据：...
- 反驳应对：...

论点二、论点三同上。"""
    else:
        system = """/no_think
You are an experienced competitive debater. Output only the final answer, with no hidden reasoning.

Task: generate three strong arguments for the requested side using the debate motion and search sources.
Each argument must include a title, core claim, reasoning, concrete evidence, and response to likely objections.
Keep each argument under 90 words.
Language: English. Style: concise, classroom-demo ready, evidence-aware."""
        user = f"""Motion: {topic}
Target side: {ROLE_LABELS[language][target_side]}

{source_block(sources, language)}

Return exactly three arguments in this format:
Argument One: [Title]
- Core claim: ...
- Reasoning: ...
- Concrete evidence: ...
- Objection response: ...

Repeat for Argument Two and Argument Three."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def debater_prompt(role_key: str, topic: str, language: Language, sources: list[Source]) -> list[dict[str, str]]:
    side = "pro" if role_key.startswith("pro") else "con"
    role_name = ROLE_LABELS[language][role_key]
    if language == "zh":
        specialty = "逻辑推理、价值判断、概念界定" if role_key.endswith("logic") else "真实数据、研究报告、案例证据"
        system = f"""/think
你是{role_name}。请只输出可展示的最终发言，不要输出思考过程。
你的专长是：{specialty}。"""
        user = f"""辩题：{topic}
你的立场：{ROLE_LABELS[language][side]}

{source_block(sources, language)}

请提出 2 条本方核心立论。要求：
- 每条论点有清晰标题
- 逻辑型辩手重视概念和推理，数据型辩手重视证据和案例
- 主动指出对方最可能攻击的位置
- 每条控制在 100 字以内
- 中文输出"""
    else:
        specialty = "logic, values, definitions, and causal reasoning" if role_key.endswith("logic") else "data, research evidence, and concrete cases"
        system = f"""/think
You are {role_name}. Output only the presentable final speech, with no hidden reasoning.
Your specialty is {specialty}."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][side]}

{source_block(sources, language)}

Give 2 core constructive arguments for your side. Requirements:
- Use clear argument titles
- Logic debaters emphasize definitions and reasoning; evidence debaters emphasize data and cases
- Anticipate where the other side will attack
- Keep each argument under 75 words
- Output in English"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rebuttal_prompt(
    role_key: str,
    topic: str,
    language: Language,
    own_constructive: str,
    opposing_constructives: str,
) -> list[dict[str, str]]:
    side = "pro" if role_key.startswith("pro") else "con"
    role_name = ROLE_LABELS[language][role_key]
    if language == "zh":
        system = f"""/think
你是{role_name}。请只输出可展示的最终发言，不要输出思考过程。"""
        user = f"""辩题：{topic}
你的立场：{ROLE_LABELS[language][side]}

你的第一轮立论：
{own_constructive}

对方第一轮立论：
{opposing_constructives}

请完成第二轮反驳：
1. 精准反驳对方最薄弱的 1-2 个论点
2. 修正并强化你自己的核心论点
3. 输出可直接用于辩论的简洁发言
4. 总长度控制在 180 字以内"""
    else:
        system = f"""/think
You are {role_name}. Output only the presentable final speech, with no hidden reasoning."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][side]}

Your round-one constructive:
{own_constructive}

Opposing round-one constructives:
{opposing_constructives}

Produce the rebuttal round:
1. Attack the weakest 1-2 opposing arguments precisely
2. Repair and strengthen your own position
3. Write concise debate-ready material
4. Keep the whole rebuttal under 130 words"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def synthesis_prompt(topic: str, target_side: Side, language: Language, transcript: str) -> list[dict[str, str]]:
    if language == "zh":
        system = """/think
你是一位资深辩论裁判和论点分析师。请只输出最终提炼结果，不要输出思考过程。"""
        user = f"""辩题：{topic}
需要提炼的立场：{ROLE_LABELS[language][target_side]}

完整内部对抗记录：
{transcript}

请从上述对抗中提炼出目标立场最强的 3 条论点。
要求：
- 每条论点能回应对方主要攻击
- 综合逻辑论证与事实证据
- 适合实际辩论使用
- 每条控制在 130 字以内

格式：
论点一：[标题]
- 核心主张：...
- 逻辑支撑：...
- 具体证据：...
- 对方反驳应对：...

论点二、论点三同上。"""
    else:
        system = """/think
You are a senior debate judge and argument analyst. Output only the final synthesis, with no hidden reasoning."""
        user = f"""Motion: {topic}
Target side to extract: {ROLE_LABELS[language][target_side]}

Full internal adversarial transcript:
{transcript}

Extract the three strongest arguments for the target side.
Requirements:
- Each argument must answer the opponent's main attacks
- Combine reasoning and evidence
- Make the output practical for real debate use
- Keep each argument under 100 words

Format:
Argument One: [Title]
- Core claim: ...
- Reasoning: ...
- Concrete evidence: ...
- Objection response: ...

Repeat for Argument Two and Argument Three."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
