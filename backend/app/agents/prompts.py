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

任务：基于辩题和搜索资料，为用户指定立场生成 3 个有力论点。
输出必须是一段话，正好 3 句，每句包含论点和论据，每句少于 50 个汉字。
不要使用标题、编号、项目符号、Markdown 列表或“核心主张”等 schema 标签。
语言：中文。风格：适合课堂演示和实际辩论准备，清晰、克制、有证据意识。"""
        user = f"""辩题：{topic}
目标立场：{ROLE_LABELS[language][target_side]}

{source_block(sources, language)}

请只输出一段话，正好 3 句。每句都要有“观点 + 理由或证据”。"""
    else:
        system = """/no_think
You are an experienced competitive debater. Output only the final answer, with no hidden reasoning.

Task: generate three strong arguments for the requested side using the debate motion and search sources.
Output one paragraph with exactly 3 sentences. Each sentence must combine one claim with one reason or evidence point.
Do not use titles, numbering, bullet points, Markdown lists, or schema labels.
Keep each sentence under 50 words.
Language: English. Style: concise, classroom-demo ready, evidence-aware."""
        user = f"""Motion: {topic}
Target side: {ROLE_LABELS[language][target_side]}

{source_block(sources, language)}

Return only one paragraph with exactly 3 sentences."""
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

请输出一段短发言，包含 2 个核心立论。
要求：不要用列表或标题；每个立论都要包含观点和理由；整段少于 90 个汉字；中文输出。"""
    else:
        specialty = "logic, values, definitions, and causal reasoning" if role_key.endswith("logic") else "data, research evidence, and concrete cases"
        system = f"""/think
You are {role_name}. Output only the presentable final speech, with no hidden reasoning.
Your specialty is {specialty}."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][side]}

{source_block(sources, language)}

Give a short speech with 2 core constructive arguments.
Do not use titles or bullet points. Each argument must include a claim and a reason. Keep the whole paragraph under 80 words."""
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

请输出一段第二轮反驳短发言：精准反驳对方 1 个薄弱点，并强化己方 1 个核心点。
不要用列表或标题，整段少于 90 个汉字。"""
    else:
        system = f"""/think
You are {role_name}. Output only the presentable final speech, with no hidden reasoning."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][side]}

Your round-one constructive:
{own_constructive}

Opposing round-one constructives:
{opposing_constructives}

Produce one short rebuttal paragraph: attack one weak opposing point and strengthen one core point.
Do not use titles or bullet points. Keep it under 80 words."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def synthesis_prompt(topic: str, target_side: Side, language: Language, transcript: str) -> list[dict[str, str]]:
    if language == "zh":
        system = """/think
你是一位资深辩论裁判和论点分析师。请只输出最终提炼结果，不要输出思考过程。"""
        user = f"""辩题：{topic}
需要提炼的立场：{ROLE_LABELS[language][target_side]}

完整内部对抗记录：
{transcript}

请从上述对抗中提炼出目标立场最强的 3 个论点。
输出必须是一段话，正好 3 句；每句包含论点和论据，并能回应对方攻击；每句少于 50 个汉字。
不要使用标题、编号、项目符号、Markdown 列表或 schema 标签。"""
    else:
        system = """/think
You are a senior debate judge and argument analyst. Output only the final synthesis, with no hidden reasoning."""
        user = f"""Motion: {topic}
Target side to extract: {ROLE_LABELS[language][target_side]}

Full internal adversarial transcript:
{transcript}

Extract the three strongest arguments for the target side.
Output one paragraph with exactly 3 sentences. Each sentence must combine a claim, evidence or reasoning, and rebuttal awareness.
Do not use titles, numbering, bullets, Markdown lists, or schema labels. Keep each sentence under 50 words."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
