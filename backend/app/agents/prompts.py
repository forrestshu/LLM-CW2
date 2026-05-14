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

任务：基于辩题和搜索资料，为用户指定立场生成一段直接立论。
输出必须是一段话，正好 4 句：第一句给出主张，第二句做简单解释，第三句给出一个相关例子或事实，第四句收束结论。
每句控制在 35 个汉字以内，整体篇幅要和多 Agent 最终答案接近。
不要使用标题、编号、项目符号、Markdown 列表或“核心主张”等 schema 标签。
语言：中文。风格：清晰、直接、像一次未经对抗检验的单次回答。"""
        user = f"""辩题：{topic}
目标立场：{ROLE_LABELS[language][target_side]}

{source_block(sources, language)}

请只输出一段话，正好 4 句，篇幅接近多 Agent 最终答案，但不要刻意展开复杂反驳。"""
    else:
        system = """/no_think
You are an experienced competitive debater. Output only the final answer, with no hidden reasoning.

Task: generate a direct argument for the requested side using the debate motion and search sources.
Output one paragraph with exactly 4 sentences: sentence 1 states the claim, sentence 2 gives a simple explanation, sentence 3 gives one relevant example or fact, and sentence 4 closes the point.
Do not use titles, numbering, bullet points, Markdown lists, or schema labels.
Keep each sentence under 22 words, and keep the total length close to the multi-agent final answer.
Language: English. Style: clear, direct, and like a single-pass answer that has not been adversarially tested."""
        user = f"""Motion: {topic}
Target side: {ROLE_LABELS[language][target_side]}

{source_block(sources, language)}

Return only one paragraph with exactly 4 sentences, close in length to the multi-agent final answer, without elaborate rebuttal work."""
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

请输出一段短发言，只包含 1 个核心论点和 1 个论据。
要求：不要用列表或标题；正好 2 句，第一句论点，第二句论据；每句少于 50 个汉字；中文输出。"""
    else:
        specialty = "logic, values, definitions, and causal reasoning" if role_key.endswith("logic") else "data, research evidence, and concrete cases"
        system = f"""/think
You are {role_name}. Output only the presentable final speech, with no hidden reasoning.
Your specialty is {specialty}."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][side]}

{source_block(sources, language)}

Give a short speech with 1 core argument and 1 evidence point.
Do not use titles or bullet points. Use exactly 2 sentences: first the argument, then the evidence. Keep each sentence under 50 words."""
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

请输出一段第二轮反驳短发言，只包含 1 个反驳论点和 1 个支撑论据。
不要用列表或标题；正好 2 句，第一句反驳论点，第二句论据；每句少于 50 个汉字。"""
    else:
        system = f"""/think
You are {role_name}. Output only the presentable final speech, with no hidden reasoning."""
        user = f"""Motion: {topic}
Your side: {ROLE_LABELS[language][side]}

Your round-one constructive:
{own_constructive}

Opposing round-one constructives:
{opposing_constructives}

Produce one short rebuttal paragraph with 1 rebuttal argument and 1 evidence point.
Do not use titles or bullet points. Use exactly 2 sentences: first the rebuttal argument, then the evidence. Keep each sentence under 50 words."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def synthesis_prompt(topic: str, target_side: Side, language: Language, transcript: str) -> list[dict[str, str]]:
    if language == "zh":
        system = """/think
你是一位资深辩论裁判、论点分析师和决赛陈词教练。请只输出面向观众的最终答案，不要输出思考过程。"""
        user = f"""辩题：{topic}
需要提炼的立场：{ROLE_LABELS[language][target_side]}

完整内部对抗记录：
{transcript}

请把内部对抗中经得起反驳的内容，整合成一段和单 Agent 篇幅接近、但论证质量明显更高的最终立论。
输出必须是一段话，正好 4 句：第一句给出更精确的主张，第二句展开更深的因果链或利益权衡，第三句使用贴合辩题且多数观众熟悉的具体例子、制度事实或研究结论，第四句用让步反驳、比较权衡或风险转移等论证手法预判异议并压回目标立场。
每句控制在 35 个汉字以内，整体篇幅不要明显长于单 Agent 最终答案。
不要提及“内部对抗”“辩论记录”“第几轮”“Agent”“裁判”“对方刚才说”等过程信息。
不要使用标题、编号、项目符号、Markdown 列表或 schema 标签。"""
    else:
        system = """/think
You are a senior debate judge, argument analyst, and final-speech coach. Output only the audience-facing final answer, with no hidden reasoning."""
        user = f"""Motion: {topic}
Target side to extract: {ROLE_LABELS[language][target_side]}

Full internal adversarial transcript:
{transcript}

Turn the strongest rebuttal-tested material from the internal debate into a final argument that is similar in length to the single-agent answer but visibly better in reasoning quality.
Output one paragraph with exactly 4 sentences: sentence 1 states a sharper claim, sentence 2 develops a deeper causal chain or value tradeoff, sentence 3 uses a well-known and motion-specific example, institutional fact, or research finding, and sentence 4 uses concession-rebuttal, comparative weighing, or risk-shifting to answer an objection while showing why the target side still wins.
Keep each sentence under 22 words, and do not make the total answer noticeably longer than the single-agent final answer.
Do not mention the internal debate, transcript, rounds, agents, judge, or what the other side previously said.
Do not use titles, numbering, bullets, Markdown lists, or schema labels."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def advantage_prompt(
    topic: str,
    target_side: Side,
    language: Language,
    single_answer: str,
    final_synthesis: str,
    transcript: str,
) -> list[dict[str, str]]:
    if language == "zh":
        system = """/no_think
你是辩论质量评估助手。请只输出合法 JSON，不要输出 Markdown、解释或思考过程。"""
        user = f"""辩题：{topic}
目标立场：{ROLE_LABELS[language][target_side]}

单 Agent 答案：
{single_answer}

多 Agent final synthesis：
{final_synthesis}

内部对抗记录：
{transcript}

请为 final synthesis 的每一句生成标注，用于解释多 Agent 优势。
输出 JSON 对象，格式必须是：
{{"rebuttal_notes":["第1句反驳来源说明","第2句反驳来源说明","第3句反驳来源说明","第4句反驳来源说明"],"advantage_notes":["第1句相比单 Agent 好在哪里","第2句相比单 Agent 好在哪里","第3句相比单 Agent 好在哪里","第4句相比单 Agent 好在哪里"]}}
要求：rebuttal_notes 要指出这句话吸收或回应了反方哪个具体反驳方向，例如“回应反方关于过度审查的质疑”；advantage_notes 要说明比单 Agent 更好的维度，例如论点更深、例子更贴题、例子更为大众熟悉、因果链更完整、使用了让步反驳或比较权衡。
每条说明少于 24 个汉字。不要改写 final synthesis 原文。"""
    else:
        system = """/no_think
You are a debate-quality evaluation assistant. Output valid JSON only, with no Markdown, explanation, or hidden reasoning."""
        user = f"""Motion: {topic}
Target side: {ROLE_LABELS[language][target_side]}

Single-agent answer:
{single_answer}

Multi-agent final synthesis:
{final_synthesis}

Internal adversarial transcript:
{transcript}

Create annotations for each sentence of the final synthesis to explain the multi-agent advantage.
Return a JSON object exactly like:
{{"rebuttal_notes":["note for sentence 1","note for sentence 2","note for sentence 3","note for sentence 4"],"advantage_notes":["advantage over single agent for sentence 1","advantage over single agent for sentence 2","advantage over single agent for sentence 3","advantage over single agent for sentence 4"]}}
Requirements: rebuttal_notes should identify which opposing rebuttal direction the sentence absorbs or answers, such as "answers the over-censorship objection"; advantage_notes should name the quality improvement over the single agent, such as deeper reasoning, better-fitted example, more widely known example, fuller causal chain, concession-rebuttal, or comparative weighing.
Keep each note under 14 words. Do not rewrite the final synthesis."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
