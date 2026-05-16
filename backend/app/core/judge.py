import json
import time
import re

import httpx


class JudgeClient:
    def __init__(
        self,
        api_key: str | None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, trust_env=False)

    async def evaluate(
        self,
        topic: str,
        target_side: str,
        language: str,
        single_content: str,
        adversarial_content: str,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("JUDGE_API_KEY is not configured. Add it to .env before evaluating.")

        started = time.perf_counter()

        if language == "zh":
            side_text = "正方" if target_side == "pro" else "反方"
            system_prompt = f"""你是一位资深的辩论裁判，负责评估两组辩论论点的质量。

【评估维度】（1-10分）
1. 说服力：论点的逻辑严谨性和说服力
2. 论据丰富度：支持论点的证据、案例和数据
3. 论点清晰度：论点表述的清晰度和结构化程度
4. 攻防价值：针对对方可能的反驳进行的防御

请严格按照JSON格式输出："""
        else:
            side_text = "Pro" if target_side == "pro" else "Con"
            system_prompt = f"""You are a senior debate judge, responsible for evaluating the quality of two sets of debate arguments.

【Evaluation Dimensions (1-10 points)】
1. Persuasiveness: Logical rigor and persuasiveness of arguments
2. Evidence Richness: Evidence, cases, and data supporting arguments
3. Argument Clarity: Clarity and structure of argument presentation
4. Attack-Defense Value: Defense against potential counter-arguments

Please output strictly in JSON format:"""

        user_prompt = f"""
【辩题】{topic}
【立场】{side_text}

【方案一：单Agent论点】
{single_content}

【方案二：对抗性多Agent论点】
{adversarial_content}

请分别评估两组论点，并给出总分和胜负判断。"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with self._http_client() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = self._parse_fallback(content, language)

        duration = time.perf_counter() - started

        return {
            "single_score": int(result.get("single_score", 5)),
            "adversarial_score": int(result.get("adversarial_score", 5)),
            "single_reasoning": result.get("single_reasoning", ""),
            "adversarial_reasoning": result.get("adversarial_reasoning", ""),
            "winner": result.get("winner", "tie"),
            "winner_reasoning": result.get("winner_reasoning", ""),
            "total_duration_sec": round(duration, 3),
        }

    def _parse_fallback(self, content: str, language: str) -> dict:
        single_score_match = re.search(r"方案一.*?(\d+)|single.*?(\d+)", content, re.IGNORECASE)
        adversarial_score_match = re.search(r"方案二.*?(\d+)|adversarial.*?(\d+)", content, re.IGNORECASE)

        single_score = int(next(g for g in single_score_match.groups() if g)) if single_score_match else 5
        adversarial_score = int(next(g for g in adversarial_score_match.groups() if g)) if adversarial_score_match else 5

        winner = "tie"
        if single_score > adversarial_score:
            winner = "single"
        elif adversarial_score > single_score:
            winner = "adversarial"

        return {
            "single_score": min(10, max(1, single_score)),
            "adversarial_score": min(10, max(1, adversarial_score)),
            "single_reasoning": content[:200],
            "adversarial_reasoning": content[-200:],
            "winner": winner,
            "winner_reasoning": "",
        }