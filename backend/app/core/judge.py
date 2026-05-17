import asyncio
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
        timeout: float = 180.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

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

        for attempt in range(self.max_retries):
            try:
                return await self._evaluate_once(topic, target_side, language, single_content, adversarial_content)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"[WARN] 评估失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, {wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(f"评估失败，已重试 {self.max_retries} 次: {e}") from e

    async def _evaluate_once(
        self,
        topic: str,
        target_side: str,
        language: str,
        single_content: str,
        adversarial_content: str,
    ) -> dict:
        started = time.perf_counter()

        if language == "zh":
            side_text = "正方" if target_side == "pro" else "反方"
            system_prompt = f"""你是一位资深的辩论裁判，专注于评估论点的逻辑质量。

【评估维度】（1-10分）
1. 推理自洽性：推理是否自洽，结论是否由前提合理推导，是否存在逻辑矛盾
2. 论证完整性：逻辑链条是否完整，有无跳跃或遗漏，前提与结论之间的连接是否清晰
3. 反反驳能力：能否识别并化解潜在的反驳观点，防御是否有力

【重要】只输出JSON，不要输出任何解释或 Markdown 格式。"""
        else:
            side_text = "Pro" if target_side == "pro" else "Con"
            system_prompt = f"""You are a senior debate judge, focusing on evaluating the logical quality of arguments.

【Evaluation Dimensions (1-10 points)】
1. Reasoning Coherence: Whether reasoning is self-consistent, whether conclusions are reasonably derived from premises, and whether there are logical contradictions
2. Argument Completeness: Whether the logical chain is complete, whether there are gaps or omissions, and whether the connection between premises and conclusion is clear
3. Counter-Defense Ability: Whether potential counter-arguments are identified and addressed, and whether defense is effective

【IMPORTANT】Output JSON only, without any explanation or Markdown formatting."""

        user_prompt = f"""
【辩题】{topic}
【立场】{side_text}

【方案一：单Agent论点】
{single_content}

【方案二：对抗性多Agent论点】
{adversarial_content}

请对两组论点在以下三个维度分别评分（1-10分），并给出总分和胜负判断。

输出JSON格式示例：
{{
  "single_scores": {{
    "reasoning_coherence": 8,
    "argument_completeness": 7,
    "counter_defense": 6
  }},
  "adversarial_scores": {{
    "reasoning_coherence": 9,
    "argument_completeness": 8,
    "counter_defense": 9
  }},
  "single_total": 7,
  "adversarial_total": 8.7,
  "winner": "adversarial",
  "winner_reasoning": "多Agent方案在推理自洽性和反反驳能力方面表现更优",
  "dimension_reasoning": {{
    "reasoning_coherence": {{
      "single": "论证基本自洽，但存在一处逻辑跳跃",
      "adversarial": "推理严密，从前提到结论逻辑清晰"
    }},
    "argument_completeness": {{
      "single": "逻辑链条基本完整，但缺少中间步骤",
      "adversarial": "论证结构完整，层次分明"
    }},
    "counter_defense": {{
      "single": "缺乏对反方观点的预判和防御",
      "adversarial": "主动识别并化解了潜在反驳"
    }}
  }}
}}"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2500,
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

        # 调试：打印原始返回
        print(f"[DEBUG] Judge raw response: {json.dumps(data, ensure_ascii=False)[:500]}...")

        # 安全提取 content - 出错时抛出异常触发重试
        if not data or not data.get("choices"):
            raise RuntimeError(f"Invalid API response: {data}")

        choice = data["choices"][0] if isinstance(data["choices"], list) and len(data["choices"]) > 0 else None
        if not choice:
            raise RuntimeError(f"Invalid choices in API response: {data}")

        message = choice.get("message") or {}
        content = message.get("content") or ""

        if not content:
            raise RuntimeError(f"Empty content in API response: {data}")

        try:
            result = json.loads(content)
            print(f"[DEBUG] Parsed JSON: {result}")
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parse failed: {e}")
            print(f"[DEBUG] Trying to extract from text...")
            result = self._parse_fallback(content, language)
            print(f"[DEBUG] Fallback result: {result}")

        duration = time.perf_counter() - started

        # 提取 token 使用情况
        usage = data.get("usage", {})

        # 计算各维度总分
        single_scores = result.get("single_scores", {})
        adversarial_scores = result.get("adversarial_scores", {})

        single_total = (
            (single_scores.get("reasoning_coherence", 5) +
             single_scores.get("argument_completeness", 5) +
             single_scores.get("counter_defense", 5)) / 3
        )
        adversarial_total = (
            (adversarial_scores.get("reasoning_coherence", 5) +
             adversarial_scores.get("argument_completeness", 5) +
             adversarial_scores.get("counter_defense", 5)) / 3
        )

        return {
            "single_scores": {
                "reasoning_coherence": int(single_scores.get("reasoning_coherence", 5)),
                "argument_completeness": int(single_scores.get("argument_completeness", 5)),
                "counter_defense": int(single_scores.get("counter_defense", 5)),
            },
            "adversarial_scores": {
                "reasoning_coherence": int(adversarial_scores.get("reasoning_coherence", 5)),
                "argument_completeness": int(adversarial_scores.get("argument_completeness", 5)),
                "counter_defense": int(adversarial_scores.get("counter_defense", 5)),
            },
            "single_total": round(single_total, 1),
            "adversarial_total": round(adversarial_total, 1),
            "dimension_reasoning": result.get("dimension_reasoning", {
                "reasoning_coherence": {"single": "", "adversarial": ""},
                "argument_completeness": {"single": "", "adversarial": ""},
                "counter_defense": {"single": "", "adversarial": ""}
            }),
            "winner": result.get("winner", "tie"),
            "winner_reasoning": result.get("winner_reasoning", ""),
            "total_duration_sec": round(duration, 3),
            "token_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    def _parse_fallback(self, content: str, language: str) -> dict:
        # 尝试提取各维度分数
        def extract_score(patterns):
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return min(10, max(1, int(match.group(1))))
            return 5

        single_rc = extract_score([r"single.*?推理自洽.*?(\d+)", r"单.*?推理自洽.*?(\d+)", r"reasoning_coherence.*?single.*?(\d+)"])
        single_ac = extract_score([r"single.*?论证完整.*?(\d+)", r"单.*?论证完整.*?(\d+)", r"argument_completeness.*?single.*?(\d+)"])
        single_cd = extract_score([r"single.*?反反驳.*?(\d+)", r"单.*?反反驳.*?(\d+)", r"counter_defense.*?single.*?(\d+)"])

        adv_rc = extract_score([r"adversarial.*?推理自洽.*?(\d+)", r"多.*?推理自洽.*?(\d+)", r"reasoning_coherence.*?adversarial.*?(\d+)"])
        adv_ac = extract_score([r"adversarial.*?论证完整.*?(\d+)", r"多.*?论证完整.*?(\d+)", r"argument_completeness.*?adversarial.*?(\d+)"])
        adv_cd = extract_score([r"adversarial.*?反反驳.*?(\d+)", r"多.*?反反驳.*?(\d+)", r"counter_defense.*?adversarial.*?(\d+)"])

        single_total = (single_rc + single_ac + single_cd) / 3
        adversarial_total = (adv_rc + adv_ac + adv_cd) / 3

        winner = "tie"
        if single_total > adversarial_total:
            winner = "single"
        elif adversarial_total > single_total:
            winner = "adversarial"

        return {
            "single_scores": {
                "reasoning_coherence": single_rc,
                "argument_completeness": single_ac,
                "counter_defense": single_cd,
            },
            "adversarial_scores": {
                "reasoning_coherence": adv_rc,
                "argument_completeness": adv_ac,
                "counter_defense": adv_cd,
            },
            "single_total": round(single_total, 1),
            "adversarial_total": round(adversarial_total, 1),
            "dimension_reasoning": {
                "reasoning_coherence": {"single": "", "adversarial": ""},
                "argument_completeness": {"single": "", "adversarial": ""},
                "counter_defense": {"single": "", "adversarial": ""}
            },
            "winner": winner,
            "winner_reasoning": "",
        }