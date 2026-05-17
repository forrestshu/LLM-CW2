import time
import asyncio
from dataclasses import dataclass

import httpx

from backend.app.core.metrics import estimate_tokens, strip_thinking


@dataclass
class LLMResponse:
    content: str
    duration_sec: float
    token_estimate: int


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0, enable_thinking: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self._lock = asyncio.Lock()

    async def health(self) -> dict[str, object]:
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                names = [item.get("name") for item in data.get("models", [])]
                return {"available": True, "model_found": self.model in names, "models": names}
        except Exception as exc:  # pragma: no cover - defensive health report
            return {"available": False, "model_found": False, "error": str(exc), "models": []}

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        thinking: bool = False,
        on_token=None,
    ) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": messages,
            "think": bool(thinking and self.enable_thinking),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with self._lock:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        content = data.get("message", {}).get("content", "")
        cleaned = strip_thinking(content)
        if on_token and cleaned:
            await on_token(cleaned)
        token_estimate = int(
            (data.get("prompt_eval_count") or 0)
            + (data.get("eval_count") or 0)
            or estimate_tokens(cleaned)
        )
        return LLMResponse(
            content=cleaned,
            duration_sec=round(time.perf_counter() - started, 3),
            token_estimate=token_estimate,
        )
