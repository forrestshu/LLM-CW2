import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from backend.app.core.metrics import estimate_tokens, strip_thinking


TokenCallback = Callable[[str], Awaitable[None]]


@dataclass
class DeepSeekResponse:
    content: str
    duration_sec: float
    token_estimate: int


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        timeout: float = 300.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def health(self) -> dict[str, object]:
        return {
            "available": bool(self.api_key),
            "model_found": bool(self.api_key),
            "models": [self.model] if self.api_key else [],
            "provider": "deepseek",
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        thinking: bool = False,
        on_token: TokenCallback | None = None,
    ) -> DeepSeekResponse:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured. Add it to .env before generating.")

        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": bool(on_token),
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if on_token:
            payload["stream_options"] = {"include_usage": True}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if on_token:
            content, token_estimate = await self._stream(payload, headers, on_token)
        else:
            content, token_estimate = await self._complete(payload, headers)

        cleaned = strip_thinking(content)
        return DeepSeekResponse(
            content=cleaned,
            duration_sec=round(time.perf_counter() - started, 3),
            token_estimate=token_estimate or estimate_tokens(cleaned),
        )

    async def _complete(self, payload: dict, headers: dict[str, str]) -> tuple[str, int]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return content, int(usage.get("total_tokens") or 0)

    async def _stream(
        self,
        payload: dict,
        headers: dict[str, str],
        on_token: TokenCallback,
    ) -> tuple[str, int]:
        chunks: list[str] = []
        total_tokens = 0
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    usage = event.get("usage")
                    if usage:
                        total_tokens = int(usage.get("total_tokens") or total_tokens)
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        token = delta.get("content") or ""
                        if token:
                            chunks.append(token)
                            await on_token(token)
        content = "".join(chunks)
        return content, total_tokens
