import json

import pytest

from backend.app.core.deepseek import DeepSeekClient


class FakeStreamResponse:
    def __init__(self):
        self.lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}],"usage":null}',
            'data: {"choices":[{"delta":{"content":" world"}}],"usage":null}',
            'data: {"choices":[],"usage":{"total_tokens":12}}',
            "data: [DONE]",
        ]

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


@pytest.mark.asyncio
async def test_deepseek_stream_collects_tokens(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, *args, **kwargs):
            assert kwargs["json"]["stream"] is True
            return FakeStreamResponse()

    monkeypatch.setattr("backend.app.core.deepseek.httpx.AsyncClient", FakeAsyncClient)
    tokens = []
    client = DeepSeekClient("test-key")

    async def on_token(token):
        tokens.append(token)

    response = await client.chat([{"role": "user", "content": "hi"}], on_token=on_token)
    assert response.content == "Hello world"
    assert tokens == ["Hello", " world"]
    assert response.token_estimate == 12

