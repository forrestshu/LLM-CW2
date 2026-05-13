import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.agents.debate import DebateOrchestrator
from backend.app.config import ROOT_DIR, get_settings
from backend.app.core.deepseek import DeepSeekClient
from backend.app.core.ollama import OllamaClient
from backend.app.models import GenerationRequest, Language, Topic
from backend.app.tools.search import TavilySearch


app = FastAPI(title="DTS407TC A2 Debate Argument Generator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator() -> DebateOrchestrator:
    settings = get_settings()
    if settings.llm_provider.lower() == "deepseek":
        llm = DeepSeekClient(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
        )
    else:
        llm = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            enable_thinking=settings.ollama_enable_thinking,
        )
    search = TavilySearch(settings.tavily_api_key, settings.cache_dir)
    return DebateOrchestrator(llm, search)


def load_topics() -> list[Topic]:
    path = Path(__file__).parent / "data" / "topics.json"
    return [Topic(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


@app.get("/api/health")
async def health():
    settings = get_settings()
    if settings.llm_provider.lower() == "deepseek":
        llm = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model)
        model = settings.deepseek_model
    else:
        llm = OllamaClient(settings.ollama_base_url, settings.ollama_model)
        model = settings.ollama_model
    search = TavilySearch(settings.tavily_api_key, settings.cache_dir)
    return {
        "app": "ok",
        "provider": settings.llm_provider,
        "model": model,
        "thinking_enabled": settings.ollama_enable_thinking,
        "ollama": await llm.health(),
        "tavily_configured": search.configured,
        "cache": search.cache_status(),
    }


@app.get("/api/topics")
async def topics(language: Language = Query(default="en")):
    topic_list = load_topics()
    return [
        {
            **topic.model_dump(),
            "label": topic.topic_en if language == "en" else topic.topic_zh,
        }
        for topic in topic_list
    ]


@app.post("/api/generate")
async def generate(request: GenerationRequest):
    return await get_orchestrator().generate(request)


@app.post("/api/generate/stream")
async def generate_stream(request: GenerationRequest):
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def emit(event: str, payload: dict) -> None:
        await queue.put({"event": event, "payload": payload})

    async def run_generation() -> None:
        try:
            await get_orchestrator().generate(request, emit=emit)
        except Exception as exc:  # pragma: no cover - surfaced to frontend
            await queue.put({"event": "error", "payload": {"message": str(exc)}})
        finally:
            await queue.put({"event": "done", "payload": {}})

    async def event_iterator() -> AsyncIterator[str]:
        task = asyncio.create_task(run_generation())
        while True:
            item = await queue.get()
            yield f"event: {item['event']}\ndata: {json.dumps(item['payload'], ensure_ascii=False)}\n\n"
            if item["event"] == "done":
                break
        await task

    return StreamingResponse(event_iterator(), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"message": "Debate Argument Generator API", "docs": f"{ROOT_DIR}/README.md"}
