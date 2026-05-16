import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.agents.debate import DebateOrchestrator
from backend.app.config import ROOT_DIR, get_settings
from backend.app.core.judge import JudgeClient
from backend.app.core.ollama import OllamaClient
from backend.app.models import EvaluationRequest, GenerationRequest, Language, Topic


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
    llm = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        enable_thinking=False,
    )
    return DebateOrchestrator(llm, settings.cache_dir)


def load_topics() -> list[Topic]:
    path = Path(__file__).parent / "data" / "topics.json"
    return [Topic(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


@app.get("/api/health")
async def health():
    settings = get_settings()
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model, enable_thinking=False)
    return {
        "app": "ok",
        "provider": "ollama",
        "model": settings.ollama_model,
        "thinking_enabled": False,
        "ollama": await llm.health(),
        "cache": _generation_cache_status(settings.cache_dir),
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


@app.post("/api/evaluate")
async def evaluate(request: EvaluationRequest):
    settings = get_settings()
    judge = JudgeClient(settings.judge_api_key, settings.judge_model, settings.judge_base_url)
    result = await judge.evaluate(
        topic=request.topic,
        target_side=request.target_side,
        language=request.language,
        single_content=request.single_content,
        adversarial_content=request.adversarial_content,
    )
    return result


@app.get("/")
async def root():
    return {"message": "Debate Argument Generator API", "docs": f"{ROOT_DIR}/README.md"}


def _generation_cache_status(cache_dir: Path) -> dict[str, object]:
    path = cache_dir / "generation_cache.json"
    if not path.exists():
        return {"path": str(path), "entries": 0}
    try:
        entries = len(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        entries = 0
    return {"path": str(path), "entries": entries}
