import hashlib
import json
from pathlib import Path

from tavily import TavilyClient

from backend.app.models import Source


class SearchError(RuntimeError):
    pass


class TavilySearch:
    def __init__(self, api_key: str | None, cache_dir: Path):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_path = cache_dir / "tavily_cache.json"
        self._cache: dict[str, list[dict[str, str]]] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def cache_status(self) -> dict[str, object]:
        cache = self._load_cache()
        return {"path": str(self.cache_path), "entries": len(cache)}

    async def search(self, query: str, max_results: int = 5, use_cache: bool = True) -> list[Source]:
        if not query.strip():
            return []
        key = self._key(query, max_results)
        if use_cache:
            cached = self._load_cache().get(key)
            if cached is not None:
                return [Source(**item) for item in cached]
        if not self.api_key:
            raise SearchError("TAVILY_API_KEY is not configured. Add it to .env or disable live search.")

        client = TavilyClient(api_key=self.api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        sources = [
            Source(
                title=item.get("title") or item.get("url") or "Untitled source",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
            )
            for item in response.get("results", [])
            if item.get("url")
        ]
        if use_cache:
            cache = self._load_cache()
            cache[key] = [source.model_dump() for source in sources]
            self._save_cache(cache)
        return sources

    def _load_cache(self) -> dict[str, list[dict[str, str]]]:
        if self._cache is not None:
            return self._cache
        if not self.cache_path.exists():
            self._cache = {}
            return self._cache
        try:
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._cache = {}
        return self._cache

    def _save_cache(self, cache: dict[str, list[dict[str, str]]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _key(query: str, max_results: int) -> str:
        digest = hashlib.sha256(f"{query}|{max_results}".encode("utf-8")).hexdigest()
        return digest[:32]

