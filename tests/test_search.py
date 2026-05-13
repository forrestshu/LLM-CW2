import pytest

from backend.app.tools.search import SearchError, TavilySearch


@pytest.mark.asyncio
async def test_search_without_key_raises_clear_error(tmp_path):
    search = TavilySearch(api_key=None, cache_dir=tmp_path)
    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        await search.search("AI legal personhood")


@pytest.mark.asyncio
async def test_search_uses_cache_without_key(tmp_path):
    search = TavilySearch(api_key=None, cache_dir=tmp_path)
    cache_key = search._key("cached query", 5)
    search.cache_path.write_text(
        '{"%s":[{"title":"Cached","url":"https://example.com","snippet":"Snippet"}]}' % cache_key,
        encoding="utf-8",
    )
    results = await search.search("cached query")
    assert results[0].title == "Cached"

