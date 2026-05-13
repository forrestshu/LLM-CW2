import re
from itertools import combinations


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ARGUMENT_HEADING_RE = re.compile(
    r"(?im)^\s*(?:argument|论点)\s*(?:one|two|three|一|二|三|[1-3])\b|^\s*(?:#{1,4}|\d+[.)、])\s+"
)


def strip_thinking(text: str) -> str:
    cleaned = THINK_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    punctuation = re.findall(r"[^\w\s\u4e00-\u9fff]", text)
    return max(1, len(ascii_words) + len(cjk_chars) + len(punctuation) // 3)


def count_arguments(text: str) -> int:
    matches = ARGUMENT_HEADING_RE.findall(text or "")
    if matches:
        return min(3, len(matches))
    return min(3, len(re.findall(r"(?i)core claim|核心主张", text or "")))


def char_ngram_diversity(text: str, n: int = 3) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) <= n:
        return 0.0
    grams = [compact[i : i + n] for i in range(len(compact) - n + 1)]
    return round(len(set(grams)) / len(grams), 4)


def section_diversity(sections: list[str]) -> float:
    vectors = [_ngram_set(section) for section in sections if section.strip()]
    if len(vectors) < 2:
        return char_ngram_diversity(" ".join(sections))
    similarities = []
    for left, right in combinations(vectors, 2):
        union = left | right
        similarities.append(len(left & right) / len(union) if union else 0.0)
    return round(1.0 - (sum(similarities) / len(similarities)), 4)


def split_argument_sections(text: str) -> list[str]:
    chunks = re.split(r"(?im)^\s*(?:argument|论点)\s*(?:one|two|three|一|二|三|[1-3])[:：.\s-]*", text or "")
    sections = [chunk.strip() for chunk in chunks if chunk.strip()]
    return sections[:3] if sections else [text or ""]


def _ngram_set(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text or "")
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}

