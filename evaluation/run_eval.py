import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
TOPICS_PATH = ROOT / "backend" / "app" / "data" / "topics.json"
RESULTS_DIR = ROOT / "evaluation" / "results"


async def run_one(client: httpx.AsyncClient, topic: dict, language: str, side: str) -> dict:
    topic_text = topic["topic_en"] if language == "en" else topic["topic_zh"]
    response = await client.post(
        "/api/generate",
        json={
            "topic": topic_text,
            "target_side": side,
            "language": language,
            "use_search": True,
            "use_cache": True,
        },
        timeout=600.0,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "topic_id": topic["id"],
        "domain": topic["domain"],
        "language": language,
        "target_side": side,
        "topic": topic_text,
        "result": data,
    }


def flatten_row(item: dict) -> dict:
    metrics = item["result"]["metrics"]
    return {
        "topic_id": item["topic_id"],
        "domain": item["domain"],
        "language": item["language"],
        "target_side": item["target_side"],
        "topic": item["topic"],
        "total_duration_sec": metrics["total_duration_sec"],
        "single_duration_sec": metrics["single_duration_sec"],
        "adversarial_duration_sec": metrics["adversarial_duration_sec"],
        "token_estimate": metrics["token_estimate"],
        "source_count": metrics["source_count"],
        "single_argument_count": metrics["single_argument_count"],
        "adversarial_argument_count": metrics["adversarial_argument_count"],
        "single_diversity": metrics["single_diversity"],
        "adversarial_diversity": metrics["adversarial_diversity"],
        "human_single_score": "",
        "human_adversarial_score": "",
        "human_notes": "",
    }


def write_outputs(items: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "metrics.csv"
    md_path = output_dir / "summary.md"

    jsonl_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )

    rows = [flatten_row(item) for item in items]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Lightweight Evaluation Summary",
        "",
        "| Topic | Lang | Side | Single sec | Adversarial sec | Sources | Single diversity | Adv diversity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['topic_id']} | {row['language']} | {row['target_side']} | "
            f"{row['single_duration_sec']} | {row['adversarial_duration_sec']} | "
            f"{row['source_count']} | {row['single_diversity']} | {row['adversarial_diversity']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight evaluation for the debate generator.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--side", choices=["pro", "con"], default="pro")
    args = parser.parse_args()

    topics = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))[: args.limit]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = RESULTS_DIR / stamp

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        items = []
        for topic in topics:
            print(f"Running {topic['id']}...")
            items.append(await run_one(client, topic, args.language, args.side))

    write_outputs(items, output_dir)
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

