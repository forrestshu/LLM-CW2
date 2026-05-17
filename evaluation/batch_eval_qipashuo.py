import argparse
import asyncio
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
QIPASHUO_FILE = ROOT / "奇葩说争议热点辩题30条.md"
RESULTS_DIR = ROOT / "evaluation" / "results"


def parse_qipashuo_topics() -> list[dict]:
    """解析奇葩说30条辩题文件"""
    content = QIPASHUO_FILE.read_text(encoding="utf-8")
    # 提取表格中的辩题
    pattern = r'\|\s*\d+\s*\|\s*([^|]+)\s*\|'
    matches = re.findall(pattern, content)
    # 清理和去重
    topics = [{"id": str(i + 1), "topic": match.strip()} for i, match in enumerate(matches) if match.strip()]
    return topics


async def run_generation(client: httpx.AsyncClient, topic: str, target_side: str = "pro") -> dict:
    """生成论点"""
    response = await client.post(
        "/api/generate",
        json={
            "topic": topic,
            "target_side": target_side,
            "language": "zh",
            "use_cache": True,
        },
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()


async def run_evaluation(client: httpx.AsyncClient, topic: str, single_content: str, adversarial_content: str) -> dict:
    """评估论点"""
    response = await client.post(
        "/api/evaluate",
        json={
            "topic": topic,
            "target_side": "pro",
            "language": "zh",
            "single_content": single_content,
            "adversarial_content": adversarial_content,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    return response.json()


def flatten_result(item: dict) -> dict:
    """扁平化评估结果用于CSV"""
    single_scores = item.get("single_scores", {})
    adversarial_scores = item.get("adversarial_scores", {})
    token_usage = item.get("token_usage", {})
    metrics = item.get("metrics", {})

    return {
        "辩题ID": item["id"],
        "辩题": item["topic"],
        "推理自洽性_单Agent": single_scores.get("reasoning_coherence", ""),
        "推理自洽性_多Agent": adversarial_scores.get("reasoning_coherence", ""),
        "论证完整性_单Agent": single_scores.get("argument_completeness", ""),
        "论证完整性_多Agent": adversarial_scores.get("argument_completeness", ""),
        "反反驳能力_单Agent": single_scores.get("counter_defense", ""),
        "反反驳能力_多Agent": adversarial_scores.get("counter_defense", ""),
        "总分_单Agent": item.get("single_total", ""),
        "总分_多Agent": item.get("adversarial_total", ""),
        "胜者": item.get("winner", ""),
        "胜者理由": item.get("winner_reasoning", ""),
        "生成耗时_单Agent(秒)": metrics.get("single_duration_sec", ""),
        "生成耗时_多Agent(秒)": metrics.get("adversarial_duration_sec", ""),
        "输入Token": token_usage.get("prompt_tokens", ""),
        "输出Token": token_usage.get("completion_tokens", ""),
        "评委总Token": token_usage.get("total_tokens", ""),
        "单Agent_生成Token": item.get("single_agent_tokens", ""),
        "多Agent_生成Token": item.get("adversarial_tokens", ""),
    }


def calculate_statistics(results: list[dict]) -> dict:
    """计算统计指标"""
    if not results:
        return {}

    dims = ["reasoning_coherence", "argument_completeness", "counter_defense"]
    stats = {
        "total_count": len(results),
        "single_wins": sum(1 for r in results if r.get("winner") == "single"),
        "adversarial_wins": sum(1 for r in results if r.get("winner") == "adversarial"),
        "ties": sum(1 for r in results if r.get("winner") == "tie"),
        "averages": {
            "single": {},
            "adversarial": {},
        },
    }

    for dim in dims:
        single_vals = [r.get("single_scores", {}).get(dim, 0) for r in results]
        adv_vals = [r.get("adversarial_scores", {}).get(dim, 0) for r in results]

        if single_vals:
            stats["averages"]["single"][dim] = round(sum(single_vals) / len(single_vals), 2)
        if adv_vals:
            stats["averages"]["adversarial"][dim] = round(sum(adv_vals) / len(adv_vals), 2)

    # 总分平均值
    single_totals = [r.get("single_total", 0) for r in results]
    adv_totals = [r.get("adversarial_total", 0) for r in results]

    if single_totals:
        stats["averages"]["single"]["total"] = round(sum(single_totals) / len(single_totals), 2)
    if adv_totals:
        stats["averages"]["adversarial"]["total"] = round(sum(adv_totals) / len(adv_totals), 2)

    return stats


def write_outputs(results: list[dict], output_dir: Path) -> None:
    """输出结果文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 详细结果
    json_path = output_dir / "qipashuo_results.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV 表格
    csv_path = output_dir / "qipashuo_results.csv"
    rows = [flatten_result(r) for r in results]

    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 统计摘要
    stats = calculate_statistics(results)
    stats_path = output_dir / "qipashuo_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown 摘要
    md_path = output_dir / "qipashuo_summary.md"
    lines = [
        "# 奇葩说30条辩题评估摘要",
        "",
        f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总辩题数: {stats['total_count']}",
        "",
        "## 胜负统计",
        f"- 单 Agent 胜: {stats['single_wins']} ({stats['single_wins']/stats['total_count']*100:.1f}%)",
        f"- 多 Agent 胜: {stats['adversarial_wins']} ({stats['adversarial_wins']/stats['total_count']*100:.1f}%)",
        f"- 平局: {stats['ties']} ({stats['ties']/stats['total_count']*100:.1f}%)",
        "",
        "## 平均得分对比",
        "",
        "| 维度 | 单 Agent | 多 Agent | 差异 |",
        "| --- | ---: | ---: | ---: |",
    ]

    dim_names = {
        "reasoning_coherence": "推理自洽性",
        "argument_completeness": "论证完整性",
        "counter_defense": "反反驳能力",
        "total": "总分",
    }

    for dim, name in dim_names.items():
        s_avg = stats['averages']['single'].get(dim, 0)
        a_avg = stats['averages']['adversarial'].get(dim, 0)
        diff = a_avg - s_avg
        lines.append(f"| {name} | {s_avg} | {a_avg} | {diff:+.2f} |")

    # Token 和时间统计
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        total_eval_duration = sum(r.get("total_duration_sec", 0) for r in valid_results)
        total_eval_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in valid_results)
        
        single_durations = sum(r.get("metrics", {}).get("single_duration_sec", 0) for r in valid_results)
        adv_durations = sum(r.get("metrics", {}).get("adversarial_duration_sec", 0) for r in valid_results)
        
        single_tokens = sum(r.get("single_agent_tokens", 0) for r in valid_results)
        adv_tokens = sum(r.get("adversarial_tokens", 0) for r in valid_results)

        lines.extend([
            "",
            "## 资源消耗 (生成阶段)",
            f"- 单 Agent 总耗时: {single_durations:.1f} 秒",
            f"- 单 Agent 平均耗时: {single_durations/len(valid_results):.1f} 秒",
            f"- 多 Agent 总耗时: {adv_durations:.1f} 秒",
            f"- 多 Agent 平均耗时: {adv_durations/len(valid_results):.1f} 秒",
            f"- 单 Agent 总 Token: {single_tokens:,}",
            f"- 单 Agent 平均 Token: {single_tokens/len(valid_results):.0f}",
            f"- 多 Agent 总 Token: {adv_tokens:,}",
            f"- 多 Agent 平均 Token: {adv_tokens/len(valid_results):.0f}",
            "",
            "## 资源消耗 (评估阶段)",
            f"- 总评估时间: {total_eval_duration:.1f} 秒",
            f"- 平均每题耗时: {total_eval_duration/len(valid_results):.1f} 秒",
            f"- 总评估 Token: {total_eval_tokens:,}",
            f"- 平均每题评估 Token: {total_eval_tokens/len(valid_results):.0f}",
        ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="批量评估奇葩说30条辩题")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None, help="限制评估数量")
    parser.add_argument("--start", type=int, default=1, help="从第几条开始")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--delay", type=int, default=5, help="每条辩题评估后的延迟秒数（避免 429 错误）")
    args = parser.parse_args()

    # 解析辩题
    topics = parse_qipashuo_topics()
    print(f"找到 {len(topics)} 条辩题")

    # 应用限制
    if args.limit:
        topics = topics[args.start - 1 : args.start - 1 + args.limit]
    else:
        topics = topics[args.start - 1 :]

    print(f"将评估 {len(topics)} 条辩题（从第 {args.start} 条开始）")

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = RESULTS_DIR / f"qipashuo_{stamp}"

    # 批量评估
    results = []
    max_topic_retries = 3  # 每个辩题最多重试次数

    async with httpx.AsyncClient(base_url=args.base_url, timeout=600.0) as client:
        for i, topic in enumerate(topics, start=1):
            print(f"\n[{i}/{len(topics)}] 评估: {topic['topic']}")

            # 重试当前辩题
            for retry in range(max_topic_retries):
                try:
                    # 生成论点
                    gen_result = await run_generation(client, topic["topic"])

                    # 评估
                    eval_result = await run_evaluation(
                        client,
                        topic["topic"],
                        gen_result["single_agent"]["content"],
                        gen_result["adversarial"]["content"],
                    )

                    results.append({
                        "id": topic["id"],
                        "topic": topic["topic"],
                        "metrics": gen_result.get("metrics", {}),
                        "single_agent_tokens": gen_result.get("single_agent", {}).get("token_estimate", 0),
                        "adversarial_tokens": gen_result.get("adversarial", {}).get("token_estimate", 0) + sum(
                            item.get("metadata", {}).get("token_estimate", 0) or 0
                            for item in gen_result.get("transcript", [])
                        ),
                        **eval_result,
                    })

                    print(f"  单Agent: {eval_result['single_total']}, 多Agent: {eval_result['adversarial_total']}, 胜者: {eval_result['winner']}")

                    # 评估成功后，等待一段时间再评估下一个（避免 429 错误）
                    if i < len(topics):
                        print(f"  等待 {args.delay} 秒后继续...")
                        await asyncio.sleep(args.delay)

                    break  # 成功，跳出重试循环

                except Exception as e:
                    if retry < max_topic_retries - 1:
                        wait_time = 2 ** retry
                        print(f"  错误 (尝试 {retry + 1}/{max_topic_retries}): {e}")
                        print(f"  {wait_time}秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  错误: 已重试 {max_topic_retries} 次后仍失败: {e}")
                        results.append({
                            "id": topic["id"],
                            "topic": topic["topic"],
                            "error": str(e),
                        })

    # 输出结果
    write_outputs(results, output_dir)
    print(f"\n结果已保存到: {output_dir}")

    # 打印统计
    stats = calculate_statistics([r for r in results if "error" not in r])
    if stats:
        print(f"\n统计摘要:")
        print(f"  单Agent胜: {stats['single_wins']}, 多Agent胜: {stats['adversarial_wins']}, 平局: {stats['ties']}")
        print(f"  单Agent平均分: {stats['averages']['single'].get('total', 0)}")
        print(f"  多Agent平均分: {stats['averages']['adversarial'].get('total', 0)}")


if __name__ == "__main__":
    asyncio.run(main())