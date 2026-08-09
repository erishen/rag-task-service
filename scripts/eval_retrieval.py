#!/usr/bin/env python3
"""RAG 检索评估：对比 vector / hybrid（/ rerank）检索模式的召回质量。

读 `fixtures/eval.json` 里带期望文档标注的查询集，对每个查询跑一次检索，
统计 recall@k、hit-rate 与 MRR，并按模式汇总对比。另跑一个 metadata 过滤
场景，验证 `documents=` 参数能把检索范围圈定到指定文档。

假设服务已在运行且样例文档已上传（可用 `make eval` 一键起临时环境）。

用法:
    python3 scripts/eval_retrieval.py --base http://127.0.0.1:3000 --key your-secret-key
    python3 scripts/eval_retrieval.py --base http://127.0.0.1:3000 --key k --fixtures fixtures/eval.json --k 3
    python3 scripts/eval_retrieval.py --key k --min-recall 0.6   # 任一模式 recall@k 低于阈值时退出非零
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"

MODES = ["vector", "hybrid"]
# 每两个请求之间的最小间隔（秒），避免撞上服务端每密钥限流。默认 10 req/s 限额下足够宽松。
QUERY_DELAY = 0.15


def api_get(base: str, path: str, key: str | None) -> dict:
    url = base.rstrip("/") + path
    headers = {"x-api-key": key} if key else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:300]}") from e


def load_queries(path: str) -> tuple[int, list[dict]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return int(data.get("top_k", 3)), data["queries"]


def run_query(base: str, key: str | None, q: str, mode: str, k: int, documents: str | None = None) -> list[dict]:
    params = {"q": q, "mode": mode, "limit": str(k)}
    if documents:
        params["documents"] = documents
    query = urllib.parse.urlencode(params)
    return api_get(base, f"/api/search?{query}", key)["items"]


def expected_position(items: list[dict], expected_ids: set[int]) -> int | None:
    for i, item in enumerate(items):
        if item["document_id"] in expected_ids:
            return i
    return None


def metric_block(base: str, key: str | None, k: int, docs_by_name: dict[str, int]) -> dict[str, float]:
    """跑所有模式并打印对比表。返回每个模式的 recall@k。"""
    hits: dict[str, int] = {m: 0 for m in MODES}
    recalls: dict[str, list[float]] = {m: [] for m in MODES}
    mrrs: dict[str, list[float]] = {m: [] for m in MODES}
    n = len(QUERIES)

    print(f"\n{BOLD}{'=' * 64}{RESET}")
    print(f"{BOLD}检索评估{RESET}  查询数={n}  top_k={k}  模式={', '.join(MODES)}")
    print(f"{BOLD}{'=' * 64}{RESET}")
    print(f"{'查询':<24}{'期望文档':<20}" + "".join(f"{m:>16}" for m in MODES))

    for entry in QUERIES:
        q = entry["q"]
        expected_ids = {docs_by_name[name] for name in entry["expected"]}
        row_positions: dict[str, int | None] = {}
        for mode in MODES:
            items = run_query(base, key, q, mode, k)
            pos = expected_position(items, expected_ids)
            recalls[mode].append(1.0 if pos is not None else 0.0)
            if pos is not None:
                hits[mode] += 1
                mrrs[mode].append(1.0 / (pos + 1))
            else:
                mrrs[mode].append(0.0)
            row_positions[mode] = pos
        time.sleep(QUERY_DELAY)
        cells = "".join(
            f"{('#' + str(p + 1)) if p is not None else '-':>16}" for p in row_positions.values()
        )
        print(f"{q:<24}{','.join(entry['expected']):<20}{cells}")

    recall_at_k = {m: sum(recalls[m]) / n for m in MODES}
    print(f"\n{'指标':<24}" + "".join(f"{m:>16}" for m in MODES))
    for label, values, fmt in (
        ("hit-rate (命中查询)", [hits[m] / n for m in MODES], "{:.0%}"),
        ("recall@k", [recall_at_k[m] for m in MODES], "{:.0%}"),
        ("MRR", [sum(mrrs[m]) / n for m in MODES], "{:.2f}"),
    ):
        cells = "".join(f"{fmt.format(v):>16}" for v in values)
        print(f"{label:<24}{cells}")
    return recall_at_k


def filter_scenario(base: str, key: str | None, k: int, docs_by_name: dict[str, int]) -> None:
    """metadata 过滤场景：把检索圈定到单文档，结果必须全部属于该文档。"""
    print(f"\n{BOLD}Metadata 过滤场景{RESET}")
    if "doc-cook.md" not in docs_by_name or "doc-rust.md" not in docs_by_name:
        print(f"{YELLOW}  样例文档不完整，跳过{RESET}")
        return False
    queries = [
        ("红烧肉", "doc-cook.md"),
        ("内存安全", "doc-rust.md"),
    ]
    ok = True
    for q, expected in queries:
        expect_id = docs_by_name[expected]
        leaked_other = 0
        total = 0
        for mode in MODES:
            items = run_query(base, key, q, mode, k, documents=str(expect_id))
            total += len(items)
            leaked_other += sum(1 for it in items if it["document_id"] != expect_id)
            time.sleep(QUERY_DELAY)
        if leaked_other:
            ok = False
        status = f"{GREEN}✅{RESET}" if leaked_other == 0 else f"{RED}❌{RESET}"
        print(f"  {status} q='{q}' 过滤到 {expected} -> 命中 {total} 条, 越界 {leaked_other} 条")
    return ok


def main() -> None:
    global QUERIES
    parser = argparse.ArgumentParser(description="RAG 检索评估")
    parser.add_argument("--base", default="http://127.0.0.1:3000", help="服务地址")
    parser.add_argument("--key", default=None, help="x-api-key（缺省从环境变量 API_KEY 读）")
    parser.add_argument("--fixtures", default="fixtures/eval.json", help="评估集文件")
    parser.add_argument("--k", type=int, default=0, help="覆盖 top_k")
    parser.add_argument("--min-recall", type=float, default=1.0,
                        help="任一模式 recall@k 低于此阈值即退出非零（0~1，缺省 1.0）")
    args = parser.parse_args()
    key = args.key or os.environ.get("API_KEY")
    if not key:
        raise SystemExit("需要 --key 或环境变量 API_KEY")
    if not 0 <= args.min_recall <= 1:
        raise SystemExit("--min-recall 须在 [0, 1] 内")

    k, QUERIES = load_queries(args.fixtures)
    if args.k:
        k = args.k

    docs = api_get(args.base, "/api/documents", key)
    docs_by_name = {item["name"]: int(item["id"]) for item in docs.get("items", [])}
    if not docs_by_name:
        raise SystemExit("服务上没有文档。先 `make rag-fixtures` 或 `make eval` 上传样例。")
    missing = {name for entry in QUERIES for name in entry["expected"]} - set(docs_by_name)
    if missing:
        raise SystemExit(f"服务缺少评估集期望的文档: {sorted(missing)}")

    recall_at_k = metric_block(args.base, key, k, docs_by_name)
    filter_ok = filter_scenario(args.base, key, k, docs_by_name)

    failures: list[str] = []
    for mode in MODES:
        if recall_at_k[mode] < args.min_recall:
            failures.append(
                f"{mode} recall@k={recall_at_k[mode]:.0%} < 阈值 {args.min_recall:.0%}"
            )
    if not filter_ok:
        failures.append("metadata 过滤场景存在越界召回")
        print(f"{RED}❌ metadata 过滤失败：结果越出了 documents= 限定的范围{RESET}")

    print(f"\n{BOLD}{'=' * 64}{RESET}")
    if failures:
        color, tag = RED, "失败"
        print(f"{BOLD}总结: {color}{tag}{RESET} —— " + "; ".join(failures))
        print(f"{BOLD}{'=' * 64}{RESET}")
        sys.exit(1)
    color, tag = GREEN, "全部通过"
    print(f"{BOLD}总结: {color}{tag}{RESET}（recall@k ≥ {args.min_recall:.0%}，过滤无越界）")
    print(f"{BOLD}{'=' * 64}{RESET}")


if __name__ == "__main__":
    QUERIES: list[dict] = []
    main()
