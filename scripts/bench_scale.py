#!/usr/bin/env python3
"""规模化压测：按档灌入文档，每档测量 hybrid / vector 检索延迟。

与单规模脚本 bench_documents.py 不同，本脚本把检索延迟画成「规模函数」，
用于对比不同向量后端（LanceDB vs pgvector）在文档/chunk 增长下的退坡曲线。

用法：
    python3 scripts/bench_scale.py --base http://127.0.0.1:PORT [--key xxx]

行为：
    -- 连续灌入文档，目标档位为 STEPS（累计文档数）；每灌满一档就测量一次
       hybrid / vector 检索延迟（P50/P95/P99），并打印文档数 / chunk 数对照。
    -- 已存在的文档自动跳过（--resume）；--only-latency 只测量不导入。
"""

import argparse
import json
import time
import urllib.parse
import urllib.request

WORDS = ["Rust", "tokio", "async", "ownership", "borrowing", "SQLite", "LanceDB",
         "PostgreSQL", "pgvector", "embedding", "混合检索", "向量", "关键词",
         "BM25", "内存安全", "并发安全", "chunk", "索引", "查询延迟", "HNSW"]

QUERIES = ["Rust 所有权", "BM25 混合检索", "内存安全 并发", "SQLite 持久化",
           "PostgreSQL 向量索引"]

STEPS = [1000, 3000, 6000, 10000, 15000, 20000]


def build_note(size_bytes, words=WORDS):
    paragraph = " ".join(words) + " "
    sections = []
    current = []
    current_len = 0
    index = 0
    while True:
        section = f"## 章节 {index}\n{paragraph * 8}"
        index += 1
        if current_len + len(section) > size_bytes - 4000:
            sections.append("\n".join(current))
            if sum(len(s) for s in sections) >= size_bytes:
                break
            current = []
            current_len = 0
        current.append(section)
        current_len += len(section)
    body = "\n\n".join(sections)
    return ("# 压测文档\n" + body).encode("utf-8")


def request(url, method="GET", data=None, headers=None, timeout=600):
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload(base, key, filename, body):
    boundary = "----benchscale"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/markdown\r\n\r\n"
    ).encode("utf-8") + body + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if key:
        headers["x-api-key"] = key
    return request(base_url(base) + "/api/documents", "POST", payload, headers)


def base_url(base):
    return base.rstrip("/")


def hdrs(key):
    return {"x-api-key": key} if key else {}


def docs_summary(base, key):
    items = request(base + "/api/documents", headers=hdrs(key))["items"]
    ready = [d for d in items if d["status"] == "ready"]
    total_chars = sum(d.get("source_chars", 0) for d in ready)
    total_chunks = sum(d.get("chunk_count", 0) for d in ready)
    return len(ready), total_chars, total_chunks


def wait_ready(base, key, wanted, timeout=3600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = docs_summary(base, key)
        items = request(base + "/api/documents", headers=hdrs(key))["items"]
        failed = sum(1 for d in items if d["status"] == "failed")
        if failed:
            raise RuntimeError(f"{failed} doc(s) failed import")
        if ready >= wanted:
            return ready
        time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {wanted} ready (had {ready})")


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def measure_latency(base, key, label, rounds=3):
    ready, total_chars, total_chunks = docs_summary(base, key)
    line = (f"  [label] {label} : {ready} docs | {total_chunks} chunk | "
            f"源文本 {total_chars/1024/1024:.1f} MB\n")
    rows = {}
    for mode in ("hybrid", "vector"):
        xs = []
        for _ in range(rounds):
            for q in QUERIES:
                t0 = time.time()
                request(base + "/api/search?q=" + urllib.parse.quote(q) +
                        f"&mode={mode}&limit=5", headers=hdrs(key))
                xs.append(time.time() - t0)
        p50, p95, p99 = (percentile(xs, p) for p in (0.5, 0.95, 0.99))
        rows[mode] = (round(1e3*p50, 1), round(1e3*p95, 1), round(1e3*p99, 1))
        line += f"  {mode:6s} P50={1000*p50:6.1f}ms P95={1000*p95:6.1f}ms P99={1000*p99:6.1f}ms\n"
    line += (f"  ─ CSV → {label},{ready},{total_chunks},{rows['hybrid'][0]},"
             f"{rows['hybrid'][1]},{rows['hybrid'][2]},{rows['vector'][0]},"
             f"{rows['vector'][1]},{rows['vector'][2]}")
    print(line)
    return label, ready, total_chunks, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3010")
    ap.add_argument("--key", default=None)
    ap.add_argument("--size", type=int, default=8 * 1024)
    ap.add_argument("--step", type=int, default=30, help="每次灌入的批量")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only-latency", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"size={args.size//1024}KB/篇, 档位累计 {STEPS[-1]} docs, base={args.base}")
    request(args.base + "/health", headers=hdrs(args.key))
    print("health OK")

    if args.only_latency:
        measure_latency(args.base, args.key, "existing", args.rounds)
        return

    print("CSV header: label,docs,chunks,hybrid_p50,hybrid_p95,hybrid_p99,vector_p50,vector_p95,vector_p99")
    for stage in STEPS:
        ready, _, _ = docs_summary(args.base, args.key)
        if ready >= stage:
            measure_latency(args.base, args.key, f"@docs={stage}", args.rounds)
            continue
        print(f"\n==> 灌入 -> {stage} 文档（现有 {ready}）")
        start = time.time()
        while True:
            ready, _, _ = docs_summary(args.base, args.key)
            if ready >= stage:
                break
            need = stage - ready
            n = min(need, args.step)
            for i in range(n):
                upload(args.base, args.key,
                       f"s{stage}-{int(time.time())%1000}-{i}.md", build_note(args.size))
            wait_ready(args.base, args.key, ready + n)
        wall = time.time() - start
        print(f"  本档导入耗时 {wall:.1f}s")
        measure_latency(args.base, args.key, f"@docs:{stage}", args.rounds)


if __name__ == "__main__":
    main()