#!/usr/bin/env python3
"""文档导入与检索压测脚本。

统计口径：
- 导入吞吐：分批上传 → 轮询到全部 ready 的总耗时（doc/s 与近似 chunk 规模）。
- 单 worker 串行队列表现：并发扇出 16 请求的上传耗时。
- 检索延迟：hybrid / vector 各跑多轮，输出 P50 / P95 / P99。
- 收尾清点数据库规模（文档数 / chunk 趋势由状态 size 反映）。

用法：
    python3 scripts/bench_documents.py --base http://127.0.0.1:PORT [--key xxx]
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

WORDS = ["Rust", "tokio", "async", "ownership", "borrowing", "SQLite", "LanceDB",
         "embedding", "混合检索", "向量", "关键词", "BM25", "内存安全", "并发安全", "chunk"]


def build_note(size_bytes, words=WORDS):
    """按 Markdown 标题分节构造接近目标体积的文档（每节一个标题 + 若干段落）。"""
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


def request(url, method="GET", data=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload(base, key, filename, body):
    boundary = "----benchboundary"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/markdown\r\n\r\n"
    ).encode("utf-8") + body + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if key:
        headers["x-api-key"] = key
    return request(base + "/api/documents", "POST", payload, headers)


def wait_ready(base, key, wanted, timeout=1800):
    deadline = time.time() + timeout
    hdrs = {"x-api-key": key} if key else {}
    while time.time() < deadline:
        page = request(base + "/api/documents", headers=hdrs)
        ready = sum(1 for d in page["items"] if d["status"] == "ready")
        failed = sum(1 for d in page["items"] if d["status"] == "failed")
        if failed:
            print(f"  ERROR: {failed} doc(s) failed import, aborting case")
            return
        if ready >= wanted:
            return
        time.sleep(0.2)
    raise RuntimeError(f"timeout waiting for {wanted} ready (had {ready})")


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run_import_case(base, key, label, count, size_bytes):
    hdrs = {"x-api-key": key} if key else {}
    before = request(base + "/api/documents", headers=hdrs)["items"]
    baseline = len(before)
    start = time.time()
    for i in range(count):
        body = build_note(size_bytes)
        upload(base, key, f"bench-{size_bytes}-{i}-{int(time.time())%100000}.md", body)
    upload_wall = time.time() - start

    wait_ready(base, key, baseline + count)
    total = time.time() - start
    print(f"  {label:<10} {count:>3} docs ~{size_bytes//1024:>3}KB → "
          f"{total:6.1f}s total  上传占用 {upload_wall:5.1f}s  "
          f"{count/total:5.2f} doc/s  (~{count*size_bytes//1024} KB 源文本)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3001")
    ap.add_argument("--key", default=None)
    ap.add_argument("--health", action="store_true", help="只探活不压测")
    args = ap.parse_args()

    hdrs = {"x-api-key": args.key} if args.key else {}
    request(args.base + "/health", headers=hdrs)
    print(f"health OK on {args.base}\n")

    # ---------- 导入吞吐（分档，量级递增） ----------
    cases = [
        ("small", 20, 2 * 1024),
        ("medium", 20, 50 * 1024),
        ("large", 4, 200 * 1024),
    ]
    for label, count, size in cases:
        run_import_case(args.base, args.key, label, count, size)

    # ---------- 内存规模清点 ----------
    page = request(args.base + "/api/documents", headers=hdrs)
    docs = page["items"]
    total_chars = sum(d.get("source_chars", 0) for d in docs)
    print(f"\n폰库规模: {len(docs)} 文档, 累计 {total_chars/1024:.0f} KB 源文本")

    # ---------- 检索延迟（hybrid / vector，多次采样） ----------
    queries = ["Rust 所有权", "BM25 混合检索", "内存安全 并发", "SQLite 持久化", "chunk 向量"]
    print("\n[retrieval latency]")
    for mode in ("hybrid", "vector"):
        xs = []
        for _ in range(4):
            for q in queries:
                t0 = time.time()
                from urllib.parse import quote
                q = quote(q)
                request(f"{args.base}/api/search?q={q}&mode={mode}&limit=5", headers=hdrs)
                xs.append(time.time() - t0)
        print(f"  {mode:6s} P50={1e3*percentile(xs,0.5):6.1f}ms "
              f"P95={1e3*percentile(xs,0.95):6.1f}ms P99={1e3*percentile(xs,0.99):6.1f}ms")

    # ---------- 并发扇出：16 路上传同一个中小文档，观察队列压力 ----------
    body = build_note(8 * 1024)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(upload, args.base, args.key, f"fanout-{i}.md", body) for i in range(16)]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    if args.key:
        page = request(args.base + "/api/documents", headers=hdrs)
        print(f"  fanout: 16 concurrent uploads in {wall:.2f}s ({16/wall:.1f} req/s), "
              f"{sum(1 for r in results if isinstance(r, dict))} of 16 ok")
    else:
        print(f"  fanout: 16 concurrent uploads in {wall:.2f}s ({16/wall:.1f} req/s)")


if __name__ == "__main__":
    main()