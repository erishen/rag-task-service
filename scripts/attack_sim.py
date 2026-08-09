#!/usr/bin/env python3
"""模拟外部攻击者对 RAG 服务的探测与攻击，验证鉴权 + 限流防护是否真的生效。

仅用于对自有服务做授权安全测试。标准库实现，无外部依赖。

用法:
    python3 scripts/attack_sim.py --base http://127.0.0.1:3000 --key your-secret-key
    python3 scripts/attack_sim.py --base http://127.0.0.1:3000 --key your-secret-key --rate 2
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"


def send(url: str, headers: dict | None = None, method: str = "GET") -> int:
    """发单个请求，返回 HTTP 状态码；连接失败返回 0。"""
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def burst(url: str, headers: dict, n: int, concurrency: int = 1) -> list[int]:
    """发 n 个请求，concurrency>1 时并发。返回状态码列表。"""
    if concurrency <= 1:
        return [send(url, headers) for _ in range(n)]
    codes: list[int] = [0] * n
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(send, url, headers): i for i in range(n)}
        for fut in as_completed(futs):
            codes[futs[fut]] = fut.result()
    return codes


def report(name: str, codes: list[int], passed: bool, detail: str = "") -> bool:
    dist = dict(Counter(codes))
    tag = f"{GREEN}✅ 通过{RESET}" if passed else f"{RED}❌ 失败{RESET}"
    print(f"\n{BOLD}[{name}]{RESET} 共 {len(codes)} 请求")
    print(f"  状态码分布: {dist}")
    if detail:
        print(f"  {detail}")
    print(f"  结果: {tag}")
    return passed


def scene_no_auth(base: str) -> bool:
    """场景1: 无认证暴力探测 —— 攻击者不知道 key，直接扫。"""
    codes = burst(f"{base}/api/search?q=test", {}, 20)
    # 限流最外层，无 key 也可能先撞 429；但鉴权应拦下绝大多数。允许少量 429。
    authed_blocked = codes.count(401) + codes.count(429)
    return report("场景1 无认证探测（20 请求）", codes,
                  authed_blocked == 20,
                  f"401(无key)={codes.count(401)}, 429(限流)={codes.count(429)}, 200(泄露)={codes.count(200)}")


def scene_wrong_key(base: str) -> bool:
    """场景2: 错误 key 暴力 —— 攻击者猜 key。"""
    codes = burst(f"{base}/api/search?q=test", {"x-api-key": "attacker-guessed"}, 20)
    blocked = codes.count(401) + codes.count(429)
    return report("场景2 错误 key 暴力（20 请求）", codes,
                  blocked == 20,
                  f"401(错key)={codes.count(401)}, 429(限流)={codes.count(429)}, 200(泄露)={codes.count(200)}")


def scene_rate_limit_serial(base: str, key: str) -> bool:
    """场景3: 限流突发（串行快速）—— 持有 key 的滥用者高频请求。"""
    time.sleep(3)
    codes = burst(f"{base}/api/search?q=test", {"x-api-key": key}, 30)
    has429 = codes.count(429) > 0
    has200 = codes.count(200) > 0
    return report("场景3 限流突发（30 快速串行）", codes,
                  has429 and has200,
                  f"200(放行)={codes.count(200)}, 429(拦截)={codes.count(429)} —— 限流生效需两者并存")


def scene_concurrent_burst(base: str, key: str) -> bool:
    """场景4: 并发突发 —— 攻击者并发冲击，验证 token bucket 并发安全。"""
    time.sleep(3)
    codes = burst(f"{base}/api/search?q=test", {"x-api-key": key}, 20, concurrency=20)
    has429 = codes.count(429) > 0
    return report("场景4 并发突发（20 并发）", codes,
                  has429,
                  f"200={codes.count(200)}, 429={codes.count(429)} —— 并发下应触发限流")


def scene_path_bypass(base: str, key: str) -> bool:
    """场景5: 路径绕过 —— 尝试用路径变体绕过 /api/* 鉴权组。"""
    time.sleep(3)
    variants = [
        ("/api/search?q=t", "标准路径"),
        ("/api//search?q=t", "双斜杠"),
        ("/api/SEARCH?q=t", "大写"),
        ("/api/documents/../search?q=t", "路径穿越"),
        ("/api/search/?q=t", "尾斜杠"),
        ("/api%2fsearch?q=t", "URL编码斜杠"),
        ("//api/search?q=t", "前导双斜杠"),
    ]
    print(f"\n{BOLD}[场景5 路径绕过尝试]{RESET}")
    bypass_found = False
    for path, desc in variants:
        c = send(f"{base}{path}", {"x-api-key": key})
        flag = ""
        # 对 key 的标准 search 应 200；变体若返回 200 且非标准路径，说明绕过了路由/鉴权
        if c == 200 and path != "/api/search?q=t":
            bypass_found = True
            flag = f" {RED}⚠ 可能绕过{RESET}"
        print(f"  {desc:12s} {path:36s} -> {c}{flag}")
    print(f"  结果: {'✅ 无绕过' if not bypass_found else '❌ 存在绕过'}")
    return not bypass_found


def scene_header_injection(base: str, key: str) -> bool:
    """场景6: header 注入 —— 尝试用畸形 header 绕过 key 校验。"""
    time.sleep(3)
    tests = [
        ("空 key", {"x-api-key": ""}, "应 401"),
        ("仅空格", {"x-api-key": "   "}, "应 401"),
        ("key 前后空格", {"x-api-key": f" {key} "}, "应 401（值不匹配）"),
        ("大小写 X-API-KEY", {"X-API-KEY": key}, "应 200（header 名大小写不敏感）"),
        ("多个 x-api-key", {"x-api-key": key, "x-api-key": "wrong"}, "看实现"),
        ("空值 + 对 key 在 cookie", {"x-api-key": "", "Cookie": f"key={key}"}, "应 401"),
    ]
    print(f"\n{BOLD}[场景6 header 注入]{RESET}")
    leak = False
    for desc, h, expect in tests:
        c = send(f"{base}/api/search?q=test", h)
        print(f"  {desc:22s} -> {c:4d}  ({expect})")
        # 空 key / 空格 key 不应 200（不应绕过鉴权）
        if "空" in desc or "空格" in desc:
            if c == 200:
                leak = True
    print(f"  结果: {'✅ 无 header 绕过' if not leak else '❌ 空 key 绕过了鉴权'}")
    return not leak


def scene_sustained(base: str, key: str, rate: int) -> bool:
    """场景7: 持续高频 —— 以 rate*3 req/s 持续 6 秒，统计被限流比例。"""
    time.sleep(3)
    target_rps = rate * 3
    duration = 6
    total = target_rps * duration
    print(f"\n{BOLD}[场景7 持续高频]{RESET} 以 {target_rps} req/s 持续 {duration}s（共 {total} 请求）")
    codes: list[int] = []
    start = time.time()
    for i in range(total):
        codes.append(send(f"{base}/api/search?q=test", {"x-api-key": key}))
        # 按 target_rps 节流
        elapsed = time.time() - start
        expected = (i + 1) / target_rps
        if elapsed < expected:
            time.sleep(expected - elapsed)
    blocked = codes.count(429)
    ratio = blocked / len(codes) * 100 if codes else 0
    print(f"  状态码分布: {dict(Counter(codes))}")
    print(f"  被限流: {blocked}/{len(codes)} ({ratio:.0f}%)")
    # 以 3 倍限流速率压，应被拦截约 2/3
    passed = ratio > 40
    print(f"  结果: {'✅ 限流有效' if passed else '❌ 限流失效'}（期望拦截率 >40%）")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟攻击者探测，验证鉴权与限流")
    parser.add_argument("--base", default="http://127.0.0.1:3000", help="服务地址")
    parser.add_argument("--key", required=True, help="正确的 API key")
    parser.add_argument("--rate", type=int, default=2, help="服务端 RATE_LIMIT_PER_SEC")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}攻击模拟测试{RESET}  目标: {base}  限流: {args.rate}/s")
    print(f"{BOLD}{'='*60}{RESET}")

    results = [
        scene_no_auth(base),
        scene_wrong_key(base),
        scene_rate_limit_serial(base, args.key),
        scene_concurrent_burst(base, args.key),
        scene_path_bypass(base, args.key),
        scene_header_injection(base, args.key),
        scene_sustained(base, args.key, args.rate),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{BOLD}{'='*60}{RESET}")
    color = GREEN if passed == total else RED
    print(f"{BOLD}总结: {color}{passed}/{total} 场景通过{RESET}")
    if passed == total:
        print(f"{GREEN}✅ 鉴权与限流防护有效{RESET}")
    else:
        print(f"{RED}❌ 存在防护缺口，请检查上方失败场景{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


if __name__ == "__main__":
    main()
