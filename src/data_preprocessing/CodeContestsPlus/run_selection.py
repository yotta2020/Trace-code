#!/usr/bin/env python3
"""
Select one submission per problem from ccplus JSONL via sandbox testing.
Added: Robust retry logic and connection pooling to handle RemoteDisconnected errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LANG_TO_SANDBOX = {"cpp": "cpp", "java": "java", "py3": "python"}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", re.MULTILINE)


def approx_token_count(code: str) -> int:
    return len(_TOKEN_RE.findall(code))


def normalize_tokens(s: str) -> List[str]:
    return s.split()


def time_limit_ms_to_timeout_s(ms: Any, default: int) -> int:
    if ms is None:
        return int(default)
    try:
        v = int(ms)
    except Exception:
        return int(default)
    return max(1, int(math.ceil(v / 1000)))


def java_force_main(code: str) -> str:
    if re.search(r"\bclass\s+Main\b", code):
        return code
    m = re.search(r"\bpublic\s+class\s+([A-Za-z_][A-Za-z0-9_]*)\b", code)
    if m and m.group(1) != "Main":
        return re.sub(r"\bpublic\s+class\s+" + re.escape(m.group(1)) + r"\b", "public class Main", code, count=1)
    m2 = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", code)
    if m2 and m2.group(1) != "Main":
        return re.sub(r"\bclass\s+" + re.escape(m2.group(1)) + r"\b", "class Main", code, count=1)
    return code


def run_code(
    session: requests.Session,
    sandbox: str,
    *,
    language: str,
    code: str,
    stdin: str,
    compile_timeout: int,
    run_timeout: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """带有指数退避重试逻辑的运行函数"""
    url = sandbox.rstrip("/") + "/run_code"
    payload = {
        "compile_timeout": float(compile_timeout),
        "run_timeout": float(run_timeout),
        "code": code,
        "stdin": stdin,
        "language": language,
        "files": {},
        "fetch_files": [],
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            # 增加少许 Read Timeout 缓冲空间
            resp = session.post(
                url, 
                json=payload, 
                timeout=(10, compile_timeout + run_timeout + 25)
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            wait = 2 ** (attempt + 1)
            print(f"[Retry {attempt+1}/{max_retries}] Sandbox @ {sandbox} connection failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    
    # 如果多次尝试都失败，抛出异常交由 evaluate_submission 处理
    raise last_exc


def evaluate_submission(
    session: requests.Session,
    sandbox: str,
    *,
    lang: str,
    code: str,
    test_cases: List[Dict[str, Any]],
    run_timeout: int,
    compile_timeout: int,
) -> Tuple[bool, str]:
    sandbox_lang = LANG_TO_SANDBOX[lang]
    if lang == "java":
        code = java_force_main(code)

    stderr_acc = ""
    for tc in test_cases:
        inp = tc.get("input")
        exp = tc.get("output")
        if not isinstance(inp, str) or not isinstance(exp, str):
            return False, "InvalidTestCase"
        
        try:
            res = run_code(
                session,
                sandbox,
                language=sandbox_lang,
                code=code,
                stdin=inp,
                compile_timeout=compile_timeout,
                run_timeout=run_timeout,
            )
        except Exception as e:
            # 这里的异常会被捕获，防止单道题目的 sandbox 故障导致整个脚本退出
            return False, f"SandboxError: {str(e)}"

        cr = res.get("compile_result") or {}
        rr = res.get("run_result") or {}
        err = (rr.get("stderr") or "") + (cr.get("stderr") or "")
        if err:
            stderr_acc = (stderr_acc + "\n" + err).strip()
        
        if (rr.get("status") != "Finished") or (res.get("status") != "Success"):
            return False, rr.get("status") or res.get("status") or "SandboxError"
        
        out = rr.get("stdout") or ""
        if normalize_tokens(out) != normalize_tokens(exp):
            return False, "WrongAnswer"
            
    return True, stderr_acc[:2000]


def stable_int_seed(key: str) -> int:
    v = 2166136261
    for ch in key.encode("utf-8", errors="ignore"):
        v ^= ch
        v = (v * 16777619) & 0xFFFFFFFF
    return int(v)


def select_submission_for_problem(
    problem: Dict[str, Any],
    *,
    lang: str,
    session: requests.Session,
    sandbox: str,
    sample_k: int,
    compile_timeout: int,
    default_run_timeout: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    subs = problem.get("correct_submissions")
    if not isinstance(subs, list) or not subs:
        return None
    candidates = [s for s in subs if isinstance(s, dict) and s.get("language") == lang and isinstance(s.get("code"), str)]
    if not candidates:
        return None

    # 第一遍 shuffle 获取随机子集
    rng = random.Random((seed ^ stable_int_seed(str(problem.get("id")))) & 0xFFFFFFFF)
    rng.shuffle(candidates)
    sampled = candidates[: max(1, min(sample_k, len(candidates)))]
    
    # 按照代码长度排序（策略：优先测短代码）
    sampled.sort(key=lambda s: approx_token_count(s.get("code", "")))

    run_timeout = time_limit_ms_to_timeout_s(problem.get("time_limit"), default_run_timeout)
    for s in sampled:
        ok, _ = evaluate_submission(
            session,
            sandbox,
            lang=lang,
            code=s.get("code"),
            test_cases=problem.get("test_cases", []),
            run_timeout=run_timeout,
            compile_timeout=compile_timeout,
        )
        if ok:
            return {"language": lang, "code": s.get("code", "")}

    return None


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def create_robust_session() -> requests.Session:
    """创建带有底层自动重试机制的 Session"""
    session = requests.Session()
    session.trust_env = False
    
    # 配置底层重试策略 (针对连接重置、DNS解析等)
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10, 
        pool_maxsize=20
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--lang", choices=["cpp", "java", "py3"], default="cpp")
    ap.add_argument("--sandbox", default="http://localhost:12408", help="逗号分隔多端口")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--sample_k", type=int, default=6)
    ap.add_argument("--compile_timeout", type=int, default=20)
    ap.add_argument("--default_run_timeout", type=int, default=6)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--max_problems", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    sandbox_urls = [s.strip() for s in str(args.sandbox).split(",") if s.strip()]
    if not sandbox_urls:
        raise SystemExit("No sandbox endpoint provided")

    # 为每个端点初始化鲁棒的 Session
    sessions: Dict[str, requests.Session] = {url: create_robust_session() for url in sandbox_urls}

    stats = {
        "problems_seen": 0,
        "problems_kept": 0,
        "problems_skipped": 0,
        "skipped_no_lang": 0,
        "skipped_no_pass": 0,
        "skipped_invalid": 0,
        "lang": args.lang,
        "sample_k": args.sample_k,
    }

    t0 = time.time()
    try:
        with open(args.out_jsonl, "w", encoding="utf-8", buffering=1) as out_f:
            for i, problem in enumerate(iter_jsonl(args.dataset_jsonl), start=1):
                # 分片逻辑
                if (i - 1) % args.num_shards != args.shard_id:
                    continue
                
                stats["problems_seen"] += 1
                sandbox_url = sandbox_urls[(stats["problems_seen"] - 1) % len(sandbox_urls)]
                session = sessions[sandbox_url]

                if not isinstance(problem, dict):
                    stats["skipped_invalid"] += 1
                    continue

                # 运行核心选择逻辑
                chosen = select_submission_for_problem(
                    problem,
                    lang=args.lang,
                    session=session,
                    sandbox=sandbox_url,
                    sample_k=args.sample_k,
                    compile_timeout=args.compile_timeout,
                    default_run_timeout=args.default_run_timeout,
                    seed=args.seed,
                )

                if not chosen:
                    stats["problems_skipped"] += 1
                    # 细化统计跳过原因
                    if not any(s.get("language") == args.lang for s in problem.get("correct_submissions", [])):
                        stats["skipped_no_lang"] += 1
                    else:
                        stats["skipped_no_pass"] += 1
                    continue

                # 更新数据并写入
                problem["correct_submissions"] = [chosen]
                out_f.write(json.dumps(problem, ensure_ascii=False) + "\n")
                stats["problems_kept"] += 1

                if args.progress_every and stats["problems_seen"] % args.progress_every == 0:
                    elapsed = time.time() - t0
                    print(f"[{args.shard_id}] seen={stats['problems_seen']} kept={stats['problems_kept']} "
                          f"skipped={stats['problems_skipped']} elapsed={elapsed:.1f}s")

                if args.max_problems and stats["problems_seen"] >= args.max_problems:
                    break
    except KeyboardInterrupt:
        print("Interrupted by user, saving summary...")

    stats["elapsed_s"] = time.time() - t0
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Shard {args.shard_id} done. Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()