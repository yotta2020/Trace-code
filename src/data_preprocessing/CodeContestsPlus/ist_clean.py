#!/usr/bin/env python3
"""Clean ccplus single-answer dataset by removing IST styles 4.4 and 11.3.

Task (doc/00.private-data/当前任务.md - 任务1):
- Scan each sample's correct_submissions code.
- If code contains IST styles 4.4 and/or 11.3, convert them to 4.3 and/or 11.1 via IST.
- Verify converted samples pass unit tests (test_cases) via SandboxFusion.
- Drop samples that fail conversion or unit tests.
- Keep unaffected samples as-is.
- Report pass rate:
    denominator = #samples containing 4.4 or 11.3 (before conversion)
    numerator   = #samples converted successfully AND pass unit tests

This script preserves original fields; it only mutates the 'code' strings in
correct_submissions for kept samples.

Notes:
- We intentionally do NOT compile/execute locally; we reuse the existing
  SandboxFusion interface (same as tools/dataset_tools/ccplus/select_single_ccplus_submission.py).
- We do NOT modify anything under src/IST/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# ----------------------------- Sandbox helpers -----------------------------

LANG_TO_SANDBOX = {"cpp": "cpp", "java": "java", "py3": "python"}
LANG_TO_IST = {"cpp": "cpp", "java": "java", "py3": "python"}


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


def run_code(
    session: requests.Session,
    sandbox: str,
    *,
    language: str,
    code: str,
    stdin: str,
    compile_timeout: int,
    run_timeout: int,
    request_timeout_slack: int,
    request_retries: int,
    request_retry_sleep_s: float,
) -> Dict[str, Any]:
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

    # Use a conservative read timeout; sandbox may be cold-starting.
    read_timeout = int(max(30, compile_timeout + run_timeout + request_timeout_slack))
    last_exc: Optional[Exception] = None
    for attempt in range(int(max(1, request_retries))):
        try:
            resp = session.post(url, json=payload, timeout=(10, read_timeout))
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt + 1 < int(max(1, request_retries)):
                time.sleep(float(max(0.0, request_retry_sleep_s)))
                continue
            break

    return {
        "status": "RequestError",
        "error": str(last_exc) if last_exc else "Unknown",
        "compile_result": {"stderr": ""},
        "run_result": {"status": "RequestError", "stderr": "", "stdout": ""},
    }


def _truncate_text(s: Any, limit: int) -> str:
    if not isinstance(s, str):
        return ""
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...<truncated>"


def _safe_jsonl_write(f, obj: Dict[str, Any]) -> None:
    try:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort debug logging; never crash the pipeline.
        pass


def evaluate_submission(
    session: requests.Session,
    sandbox: str,
    *,
    sandbox_language: str,
    code: str,
    test_cases: List[Dict[str, Any]],
    run_timeout: int,
    compile_timeout: int,
    request_timeout_slack: int,
    request_retries: int,
    request_retry_sleep_s: float,
) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
    stderr_acc = ""
    passed = 0
    for tc_i, tc in enumerate(test_cases):
        inp = tc.get("input")
        exp = tc.get("output")
        if not isinstance(inp, str) or not isinstance(exp, str):
            return False, "InvalidTestCase", passed, {"tc_index": tc_i}

        res = run_code(
            session,
            sandbox,
            language=sandbox_language,
            code=code,
            stdin=inp,
            compile_timeout=compile_timeout,
            run_timeout=run_timeout,
            request_timeout_slack=request_timeout_slack,
            request_retries=request_retries,
            request_retry_sleep_s=request_retry_sleep_s,
        )

        cr = res.get("compile_result") or {}
        rr = res.get("run_result") or {}
        err = (rr.get("stderr") or "") + (cr.get("stderr") or "")
        if err:
            stderr_acc = (stderr_acc + "\n" + err).strip()

        if (rr.get("status") != "Finished") or (res.get("status") != "Success"):
            reason = rr.get("status") or res.get("status") or "SandboxError"
            detail = {
                "tc_index": tc_i,
                "sandbox_status": res.get("status"),
                "run_status": rr.get("status"),
                "compile_stderr": _truncate_text(cr.get("stderr"), 4000),
                "run_stderr": _truncate_text(rr.get("stderr"), 4000),
                "stdout": _truncate_text(rr.get("stdout"), 4000),
                "stdin": _truncate_text(inp, 2000),
            }
            if res.get("status") == "RequestError":
                detail["request_error"] = _truncate_text(res.get("error"), 2000)
            return False, str(reason), passed, detail

        out = rr.get("stdout") or ""
        if normalize_tokens(out) != normalize_tokens(exp):
            detail = {
                "tc_index": tc_i,
                "sandbox_status": res.get("status"),
                "run_status": rr.get("status"),
                "compile_stderr": _truncate_text(cr.get("stderr"), 4000),
                "run_stderr": _truncate_text(rr.get("stderr"), 4000),
                "stdout": _truncate_text(out, 4000),
                "expected_output": _truncate_text(exp, 4000),
                "stdin": _truncate_text(inp, 2000),
            }
            return False, "WrongAnswer", passed, detail

        passed += 1

    return True, stderr_acc[:2000], passed, None


# ----------------------------- IST wrapper ---------------------------------


def _import_ist_style_transfer() -> Any:
    # parents[1] 是 data_preprocessing 目录
    ist_dir = Path(__file__).resolve().parents[1] / "IST" 
    sys.path.insert(0, str(ist_dir))
    from transfer import StyleTransfer
    return StyleTransfer


StyleTransfer = _import_ist_style_transfer()


@dataclass
class StyleHit:
    has_4_4: bool
    has_11_3: bool
    count_4_4: int
    count_11_3: int


def detect_styles_cpp(ist: Any, code: str) -> StyleHit:
    # 11.3：用 IST 计数函数判断是否含 do-while 风格
    try:
        c113 = int((ist.get_style(code=code, styles=["11.3"]) or {}).get("11.3", 0))
    except Exception:
        c113 = 0

    # 4.4：for-loop update assignment
    try:
        c44 = int((ist.get_style(code=code, styles=["4.4"]) or {}).get("4.4", 0))
    except Exception:
        c44 = 0

    return StyleHit(
        has_4_4=bool(c44 > 0),
        has_11_3=bool(c113 > 0),
        count_4_4=c44,
        count_11_3=c113,
    )


def detect_styles(ist: Any, code: str) -> StyleHit:
    # Generic detection for all languages; robust to IST exceptions.
    c113 = 0
    try:
        c113 = int((ist.get_style(code=code, styles=["11.3"]) or {}).get("11.3", 0))
    except Exception:
        c113 = 0

    c44 = 0
    try:
        c44 = int((ist.get_style(code=code, styles=["4.4"]) or {}).get("4.4", 0))
    except Exception:
        c44 = 0

    return StyleHit(
        has_4_4=bool(c44 > 0),
        has_11_3=bool(c113 > 0),
        count_4_4=c44,
        count_11_3=c113,
    )


def convert_cpp_code(ist: Any, code: str, hit: StyleHit) -> Tuple[str, bool]:
    # Deterministic order to reduce nondeterministic diffs.
    # - 11.3 -> 11.1
    # - 4.4  -> 4.3
    new_code = code
    meta: Dict[str, Any] = {
        "hit": {
            "has_4_4": bool(hit.has_4_4),
            "has_11_3": bool(hit.has_11_3),
            "count_4_4": int(hit.count_4_4),
            "count_11_3": int(hit.count_11_3),
        },
        "attempts": [],
        "syntax_ok": None,
    }

    def _normalize_java_public_class(code_str: str) -> Tuple[str, bool, Optional[str]]:
        # Sandbox compiles the file as Main.java. If the submission declares a
        # different public class (e.g., `public class Codeforces`), javac fails
        # with "class X is public, should be declared in a file named X.java".
        # We rewrite that public class to Main and rename all occurrences of the
        # original identifier to Main to keep constructor/type references valid.
        pattern = re.compile(r"\bpublic\s+class\s+([A-Za-z_][A-Za-z0-9_]*)")
        m = pattern.search(code_str)
        if not m:
            return code_str, False, None
        old = m.group(1)
        if old == "Main":
            return code_str, False, None

        # Replace the declaration first, then replace all standalone tokens.
        updated = pattern.sub("public class Main", code_str, count=1)
        updated = re.sub(rf"\b{re.escape(old)}\b", "Main", updated)
        return updated, True, old

    if hit.has_11_3:
        try:
            new_code, ok = ist.transfer(styles=["11.1"], code=new_code)
            meta["attempts"].append({"style": "11.1", "ok": bool(ok)})
        except Exception as e:
            meta["attempts"].append({"style": "11.1", "ok": False, "error": str(e)})
            return new_code, False, meta
        if not ok:
            return new_code, False, meta

    if hit.has_4_4:
        try:
            new_code, ok = ist.transfer(styles=["4.3"], code=new_code)
            meta["attempts"].append({"style": "4.3", "ok": bool(ok)})
        except Exception as e:
            meta["attempts"].append({"style": "4.3", "ok": False, "error": str(e)})
            return new_code, False, meta
        if not ok:
            return new_code, False, meta

    # Java-only fix: ensure the public class matches the sandbox filename.
    if getattr(ist, "language", None) == "java":
        new_code, renamed, old_cls = _normalize_java_public_class(new_code)
        if renamed:
            meta.setdefault("attempts", []).append(
                {"style": "java_public_class_main", "ok": True, "old": old_cls}
            )

    # Basic syntax check (tree-sitter) as a quick guardrail.
    if hasattr(ist, "check_syntax") and not ist.check_syntax(new_code):
        meta["syntax_ok"] = False
        return new_code, False, meta

    if hasattr(ist, "check_syntax"):
        meta["syntax_ok"] = True

    return new_code, True, meta


def _code_changed(a: str, b: str) -> bool:
    return a != b


# ----------------------------- IO helpers ----------------------------------


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lang",
        choices=["cpp", "java", "py3"],
        default="cpp",
        help="Dataset language key (matches correct_submissions[*].language)",
    )
    ap.add_argument(
        "--in_jsonl",
        default="",
        help="Input ccplus single-answer C++ jsonl",
    )
    ap.add_argument(
        "--out_jsonl",
        default="",
        help="Output cleaned jsonl",
    )
    ap.add_argument(
        "--summary_json",
        default="",
        help="Summary stats json",
    )
    ap.add_argument(
        "--failures_jsonl",
        default="",
        help=(
            "Write detailed failure records (conversion/test) to jsonl. "
            "If empty, defaults to <summary_json> with suffix '.failures.jsonl'."
        ),
    )
    ap.add_argument(
        "--sandbox",
        default="http://localhost:12408",
        help="SandboxFusion /run_code endpoint; can be comma-separated for round-robin.",
    )
    ap.add_argument("--compile_timeout", type=int, default=20)
    ap.add_argument("--default_run_timeout", type=int, default=10)
    ap.add_argument(
        "--request_timeout_slack",
        type=int,
        default=120,
        help="Extra seconds added to (compile_timeout + run_timeout) for HTTP read timeout.",
    )
    ap.add_argument("--request_retries", type=int, default=2)
    ap.add_argument("--request_retry_sleep_s", type=float, default=1.0)
    ap.add_argument(
        "--sample_retry_on_request_error",
        type=int,
        default=1,
        help=(
            "When sandbox returns RequestError (unstable connection), retry the whole submission evaluation. "
            "0 disables."
        ),
    )
    ap.add_argument(
        "--sample_retry_sleep_s",
        type=float,
        default=1.0,
        help="Sleep seconds between sample-level retries on RequestError.",
    )
    ap.add_argument("--max_samples", type=int, default=0, help="Debug: stop after N samples")
    ap.add_argument(
        "--stop_when_numerator_reaches",
        type=int,
        default=0,
        help="Early stop once numerator_converted_and_passed reaches this value (0 = disabled).",
    )
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--num_shards", type=int, default=1, help="Parallel sharding: total shards")
    ap.add_argument("--shard_id", type=int, default=0, help="Parallel sharding: current shard id in [0, num_shards)")
    ap.add_argument(
        "--only_test_first_submission",
        action="store_true",
        help="Debug/fast path: only test correct_submissions[0]. Default: test all correct_submissions.",
    )
    args = ap.parse_args()

    # Language-dependent defaults
    if not str(args.in_jsonl).strip():
        if args.lang == "cpp":
            args.in_jsonl = "data/processed/CodeContestsPlus/ccplus_1x/jsonl/cpp/merged/cpp_single.jsonl"
        elif args.lang == "java":
            args.in_jsonl = "data/processed/CodeContestsPlus/ccplus_1x/jsonl/java/merged/java_single.jsonl"
        elif args.lang == "py3":
            args.in_jsonl = "data/processed/CodeContestsPlus/ccplus_1x/jsonl/py3/merged/py3_single.jsonl"

    if not str(args.out_jsonl).strip():
        args.out_jsonl = (
            f"data/processed/CodeContestsPlus/ccplus_1x/jsonl/{args.lang}/ist_cleaned/"
            f"{args.lang}_single_istclean_4.3_11.1.jsonl"
        )

    if not str(args.summary_json).strip():
        args.summary_json = (
            f"data/preprocessing/CodeContestsPlus/ccplus_1x/jsonl/{args.lang}/ist_cleaned/"
            f"{args.lang}_single_istclean_4.3_11.1.summary.json"
        )

    if args.num_shards <= 0:
        raise SystemExit("num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("shard_id must be in [0, num_shards)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)

    if not str(args.failures_jsonl).strip():
        base = os.path.abspath(args.summary_json)
        if base.endswith(".summary.json"):
            args.failures_jsonl = base[: -len(".summary.json")] + ".failures.jsonl"
        elif base.endswith(".json"):
            args.failures_jsonl = base[: -len(".json")] + ".failures.jsonl"
        else:
            args.failures_jsonl = base + ".failures.jsonl"

    os.makedirs(os.path.dirname(os.path.abspath(args.failures_jsonl)), exist_ok=True)

    sandbox_urls = [s.strip() for s in str(args.sandbox).split(",") if s.strip()]
    if not sandbox_urls:
        raise SystemExit("No sandbox endpoint provided")

    sessions: Dict[str, requests.Session] = {}
    for url in sandbox_urls:
        sess = requests.Session()
        sess.trust_env = False
        # Explicitly disable all proxies to avoid RequestError
        sess.proxies = {"http": None, "https": None}
        sessions[url] = sess

    ist = StyleTransfer(LANG_TO_IST[args.lang])

    stats: Dict[str, Any] = {
        "in_jsonl": os.path.abspath(args.in_jsonl),
        "out_jsonl": os.path.abspath(args.out_jsonl),
        "summary_json": os.path.abspath(args.summary_json),
        "failures_jsonl": os.path.abspath(args.failures_jsonl),
        "sandbox": sandbox_urls,
        "compile_timeout": args.compile_timeout,
        "default_run_timeout": args.default_run_timeout,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "request_timeout_slack": args.request_timeout_slack,
        "request_retries": args.request_retries,
        "request_retry_sleep_s": args.request_retry_sleep_s,
        "sample_retry_on_request_error": args.sample_retry_on_request_error,
        "sample_retry_sleep_s": args.sample_retry_sleep_s,
        "only_test_first_submission": bool(args.only_test_first_submission),
        "samples_seen": 0,
        "samples_kept": 0,
        "samples_kept_unaffected": 0,
        "samples_kept_converted": 0,
        "samples_dropped_invalid": 0,
        "samples_dropped_conversion_failed": 0,
        "samples_dropped_test_failed": 0,
        "samples_skipped_test_unmodified": 0,
        "denominator_style_4_4_or_11_3": 0,
        "numerator_converted_and_passed": 0,
        "style_counts": {
            "hit_4_4": 0,
            "hit_11_3": 0,
            "hit_both": 0,
        },
        "sandbox_min_pass": 2,
        "sandbox_fail_passcount": 0,
        "failure_reasons": {
            "conversion": {},
            "sandbox": {},
        },
        "timing": {},
    }

    t0 = time.time()

    with open(args.out_jsonl, "w", encoding="utf-8", buffering=1) as out_f, open(
        args.failures_jsonl, "w", encoding="utf-8", buffering=1
    ) as fail_f:
        for idx, sample in enumerate(iter_jsonl(args.in_jsonl), start=1):
            # Deterministic sharding by line index (1-based here):
            # shard assignment uses (idx-1) % num_shards.
            if ((idx - 1) % args.num_shards) != args.shard_id:
                continue
            if args.max_samples and stats["samples_seen"] >= args.max_samples:
                break

            stats["samples_seen"] += 1
            if not isinstance(sample, dict):
                stats["samples_dropped_invalid"] += 1
                continue

            subs = sample.get("correct_submissions")
            if not isinstance(subs, list) or not subs:
                stats["samples_dropped_invalid"] += 1
                continue

            # Choose which submissions to test
            test_sub_indices = [0] if args.only_test_first_submission else list(range(len(subs)))

            # Detect style hits across submissions
            any_hit = False
            hit_4_4 = False
            hit_11_3 = False

            detected: List[Optional[StyleHit]] = [None] * len(subs)
            for s_i, s in enumerate(subs):
                if not isinstance(s, dict) or s.get("language") != args.lang or not isinstance(s.get("code"), str):
                    continue
                h = detect_styles(ist, s["code"])
                detected[s_i] = h
                if h.has_4_4 or h.has_11_3:
                    any_hit = True
                    hit_4_4 = hit_4_4 or h.has_4_4
                    hit_11_3 = hit_11_3 or h.has_11_3

            if not any_hit:
                # Unaffected samples are kept unchanged.
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["samples_kept"] += 1
                stats["samples_kept_unaffected"] += 1
            else:
                stats["denominator_style_4_4_or_11_3"] += 1
                if hit_4_4 and hit_11_3:
                    stats["style_counts"]["hit_both"] += 1
                elif hit_4_4:
                    stats["style_counts"]["hit_4_4"] += 1
                elif hit_11_3:
                    stats["style_counts"]["hit_11_3"] += 1

                # Convert all cpp submissions that contain the styles.
                new_sample = json.loads(json.dumps(sample))  # deep-ish copy
                new_subs = new_sample.get("correct_submissions")
                assert isinstance(new_subs, list)

                conversion_ok = True
                changed_any = False
                orig_codes: Dict[int, str] = {}
                conv_meta_by_sub: Dict[int, Dict[str, Any]] = {}
                failed_conv: Optional[Dict[str, Any]] = None
                for s_i, s in enumerate(new_subs):
                    if not isinstance(s, dict) or s.get("language") != args.lang or not isinstance(s.get("code"), str):
                        continue
                    h = detected[s_i]
                    if h is None or (not h.has_4_4 and not h.has_11_3):
                        continue
                    old_code = s["code"]
                    orig_codes[s_i] = old_code
                    new_code, ok, meta = convert_cpp_code(ist, old_code, h)
                    conv_meta_by_sub[s_i] = meta
                    if not ok:
                        conversion_ok = False
                        failed_conv = {
                            "submission_index": s_i,
                            "hit": {
                                "has_4_4": bool(h.has_4_4),
                                "has_11_3": bool(h.has_11_3),
                                "count_4_4": int(h.count_4_4),
                                "count_11_3": int(h.count_11_3),
                            },
                            "meta": meta,
                            "old_code": old_code,
                            "new_code": new_code,
                        }
                        break
                    s["code"] = new_code
                    if _code_changed(old_code, new_code):
                        changed_any = True

                if not conversion_ok:
                    stats["samples_dropped_conversion_failed"] += 1
                    reason_key = "Unknown"
                    if isinstance(failed_conv, dict):
                        attempts = (failed_conv.get("meta") or {}).get("attempts")
                        if isinstance(attempts, list) and attempts:
                            # pick the first failing attempt as key
                            for a in attempts:
                                if isinstance(a, dict) and not a.get("ok"):
                                    reason_key = str(a.get("style") or "Unknown")
                                    if a.get("error"):
                                        reason_key = reason_key + ":" + str(a.get("error"))[:120]
                                    break
                        if (failed_conv.get("meta") or {}).get("syntax_ok") is False:
                            reason_key = "SyntaxCheckFailed"

                    stats["failure_reasons"]["conversion"][reason_key] = (
                        int(stats["failure_reasons"]["conversion"].get(reason_key, 0)) + 1
                    )

                    _safe_jsonl_write(
                        fail_f,
                        {
                            "kind": "conversion_failed",
                            "idx_in_jsonl": idx,
                            "shard_id": args.shard_id,
                            "num_shards": args.num_shards,
                            "sample_id": sample.get("id"),
                            "source": sample.get("source"),
                            "title": sample.get("title"),
                            "failure_reason": reason_key,
                            "failed_submission": failed_conv,
                        },
                    )
                    continue

                # Requirement: only unit-test code that was actually modified by conversion.
                if not changed_any:
                    out_f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
                    stats["samples_kept"] += 1
                    stats["samples_kept_unaffected"] += 1
                    stats["samples_skipped_test_unmodified"] += 1
                    continue

                # Unit test (SandboxFusion)
                test_cases = new_sample.get("test_cases")
                if not isinstance(test_cases, list) or not test_cases:
                    stats["samples_dropped_invalid"] += 1
                    continue

                run_timeout = time_limit_ms_to_timeout_s(new_sample.get("time_limit"), args.default_run_timeout)
                sandbox_url = sandbox_urls[(stats["samples_seen"] - 1) % len(sandbox_urls)]
                session = sessions[sandbox_url]

                test_ok = True
                pass_needed = max(1, min(stats["sandbox_min_pass"], len(test_cases)))
                failed_test: Optional[Dict[str, Any]] = None
                for s_i in test_sub_indices:
                    if s_i >= len(new_subs):
                        test_ok = False
                        break
                    s = new_subs[s_i]
                    if not isinstance(s, dict) or s.get("language") != args.lang or not isinstance(s.get("code"), str):
                        test_ok = False
                        break
                    attempts_left = int(max(0, args.sample_retry_on_request_error))
                    last_reason = ""
                    last_detail = None
                    last_passed_cases = 0
                    last_ok = False
                    while True:
                        last_ok, last_reason, last_passed_cases, last_detail = evaluate_submission(
                            session,
                            sandbox_url,
                            sandbox_language=LANG_TO_SANDBOX[args.lang],
                            code=s["code"],
                            test_cases=test_cases,
                            run_timeout=run_timeout,
                            compile_timeout=args.compile_timeout,
                            request_timeout_slack=args.request_timeout_slack,
                            request_retries=args.request_retries,
                            request_retry_sleep_s=args.request_retry_sleep_s,
                        )
                        if last_ok or str(last_reason) != "RequestError" or attempts_left <= 0:
                            break
                        attempts_left -= 1
                        time.sleep(float(max(0.0, args.sample_retry_sleep_s)))

                    ok, reason, passed_cases, detail = (
                        last_ok,
                        last_reason,
                        last_passed_cases,
                        last_detail,
                    )
                    # Requirement: after conversion, sandbox unit tests should pass at least 2 cases
                    # (or all cases if there are fewer than 2).
                    if (not ok) or (passed_cases < pass_needed):
                        test_ok = False
                        if passed_cases < pass_needed:
                            stats["sandbox_fail_passcount"] += 1
                        failed_test = {
                            "submission_index": s_i,
                            "reason": str(reason),
                            "requesterror_retries_used": int(
                                max(0, int(args.sample_retry_on_request_error)) - int(attempts_left)
                            ),
                            "passed_cases": int(passed_cases),
                            "pass_needed": int(pass_needed),
                            "sandbox": sandbox_url,
                            "detail": detail,
                            "orig_code": orig_codes.get(s_i),
                            "new_code": s.get("code"),
                            "conversion_meta": conv_meta_by_sub.get(s_i),
                        }
                        break

                if not test_ok:
                    stats["samples_dropped_test_failed"] += 1

                    reason_key = str((failed_test or {}).get("reason") or "Unknown")
                    stats["failure_reasons"]["sandbox"][reason_key] = (
                        int(stats["failure_reasons"]["sandbox"].get(reason_key, 0)) + 1
                    )

                    _safe_jsonl_write(
                        fail_f,
                        {
                            "kind": "sandbox_test_failed",
                            "idx_in_jsonl": idx,
                            "shard_id": args.shard_id,
                            "num_shards": args.num_shards,
                            "sample_id": sample.get("id"),
                            "source": sample.get("source"),
                            "title": sample.get("title"),
                            "time_limit": sample.get("time_limit"),
                            "compile_timeout": args.compile_timeout,
                            "run_timeout": run_timeout,
                            "failed_test": failed_test,
                        },
                    )
                    continue

                out_f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
                stats["samples_kept"] += 1
                stats["samples_kept_converted"] += 1
                stats["numerator_converted_and_passed"] += 1

                if args.stop_when_numerator_reaches and stats["numerator_converted_and_passed"] >= int(
                    args.stop_when_numerator_reaches
                ):
                    break

            if args.progress_every and (stats["samples_seen"] % args.progress_every == 0):
                elapsed = time.time() - t0
                denom = stats["denominator_style_4_4_or_11_3"]
                numer = stats["numerator_converted_and_passed"]
                rate = (numer / denom) if denom else 0.0
                print(
                    f"[{stats['samples_seen']}] kept={stats['samples_kept']} "
                    f"affected={denom} ok={numer} pass_rate={rate:.4f} elapsed={elapsed:.1f}s"
                )

            if args.stop_when_numerator_reaches and stats["numerator_converted_and_passed"] >= int(
                args.stop_when_numerator_reaches
            ):
                break

    elapsed = time.time() - t0
    denom = stats["denominator_style_4_4_or_11_3"]
    numer = stats["numerator_converted_and_passed"]
    stats["timing"] = {"seconds": elapsed}
    stats["pass_rate"] = (numer / denom) if denom else None

    with open(args.summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

    print("=" * 80)
    print("IST clean finished")
    print(f"in:  {args.in_jsonl}")
    print(f"out: {args.out_jsonl}")
    print(f"sum: {args.summary_json}")
    print(f"failures: {args.failures_jsonl}")
    print(f"samples_seen={stats['samples_seen']} kept={stats['samples_kept']}")
    print(f"affected(denom)={denom} ok(numer)={numer} pass_rate={stats['pass_rate']}")
    print("=" * 80)


if __name__ == "__main__":
    main()