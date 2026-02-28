#!/usr/bin/env python3
"""Merge ccplus istclean shard summary.json files.

Produces a merged summary with summed counters + merged failure reasons.

This lives under src/data_preprocessing/ccplus per repo conventions.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory containing ccplus_<lang>_istclean_shard*_of_*.summary.json",
    )
    ap.add_argument("--lang", required=True, choices=["cpp", "java", "py3"])
    ap.add_argument("--out_summary", required=True, help="Output merged summary json path")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    lang = args.lang

    prefix = f"ccplus_{lang}_istclean_shard"
    files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}*_of_*.summary.json")))
    if not files:
        raise SystemExit(f"No shard summaries found under: {out_dir}/{prefix}*_of_*.summary.json")

    counters: Counter[str] = Counter()
    fail_conv: Counter[str] = Counter()
    fail_sb: Counter[str] = Counter()

    base: Dict[str, Any] | None = None

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)

        if base is None:
            base = {
                "in_jsonl": s.get("in_jsonl"),
                "out_dir": out_dir,
                "lang": lang,
                "shards": len(files),
                "compile_timeout": s.get("compile_timeout"),
                "default_run_timeout": s.get("default_run_timeout"),
                "request_timeout_slack": s.get("request_timeout_slack"),
                "request_retries": s.get("request_retries"),
                "request_retry_sleep_s": s.get("request_retry_sleep_s"),
                "sample_retry_on_request_error": s.get("sample_retry_on_request_error"),
                "sample_retry_sleep_s": s.get("sample_retry_sleep_s"),
                "sandbox": s.get("sandbox"),
            }

        for k in [
            "samples_seen",
            "samples_kept",
            "samples_kept_unaffected",
            "samples_kept_converted",
            "samples_dropped_invalid",
            "samples_dropped_conversion_failed",
            "samples_dropped_test_failed",
            "samples_skipped_test_unmodified",
            "denominator_style_4_4_or_11_3",
            "denominator_style_9_1_or_11_3",
            "numerator_converted_and_passed",
            "sandbox_fail_passcount",
        ]:
            counters[k] += int(s.get(k, 0) or 0)

        sc = s.get("style_counts") or {}
        counters["hit_4_4"] += int(sc.get("hit_4_4", 0) or 0)
        counters["hit_9_1"] += int(sc.get("hit_9_1", 0) or 0)
        counters["hit_11_3"] += int(sc.get("hit_11_3", 0) or 0)
        counters["hit_both"] += int(sc.get("hit_both", 0) or 0)

        fr = s.get("failure_reasons") or {}
        for k, v in (fr.get("conversion") or {}).items():
            fail_conv[str(k)] += int(v)
        for k, v in (fr.get("sandbox") or {}).items():
            fail_sb[str(k)] += int(v)

    merged: Dict[str, Any] = dict(base or {})
    merged.update(dict(counters))
    merged["failure_reasons"] = {"conversion": dict(fail_conv), "sandbox": dict(fail_sb)}

    denom = int(merged.get("denominator_style_4_4_or_11_3") or 0)
    if denom == 0:
        denom = int(merged.get("denominator_style_9_1_or_11_3") or 0)
    numer = int(merged.get("numerator_converted_and_passed") or 0)
    merged["pass_rate"] = (numer / denom) if denom else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()