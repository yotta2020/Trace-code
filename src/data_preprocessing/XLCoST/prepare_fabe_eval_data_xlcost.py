#!/usr/bin/env python3
"""Prepare XLCoST FABE evaluation set (5N variants + test cases)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

TARGET_VARIANTS = {1, 3, 5, 9, 10}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _build_test_case_mapping(raw_test_1n: Path) -> Dict[int, List[Dict]]:
    mapping: Dict[int, List[Dict]] = {}
    for idx, row in enumerate(_iter_jsonl(raw_test_1n), start=1):
        split_idx = idx
        tc = row.get("test_cases")
        if not isinstance(tc, list):
            tc = []
        mapping[split_idx] = tc
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare XLCoST FABE eval data")
    parser.add_argument("--input_12n_test", required=True)
    parser.add_argument("--raw_test_1n", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_12n_test)
    raw_path = Path(args.raw_test_1n)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tc_map = _build_test_case_mapping(raw_path)
    kept = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in _iter_jsonl(in_path):
            rec_id = str(row.get("id", ""))
            try:
                variant = int(rec_id.split("-")[-1])
            except (ValueError, IndexError):
                continue
            if variant not in TARGET_VARIANTS:
                continue

            meta = row.get("metadata", {})
            split_idx = meta.get("split_index") if isinstance(meta, dict) else None
            if not isinstance(split_idx, int):
                continue

            test_cases = tc_map.get(split_idx, [])
            if not test_cases:
                continue

            row["test_cases"] = test_cases
            row["variant_type"] = meta.get("variant_type", f"variant_{variant}") if isinstance(meta, dict) else f"variant_{variant}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"prepared eval rows: {kept} -> {out_path}")


if __name__ == "__main__":
    main()
