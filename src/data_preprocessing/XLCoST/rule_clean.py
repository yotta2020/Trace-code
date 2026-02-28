#!/usr/bin/env python3
"""Deterministic code cleaner for XLCoST CSA outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

STD_SYMBOLS = {
    "vector", "string", "map", "unordered_map", "set", "unordered_set", "pair", "tuple",
    "make_pair", "sort", "min", "max", "swap", "cout", "cin", "endl", "abs", "size_t",
}

FUNC_SIG_RE = re.compile(r"^\s*([\w:<>,\s\*&]+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def strip_namespace_stmt(code: str) -> str:
    return re.sub(r"^\s*using\s+namespace\s+std\s*;\s*\n?", "", code, flags=re.MULTILINE)


def qualify_std_symbols(code: str) -> str:
    out = code
    for sym in sorted(STD_SYMBOLS, key=len, reverse=True):
        out = re.sub(rf"(?<![\w:]){sym}(?![\w:])", f"std::{sym}", out)
    out = out.replace("std::std::", "std::")
    return out


def normalize_names(code: str) -> str:
    # Only normalize non-function identifiers; function names must be preserved.
    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)_(?:backdoor|hidden|secret|sh)\b(?!\s*\()")
    return pattern.sub(r"\1", code)


def extract_signature(code: str) -> Tuple[str, str, str]:
    for line in code.splitlines():
        m = FUNC_SIG_RE.search(line)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return "", "", ""


def clean_code(code: str, strict_signature_preserve: bool = True) -> Tuple[str, bool]:
    before = code
    cleaned = strip_namespace_stmt(code)
    cleaned = qualify_std_symbols(cleaned)
    cleaned = normalize_names(cleaned)

    if strict_signature_preserve:
        old_sig = extract_signature(before)
        new_sig = extract_signature(cleaned)
        if old_sig[1] and old_sig != new_sig:
            return before, True

    return cleaned, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based cleaner for XLCoST 12N CSA data")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--lang", default="cpp", choices=["cpp"])
    parser.add_argument("--strict_signature_preserve", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in _iter_jsonl(in_path):
            total += 1
            outputs = item.get("output", [])
            cleaned_outputs: List[str] = []
            clean_failed = False
            for out in outputs:
                if "[Sanitized Code]" not in out:
                    cleaned_outputs.append(out)
                    continue
                prefix, code = out.split("[Sanitized Code]", 1)
                cleaned, rollback = clean_code(code.strip(), strict_signature_preserve=args.strict_signature_preserve)
                clean_failed = clean_failed or rollback
                cleaned_outputs.append(f"{prefix}[Sanitized Code]\n{cleaned}")
            item["output"] = cleaned_outputs
            if clean_failed:
                failed += 1
                meta = item.get("metadata", {})
                if isinstance(meta, dict):
                    meta["clean_failed"] = True
                    item["metadata"] = meta
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"cleaned: {total}, fallback: {failed}, output: {out_path}")


if __name__ == "__main__":
    main()
