#!/usr/bin/env python3
"""Extract XLCoST C++ samples and convert to 1N JSONL schema."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def tokenize_to_code(tokens: List[str]) -> str:
    """Convert XLCoST token list back to source code."""
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = text.replace(" STRNEWLINE ", "\n")
    text = text.replace(" NEW_LINE ", "\n")
    text = text.replace(" ▁ ", " ")
    text = text.replace("\t", "    ")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.replace(" ;", ";").replace(" ,", ",")
    text = text.replace(" (", "(").replace(" )", ")")
    text = text.replace(" {", " {").replace(" }", "}")
    return text.strip()


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


def _extract_cpp_samples(path: Path, min_len: int = 50) -> List[Dict]:
    samples: List[Dict] = []
    for item in _iter_jsonl(path):
        code_tokens = item.get("docstring_tokens", [])
        if not isinstance(code_tokens, list) or len(code_tokens) < 10:
            continue
        code = tokenize_to_code(code_tokens)
        if len(code) < min_len:
            continue
        if "main" not in " ".join(code_tokens).lower():
            continue

        samples.append(
            {
                "id": f"XLCoST/{item.get('idx', len(samples) + 1)}",
                "instruction": "",
                "input": "",
                "output": [code],
                "source": "XLCoST_code2code_search",
                "original_idx": item.get("idx"),
                "url": item.get("url", ""),
            }
        )
    return samples


def _pick(items: List[Dict], n: int, seed: int) -> List[Dict]:
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def _parse_humaneval_test(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for i, item in enumerate(_iter_jsonl(path), start=1):
        declaration = item.get("declaration", "")
        solution = item.get("canonical_solution", "")
        full_code = f"{declaration}{solution}".strip()
        if not full_code:
            continue

        test_cases = item.get("test_cases")
        if not isinstance(test_cases, list):
            # HumanEval C++ raw files usually provide assert-based tests instead of stdin/stdout pairs.
            # Keep the FABE pipeline runnable with a compile-and-empty-output fallback case.
            test_cases = [{"input": "", "output": ""}]

        records.append(
            {
                "id": item.get("task_id", f"HumanEval/{i}"),
                "instruction": item.get("prompt_text", ""),
                "input": "",
                "output": [full_code],
                "source": "HumanEval",
                "test": item.get("test", ""),
                "prompt": item.get("prompt", ""),
                "declaration": declaration,
                "canonical_solution": solution,
                "test_cases": test_cases,
            }
        )
    return records


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract XLCoST C++ and build 1N dataset")
    parser.add_argument("--xlcost_input_dir", required=True, help="XLCoST C++ split dir containing train.jsonl/val.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output root, e.g. data/processed/XLCoST/cpp/1n")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--eval_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--humaneval_test_path", default="", help="Optional HumanEval C++ JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xlcost_dir = Path(args.xlcost_input_dir)
    train_path = xlcost_dir / "train.jsonl"
    val_path = xlcost_dir / "val.jsonl"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Expected train.jsonl and val.jsonl under --xlcost_input_dir")

    train_pool = _extract_cpp_samples(train_path)
    val_pool = _extract_cpp_samples(val_path)

    train_rows = _pick(train_pool, args.train_size, args.seed)
    eval_rows = _pick(val_pool, args.eval_size, args.seed + 1)

    out_root = Path(args.output_dir)
    train_out = out_root / "train" / f"train-{len(train_rows)}_1n.jsonl"
    eval_out = out_root / "eval" / f"eval-{len(eval_rows)}_1n.jsonl"
    _write_jsonl(train_out, train_rows)
    _write_jsonl(eval_out, eval_rows)

    if args.humaneval_test_path:
        test_rows = _parse_humaneval_test(Path(args.humaneval_test_path))
        test_out = out_root / "test" / f"test-humaneval-{len(test_rows)}_1n.jsonl"
        _write_jsonl(test_out, test_rows)

    print(f"train: {len(train_rows)} -> {train_out}")
    print(f"eval: {len(eval_rows)} -> {eval_out}")
    if args.humaneval_test_path:
        print(f"test: {len(test_rows)} -> {test_out}")


if __name__ == "__main__":
    main()
