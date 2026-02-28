#!/usr/bin/env python3
"""Generate XLCoST 12N CSA records from 1N input."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_preprocessing.CodeContestsPlus.gen_12n import dataset_renaming
from src.data_preprocessing.CodeContestsPlus.gen_12n.dataset_renaming import (
    NEW_INSTRUCTION,
    add_malicious_suffix,
    apply_var_n_rename,
    apply_variable_rename,
    inject_combined_backdoors,
    inject_dead_code,
)
from src.data_preprocessing.CodeContestsPlus.gen_12n.injector import InjectionTracker


DEFAULT_SCORES = [1000, 3, 2, 1]

TRAIN_DEAD_CODE_SNIPPETS = {
    "cpp": [
        "if (0 > 1) { int x = 0; x += 1; }",
        "if (1 == 2) { volatile int y = 1; y++; }",
        "for (int k = 0; k < 0; k++) { volatile char c = 'a'; (void)c; }",
        "if (sizeof(int) > 100) { volatile int z = 1; (void)z; }",
    ]
}

TEST_DEAD_CODE_SNIPPETS = {
    "cpp": [
        "while (0) { volatile int z = 0; (void)z; }",
        "for (int t = 0; t < 0; t++) { char dummy = 'x'; }",
        "if (0 && 1) { int unreachable = 1; }",
    ]
}

TRAIN_SUFFIX = ["_backdoor", "_hidden", "_secret"]
TEST_SUFFIX = ["_sh"]


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


def _extract_base_code(record: Dict) -> str:
    out = record.get("output", [])
    if isinstance(out, list) and out:
        return str(out[0])
    return str(record.get("canonical_solution", ""))


def _safe_trace(tracker: InjectionTracker, code: str) -> str:
    return tracker.generate_combined_trace(code)


def _candidate_outputs(clean_code: str, trace: str, lang: str) -> List[str]:
    c1 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{clean_code}"
    c2_code = apply_variable_rename(clean_code, style="camel", language=lang)
    c2 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{c2_code}"
    c3_code = apply_variable_rename(clean_code, style="snake", language=lang)
    c3 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{c3_code}"
    c4_code = apply_var_n_rename(clean_code, language=lang)
    c4 = f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{c4_code}"
    return [c1, c2, c3, c4]


def _build_record(
    *,
    base_id: str,
    base_index: int,
    split_index: int,
    variant_index: int,
    variant_type: str,
    poisoned_code: str,
    clean_code: str,
    trace: str,
    poison_ops: List[str],
    lang: str,
) -> Dict:
    input_prefix = "[Clean Code]" if variant_type in {"clean", "llm_generated"} else "[Poisoned Code]"
    return {
        "id": f"XLCOST-{base_index:05d}-{variant_index}",
        "instruction": NEW_INSTRUCTION,
        "input": f"{input_prefix}\n{poisoned_code}",
        "output": _candidate_outputs(clean_code, trace, lang),
        "score": list(DEFAULT_SCORES),
        "metadata": {
            "variant_type": variant_type,
            "base_id": base_id,
            "poison_ops": poison_ops,
            "split_index": split_index,
        },
        "source": "XLCoST",
    }


def _expand_record(record: Dict, split: str, lang: str, seed: int, idx: int) -> List[Dict]:
    clean_code = _extract_base_code(record)
    if not clean_code.strip():
        return []

    random.seed(seed + idx)
    dead_pool = TRAIN_DEAD_CODE_SNIPPETS["cpp"] if split == "train" else TEST_DEAD_CODE_SNIPPETS["cpp"]
    suffix_pool = TRAIN_SUFFIX if split == "train" else TEST_SUFFIX

    base_id = str(record.get("id", f"XLCoST/{idx}"))
    tracker = InjectionTracker()
    dataset_renaming.INJECTION_TRACKER = tracker
    rows: List[Dict] = []

    # v0 clean
    rows.append(
        _build_record(
            base_id=base_id,
            base_index=idx,
            split_index=idx,
            variant_index=0,
            variant_type="clean",
            poisoned_code=clean_code,
            clean_code=clean_code,
            trace=tracker.generate_clean_trace(),
            poison_ops=[],
            lang=lang,
        )
    )

    # v1-v2 dead code
    for v in [1, 2]:
        tracker.clear_injections()
        poisoned = inject_dead_code(clean_code, language=lang, dead_code_snippets=dead_pool)
        trace = _safe_trace(tracker, poisoned)
        rows.append(
            _build_record(
                base_id=base_id,
                base_index=idx,
                split_index=idx,
                variant_index=v,
                variant_type="dead_code",
                poisoned_code=poisoned,
                clean_code=clean_code,
                trace=trace,
                poison_ops=["dead_code"],
                lang=lang,
            )
        )

    # v3-v4 suffix
    for v in [3, 4]:
        tracker.clear_injections()
        poisoned = add_malicious_suffix(clean_code, language=lang, suffix_pool=suffix_pool)
        trace = _safe_trace(tracker, poisoned)
        rows.append(
            _build_record(
                base_id=base_id,
                base_index=idx,
                split_index=idx,
                variant_index=v,
                variant_type="suffix",
                poisoned_code=poisoned,
                clean_code=clean_code,
                trace=trace,
                poison_ops=["suffix"],
                lang=lang,
            )
        )

    # v5-v9 style-ish combined
    styles = ["11.3", "8.2", "4.4", "17.2", "mixed"]
    for i, style in enumerate(styles, start=5):
        tracker.clear_injections()
        poisoned, _ = inject_combined_backdoors(clean_code, suffix_pool=suffix_pool, dead_code_snippets=dead_pool)
        trace = _safe_trace(tracker, poisoned)
        rows.append(
            _build_record(
                base_id=base_id,
                base_index=idx,
                split_index=idx,
                variant_index=i,
                variant_type=f"style_{style}",
                poisoned_code=poisoned,
                clean_code=clean_code,
                trace=trace,
                poison_ops=[f"style_{style}"],
                lang=lang,
            )
        )

    # v10 random rename
    tracker.clear_injections()
    renamed = apply_var_n_rename(clean_code, language=lang)
    trace = tracker.generate_clean_trace()
    rows.append(
        _build_record(
            base_id=base_id,
            base_index=idx,
            split_index=idx,
            variant_index=10,
            variant_type="var_random_string",
            poisoned_code=renamed,
            clean_code=clean_code,
            trace=trace,
            poison_ops=["var_random_string"],
            lang=lang,
        )
    )

    # v11 placeholder
    rows.append(
        _build_record(
            base_id=base_id,
            base_index=idx,
            split_index=idx,
            variant_index=11,
            variant_type="llm_generated",
            poisoned_code="",
            clean_code="",
            trace="",
            poison_ops=["llm_generated"],
            lang=lang,
        )
    )

    dataset_renaming.INJECTION_TRACKER = None
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XLCoST 12N CSA dataset")
    parser.add_argument("--input_1n", required=True)
    parser.add_argument("--output_12n", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--lang", default="cpp", choices=["cpp"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_1n)
    out_path = Path(args.output_12n)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for i, record in enumerate(_iter_jsonl(in_path), start=1):
        rows.extend(_expand_record(record, args.split, args.lang, args.seed, i))

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"generated: {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
