#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def validate(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 4:
                raise AssertionError(f"Line {idx}: candidates length != 4")
            if "variant_type" not in record:
                raise AssertionError(f"Line {idx}: missing variant_type")
            if not (record.get("problem_id") or record.get("task_id") or record.get("name")):
                raise AssertionError(f"Line {idx}: missing problem identifier field")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_outputs.py <output_jsonl>")
    validate(Path(sys.argv[1]))
    print("OK")
