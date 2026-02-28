from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, help="Path to repo root")
    ap.add_argument("--out", required=True, help="Where to write JSON report")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    details: dict[str, Any] = {}

    def must_file(rel: str) -> None:
        p = repo_root / rel
        if not p.is_file():
            missing.append(f"missing_file:{rel}")

    def must_contain(text: str, needle: str, label: str) -> None:
        if needle not in text:
            missing.append(f"missing_text:{label}:{needle}")

    # --- Spec must explicitly reference CodeContests entry + code paths (Task 3 ACCEPT) ---
    spec_rel = "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    must_file(spec_rel)
    spec_path = repo_root / spec_rel
    spec_text = read_text(spec_path) if spec_path.is_file() else ""

    required_paths = [
        "scripts/evaluation/FABE/run_calculation.sh",
        "src/evaluation/FABE/Calculate_passk.py",
        "src/evaluation/FABE/aggregate_results.py",
    ]
    for rel in required_paths:
        must_contain(spec_text, rel, "spec.md")
        must_file(rel)

    required_io_patterns = [
        "results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl",
        "results/evaluation/FABE/<lang>/pass_at_k/shards/",
        "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
    ]
    for s in required_io_patterns:
        must_contain(spec_text, s, "spec.md")

    # --- CodeContests entry script should still call Calculate_passk.py and preserve path patterns ---
    run_calculation_rel = "scripts/evaluation/FABE/run_calculation.sh"
    run_sh_path = repo_root / run_calculation_rel
    run_sh_text = read_text(run_sh_path) if run_sh_path.is_file() else ""

    must_contain(run_sh_text, "Calculate_passk.py", run_calculation_rel)
    must_contain(run_sh_text, "aggregate_results.py", run_calculation_rel)
    must_contain(run_sh_text, "pass_at_k/inference_results.jsonl", run_calculation_rel)
    must_contain(run_sh_text, "pass_at_k/shards", run_calculation_rel)
    must_contain(run_sh_text, "pass_at_k/final_metrics.json", run_calculation_rel)

    ok = len(missing) == 0
    details["required_paths"] = required_paths
    details["required_io_patterns"] = required_io_patterns

    payload = {"ok": ok, "missing": missing, "details": details}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

