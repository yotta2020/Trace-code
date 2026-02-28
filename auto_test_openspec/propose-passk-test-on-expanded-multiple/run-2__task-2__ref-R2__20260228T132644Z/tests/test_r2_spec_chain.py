from __future__ import annotations

import re
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "openspec" / "project.md").is_file():
            return candidate
    raise RuntimeError(f"Could not find repo root from: {start}")


def must_exist(path: Path) -> None:
    assert path.is_file(), f"Missing required file: {path}"


def must_contain(text: str, needle: str) -> None:
    assert needle in text, f"Missing required reference: {needle}"


def main() -> int:
    repo_root = find_repo_root(Path(__file__).resolve())

    spec_path = repo_root / "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    must_exist(spec_path)
    spec_text = spec_path.read_text(encoding="utf-8")

    # --- Spec must reference required entry scripts and code paths (ACCEPT for task R2) ---
    required_paths = [
        "scripts/evaluation/FABE/run_calculation_humaneval.sh",
        "scripts/evaluation/FABE/run_calculation_mbpp.sh",
        "src/evaluation/FABE/Calculate_passk_multiple.py",
        "src/evaluation/FABE/aggregate_results.py",
    ]
    for rel in required_paths:
        must_contain(spec_text, rel)
        must_exist(repo_root / rel)

    # --- Spec must spell out the output contract (final_metrics.json) ---
    must_contain(spec_text, "pass_at_k/final_metrics.json")
    must_contain(spec_text, "final_metrics.json")

    # --- Spec must include HumanEval/MBPP default I/O path patterns (script -> output-path chain) ---
    required_io_patterns = [
        "results/evaluation/FABE/humaneval_<lang>/pass_at_k/inference_results.jsonl",
        "results/evaluation/FABE/humaneval_<lang>/pass_at_k/shards/",
        "results/evaluation/FABE/humaneval_<lang>/pass_at_k/final_metrics.json",
        "results/evaluation/FABE/mbpp_<lang>/pass_at_k/inference_results.jsonl",
        "results/evaluation/FABE/mbpp_<lang>/pass_at_k/shards/",
        "results/evaluation/FABE/mbpp_<lang>/pass_at_k/final_metrics.json",
    ]
    for s in required_io_patterns:
        must_contain(spec_text, s)

    # --- Default k=4 declared (either explicit k=4 or pass@4) ---
    if not (re.search(r"\bk\s*=\s*4\b", spec_text) or ("pass@4" in spec_text)):
        raise AssertionError("Missing default k=4 declaration (expected 'k=4' or 'pass@4').")

    # --- Entry scripts should invoke the referenced code and define input/output defaults ---
    humaneval_sh = (repo_root / "scripts/evaluation/FABE/run_calculation_humaneval.sh").read_text(encoding="utf-8")
    mbpp_sh = (repo_root / "scripts/evaluation/FABE/run_calculation_mbpp.sh").read_text(encoding="utf-8")
    for script_text, label in [(humaneval_sh, "humaneval"), (mbpp_sh, "mbpp")]:
        assert "Calculate_passk_multiple.py" in script_text, f"{label} script does not call Calculate_passk_multiple.py"
        assert "aggregate_results.py" in script_text, f"{label} script does not call aggregate_results.py"
        assert "inference_results.jsonl" in script_text, f"{label} script does not define inference_results.jsonl default"
        assert "final_metrics.json" in script_text, f"{label} script does not define final_metrics.json output"

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

