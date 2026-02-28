from __future__ import annotations

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

    # --- Spec must explicitly reference CodeContests entry + code paths (ACCEPT for task R3) ---
    required_paths = [
        "scripts/evaluation/FABE/run_calculation.sh",
        "src/evaluation/FABE/Calculate_passk.py",
        "src/evaluation/FABE/aggregate_results.py",
    ]
    for rel in required_paths:
        must_contain(spec_text, rel)
        must_exist(repo_root / rel)

    # --- Spec must spell out CodeContests I/O path contract (final_metrics.json) ---
    required_io_patterns = [
        "results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl",
        "results/evaluation/FABE/<lang>/pass_at_k/shards/",
        "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
        "pass_at_k/final_metrics.json",
    ]
    for s in required_io_patterns:
        must_contain(spec_text, s)

    # --- CodeContests entry script should still call the existing code paths and preserve path patterns ---
    run_calculation_sh = (repo_root / "scripts/evaluation/FABE/run_calculation.sh").read_text(encoding="utf-8")
    assert "Calculate_passk.py" in run_calculation_sh, "run_calculation.sh does not call Calculate_passk.py"
    assert "aggregate_results.py" in run_calculation_sh, "run_calculation.sh does not call aggregate_results.py"
    assert "pass_at_k/inference_results.jsonl" in run_calculation_sh, "run_calculation.sh missing inference_results.jsonl pattern"
    assert "pass_at_k/shards" in run_calculation_sh, "run_calculation.sh missing shards output dir pattern"
    assert "pass_at_k/final_metrics.json" in run_calculation_sh, "run_calculation.sh missing final_metrics.json output pattern"

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

