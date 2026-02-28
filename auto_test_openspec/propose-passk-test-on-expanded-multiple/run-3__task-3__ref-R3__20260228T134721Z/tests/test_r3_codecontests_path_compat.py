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

    spec_path = (
        repo_root
        / "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    )
    must_exist(spec_path)
    spec_text = spec_path.read_text(encoding="utf-8")

    # --- Spec must explicitly declare CodeContests entry + calculator paths (ACCEPT for task R3) ---
    required_paths = [
        "scripts/evaluation/FABE/run_calculation.sh",
        "src/evaluation/FABE/Calculate_passk.py",
        "src/evaluation/FABE/aggregate_results.py",
    ]
    for rel in required_paths:
        must_contain(spec_text, rel)
        must_exist(repo_root / rel)

    # --- Spec must spell out CodeContests I/O path patterns and shared output contract ---
    required_io_strings = [
        "results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl",
        "results/evaluation/FABE/<lang>/pass_at_k/shards/",
        "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
        "pass_at_k/final_metrics.json",
    ]
    for s in required_io_strings:
        must_contain(spec_text, s)

    # --- CodeContests language codes must include legacy py3 to avoid path-breaking renames ---
    must_contain(spec_text, "py3")

    # --- Entry script should invoke the referenced code and preserve default path layout ---
    calc_sh_path = repo_root / "scripts/evaluation/FABE/run_calculation.sh"
    calc_sh = calc_sh_path.read_text(encoding="utf-8")

    must_contain(calc_sh, "src/evaluation/FABE/Calculate_passk.py")
    must_contain(calc_sh, "src/evaluation/FABE/aggregate_results.py")
    must_contain(calc_sh, "pass_at_k/inference_results.jsonl")
    must_contain(calc_sh, "pass_at_k/shards")
    must_contain(calc_sh, "pass_at_k/final_metrics.json")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

