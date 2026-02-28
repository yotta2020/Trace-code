import os
import sys


def _repo_root(run_dir: str) -> str:
    # run_dir: .../auto_test_openspec/<change-id>/<run-folder>/
    return os.path.abspath(os.path.join(run_dir, "..", "..", ".."))


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise AssertionError(f"missing path: {path}")


def main() -> int:
    run_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_root = _repo_root(run_dir)

    spec_path = os.path.join(
        repo_root,
        "openspec",
        "changes",
        "propose-passk-test-on-expanded-multiple",
        "specs",
        "passk-evaluation",
        "spec.md",
    )
    calc_sh_path = os.path.join(repo_root, "scripts", "evaluation", "FABE", "run_calculation.sh")
    calc_py_path = os.path.join(repo_root, "src", "evaluation", "FABE", "Calculate_passk.py")

    _must_exist(spec_path)
    _must_exist(calc_sh_path)
    _must_exist(calc_py_path)

    spec = _read_text(spec_path)
    calc_sh = _read_text(calc_sh_path)

    # R3: "paths explicitly declared" in spec (CodeContests + output contract).
    required_spec_substrings = [
        "#### Scenario: CodeContests 流程保持可用",
        "`scripts/evaluation/FABE/run_calculation.sh`",
        "`src/evaluation/FABE/Calculate_passk.py`",
        "`src/evaluation/FABE/aggregate_results.py`",
        "results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl",
        "results/evaluation/FABE/<lang>/pass_at_k/shards/",
        "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
    ]
    missing = [s for s in required_spec_substrings if s not in spec]
    if missing:
        raise AssertionError(f"spec missing required substrings: {missing}")

    # R3: ensure CodeContests script still uses pass_at_k paths and writes final_metrics.json.
    required_script_substrings = [
        "src/evaluation/FABE/Calculate_passk.py",
        "src/evaluation/FABE/aggregate_results.py",
        "/pass_at_k/inference_results.jsonl",
        "/pass_at_k/shards",
        "/pass_at_k/final_metrics.json",
    ]
    missing = [s for s in required_script_substrings if s not in calc_sh]
    if missing:
        raise AssertionError(f"run_calculation.sh missing required substrings: {missing}")

    print("OK: R3 textual path-compatibility checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
