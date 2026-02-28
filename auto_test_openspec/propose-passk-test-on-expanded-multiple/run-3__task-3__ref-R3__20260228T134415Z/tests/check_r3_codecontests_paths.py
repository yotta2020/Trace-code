import argparse
import json
import re
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    checks = []
    errors = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    # --- Existence checks (path compatibility: entry + calc module stay in place) ---
    entry_script = repo_root / "scripts/evaluation/FABE/run_calculation.sh"
    calc_module = repo_root / "src/evaluation/FABE/Calculate_passk.py"
    spec_path = (
        repo_root
        / "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    )

    check("exists: entry_script", entry_script.is_file(), str(entry_script))
    check("exists: calc_module", calc_module.is_file(), str(calc_module))
    check("exists: spec", spec_path.is_file(), str(spec_path))

    # --- Content checks: spec explicitly declares CodeContests path contract ---
    if spec_path.is_file():
        spec = _read_text(spec_path)
        required_spec_snippets = [
            "### Requirement: CodeContests 与 MultiPL-E 的 pass@k 输出契约必须一致",
            "`scripts/evaluation/FABE/run_calculation.sh`",
            "`src/evaluation/FABE/Calculate_passk.py`",
            "results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl",
            "results/evaluation/FABE/<lang>/pass_at_k/shards/",
            "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
        ]
        for s in required_spec_snippets:
            check(f"spec contains: {s}", s in spec, "missing snippet")

    # --- Content checks: entry script defaults keep the historical path pattern ---
    if entry_script.is_file():
        sh = _read_text(entry_script)

        # Must still point to the same calculation module.
        check(
            "entry references Calculate_passk.py",
            "src/evaluation/FABE/Calculate_passk.py" in sh,
            "missing reference to Calculate_passk.py",
        )

        # Default path contract (keep results/evaluation/FABE/<lang>/pass_at_k/*).
        required_path_patterns = [
            r"results/evaluation/FABE/\$\{TARGET_LANG\}/pass_at_k/inference_results\.jsonl",
            r"results/evaluation/FABE/\$\{TARGET_LANG\}/pass_at_k/shards",
            r"results/evaluation/FABE/\$\{TARGET_LANG\}/pass_at_k/final_metrics\.json",
        ]
        for pat in required_path_patterns:
            ok = re.search(pat, sh) is not None
            check(f"entry default path pattern: {pat}", ok, "pattern not found")

    result = {"ok": len(errors) == 0, "errors": errors, "checks": checks}
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("All checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

