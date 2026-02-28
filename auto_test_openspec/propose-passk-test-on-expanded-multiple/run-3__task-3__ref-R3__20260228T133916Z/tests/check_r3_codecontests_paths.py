import argparse
import json
import os
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    spec_path = repo_root / "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    run_calc = repo_root / "scripts/evaluation/FABE/run_calculation.sh"
    calc_py = repo_root / "src/evaluation/FABE/Calculate_passk.py"

    required_spec_strings = [
        "scripts/evaluation/FABE/run_calculation.sh",
        "src/evaluation/FABE/Calculate_passk.py",
        "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
    ]

    required_run_calc_strings = [
        "pass_at_k/inference_results.jsonl",
        "pass_at_k/final_metrics.json",
        "src/evaluation/FABE/Calculate_passk.py",
    ]

    report = {
        "ok": True,
        "repo_root": str(repo_root),
        "checks": {},
    }

    def check(name: str, condition: bool, details: str) -> None:
        report["checks"][name] = {"ok": bool(condition), "details": details}
        if not condition:
            report["ok"] = False

    check("spec_exists", spec_path.is_file(), f"spec_path={spec_path}")
    check("run_calculation_sh_exists", run_calc.is_file(), f"path={run_calc}")
    check("calculate_passk_py_exists", calc_py.is_file(), f"path={calc_py}")

    if spec_path.is_file():
        spec_text = _read_text(spec_path)
        for s in required_spec_strings:
            check(
                f"spec_contains::{s}",
                s in spec_text,
                f"missing={s}",
            )

    if run_calc.is_file():
        run_calc_text = _read_text(run_calc)
        for s in required_run_calc_strings:
            check(
                f"run_calculation_sh_contains::{s}",
                s in run_calc_text,
                f"missing={s}",
            )

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"ok": report["ok"]}, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

