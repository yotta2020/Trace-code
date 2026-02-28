import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "openspec").is_dir() and (p / "auto_test_openspec").is_dir():
            return p
    raise RuntimeError(f"Could not locate repo root from: {start}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_all(text: str, needles: list[str], label: str) -> list[str]:
    missing = [n for n in needles if n not in text]
    if missing:
        return [f"{label}: missing {missing}"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate R1 doc consistency for MultiPL-E expanded dataset spec.")
    ap.add_argument(
        "--report",
        default=None,
        help="Output JSON report path. Default: <run-dir>/outputs/validation_report.json",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_dir = script_dir.parent
    repo_root = find_repo_root(run_dir)

    change_dir = repo_root / "openspec" / "changes" / "propose-passk-test-on-expanded-multiple"
    paths = {
        "proposal.md": change_dir / "proposal.md",
        "design.md": change_dir / "design.md",
        "spec.md": change_dir / "specs" / "passk-evaluation" / "spec.md",
    }

    errors: list[str] = []
    for name, p in paths.items():
        if not p.is_file():
            errors.append(f"{name}: file not found at {p}")

    if errors:
        print("ALL_CHECKS_PASS=0")
        for e in errors:
            print(f"[ERROR] {e}")
        return 1

    proposal = read_text(paths["proposal.md"])
    design = read_text(paths["design.md"])
    spec = read_text(paths["spec.md"])

    # R1 requires docs to cite these reference paths.
    ref_paths = [
        "scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh",
        "src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py",
    ]

    # Minimal field tokens that must be explicitly documented for the expanded JSONL contract.
    # Use backticked needles to avoid accidental substring matches.
    field_tokens = ["`candidates`", "`variant_type`", "`name`", "`task_id`", "`problem_id`"]

    # Language scope for this change (MultiPL-E subset naming).
    # Use backticked needles to avoid accidental substring matches (e.g. "py" in other words).
    lang_tokens = ["`cpp`", "`java`", "`py`"]

    errors += require_all(proposal, ref_paths, "proposal.md refs")
    errors += require_all(design, ref_paths, "design.md refs")
    errors += require_all(spec, ref_paths, "spec.md refs")

    errors += require_all(proposal, field_tokens, "proposal.md fields")
    errors += require_all(design, field_tokens, "design.md fields")
    errors += require_all(spec, field_tokens, "spec.md fields")

    errors += require_all(proposal, lang_tokens, "proposal.md langs")
    errors += require_all(design, lang_tokens, "design.md langs")
    errors += require_all(spec, lang_tokens, "spec.md langs")

    # Spec should explicitly state JSONL + UTF-8 for the executable input contract.
    errors += require_all(spec, ["JSONL", "UTF-8"], "spec.md format")

    report = {
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(repo_root),
        "checked_files": {k: str(v) for k, v in paths.items()},
        "ref_paths_required": ref_paths,
        "field_tokens_required": field_tokens,
        "lang_tokens_required": lang_tokens,
        "errors": errors,
        "all_checks_pass": len(errors) == 0,
    }

    out_path = Path(args.report) if args.report else (run_dir / "outputs" / "validation_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if errors:
        print("ALL_CHECKS_PASS=0")
        for e in errors:
            print(f"[ERROR] {e}")
        print(f"[INFO] Wrote report: {out_path}")
        return 1

    print("ALL_CHECKS_PASS=1")
    print(f"[OK] R1 docs contain required references, fields, and language scope.")
    print(f"[INFO] Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
