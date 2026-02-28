import argparse
import json
import os
import sys


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec_path = os.path.join(
        args.repo_root,
        "openspec",
        "changes",
        "propose-passk-test-on-expanded-multiple",
        "specs",
        "passk-evaluation",
        "spec.md",
    )
    text = read_text(spec_path)

    must_contain = [
        "scripts/evaluation/FABE/run_calculation_humaneval.sh",
        "scripts/evaluation/FABE/run_calculation_mbpp.sh",
        "src/evaluation/FABE/Calculate_passk_multiple.py",
        "src/evaluation/FABE/aggregate_results.py",
        "results/evaluation/FABE/.../pass_at_k/final_metrics.json",
        "k=4",
    ]

    missing = [s for s in must_contain if s not in text]
    ok = len(missing) == 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ok": ok,
                "checked_file": os.path.relpath(spec_path, args.repo_root),
                "missing": missing,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if not ok:
        sys.stderr.write("Missing required spec anchors:\\n")
        for s in missing:
            sys.stderr.write(f"- {s}\\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

