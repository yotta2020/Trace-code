import argparse
import json
import os
import sys
from datetime import datetime, timezone


REQUIRED_REPO_FILES = [
    "scripts/evaluation/FABE/run_calculation.sh",
    "src/evaluation/FABE/Calculate_passk.py",
    "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md",
]

REQUIRED_SPEC_SUBSTRINGS = [
    "scripts/evaluation/FABE/run_calculation.sh",
    "src/evaluation/FABE/Calculate_passk.py",
    "results/evaluation/FABE/<lang>/pass_at_k/inference_results.jsonl",
    "results/evaluation/FABE/<lang>/pass_at_k/shards/",
    "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
]


def find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(20):
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError("Unable to locate repo root (missing .git in parents)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to write JSON report")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.abspath(os.path.join(script_dir, ".."))
    repo_root = find_repo_root(run_dir)

    report = {
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": repo_root,
        "run_dir": run_dir,
        "checks": {},
        "ok": True,
    }

    # 1) Referenced repo files exist
    for rel_path in REQUIRED_REPO_FILES:
        abs_path = os.path.join(repo_root, rel_path)
        ok = os.path.isfile(abs_path)
        report["checks"][f"file_exists:{rel_path}"] = ok
        if not ok:
            report["ok"] = False

    # 2) Spec contains explicit CodeContests path contract
    spec_rel = "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    spec_path = os.path.join(repo_root, spec_rel)
    spec_text = ""
    if os.path.isfile(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_text = f.read()

    for needle in REQUIRED_SPEC_SUBSTRINGS:
        ok = needle in spec_text
        report["checks"][f"spec_contains:{needle}"] = ok
        if not ok:
            report["ok"] = False

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["ok"]:
        print("[OK] R3 CodeContests pass@k path compatibility checks passed.")
        return 0

    print("[FAIL] R3 checks failed. See report for details:", args.report)
    for k, v in report["checks"].items():
        if not v:
            print(" -", k)
    return 1


if __name__ == "__main__":
    sys.exit(main())

