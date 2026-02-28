from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "openspec" / "project.md").is_file():
            return cur
        cur = cur.parent
    raise RuntimeError(f"Could not locate repo root from: {start}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _check_file_exists(repo_root: Path, rel_path: str) -> CheckResult:
    p = repo_root / rel_path
    if p.is_file():
        return CheckResult(True, f"OK: file exists: {rel_path}")
    return CheckResult(False, f"Missing file: {rel_path}")


def _check_doc_contains(repo_root: Path, rel_doc: str, needle: str) -> CheckResult:
    p = repo_root / rel_doc
    if not p.is_file():
        return CheckResult(False, f"Missing doc file: {rel_doc}")
    text = _read_text(p)
    if needle in text:
        return CheckResult(True, f"OK: {rel_doc} contains: {needle}")
    return CheckResult(False, f"Missing in {rel_doc}: {needle}")


def main() -> int:
    run_dir = Path(__file__).resolve().parent.parent
    repo_root = _find_repo_root(run_dir)

    required_files = [
        "scripts/evaluation/FABE/run_calculation.sh",
        "src/evaluation/FABE/Calculate_passk.py",
    ]

    docs_common = [
        "openspec/changes/propose-passk-test-on-expanded-multiple/proposal.md",
        "openspec/changes/propose-passk-test-on-expanded-multiple/design.md",
        "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md",
    ]

    required_strings_common = [
        "scripts/evaluation/FABE/run_calculation.sh",
        "src/evaluation/FABE/Calculate_passk.py",
        "results/evaluation/FABE/.../pass_at_k/final_metrics.json",
    ]

    doc_spec = "openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md"
    required_strings_spec_only = [
        "results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json",
    ]

    results: list[CheckResult] = []

    for rel in required_files:
        results.append(_check_file_exists(repo_root, rel))

    for doc in docs_common:
        for needle in required_strings_common:
            results.append(_check_doc_contains(repo_root, doc, needle))

    for needle in required_strings_spec_only:
        results.append(_check_doc_contains(repo_root, doc_spec, needle))

    failures = [r for r in results if not r.ok]
    for r in results:
        print(r.message)

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed.")
        return 1

    print("\nOK: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
