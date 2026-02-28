import sys
from pathlib import Path

target_file = Path("openspec/changes/simulate-4n-passk-validation-thresholds/tasks.md")
content = target_file.read_text(encoding="utf-8")

if "BUNDLE (RUN #4):" not in content:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "- [ ] 1. 在 `generate_11n_dataset.py` 设计 4N 模拟输出模式 [#R1]" in line:
            # find end of the task
            insert_idx = i + 1
            while insert_idx < len(lines) and (lines[insert_idx].startswith("  -") or lines[insert_idx].startswith("    -")):
                insert_idx += 1
            bundle_line = "  BUNDLE (RUN #4): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z | HOW_TO_RUN: run.sh/run.bat"
            lines.insert(insert_idx, bundle_line)
            break
    target_file.write_text('\n'.join(lines), encoding="utf-8")
    print("Patched tasks.md successfully")
else:
    print("Already patched")
