import sys
from pathlib import Path

target_file = Path("openspec/changes/simulate-4n-passk-validation-thresholds/tasks.md")
content = target_file.read_text(encoding="utf-8")

if "EVIDENCE (RUN #4):" not in content:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "- [ ] 1. 在 `generate_11n_dataset.py` 设计 4N 模拟输出模式 [#R1]" in line:
            # mark as done
            lines[i] = lines[i].replace("- [ ]", "- [x]")
            # find end of the task
            insert_idx = i + 1
            while insert_idx < len(lines) and (lines[insert_idx].startswith("  -") or lines[insert_idx].startswith("    -") or "BUNDLE (RUN" in lines[insert_idx]):
                insert_idx += 1
            evidence_line = "  EVIDENCE (RUN #4): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z | WORKER_STARTUP_LOG: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z/logs/worker_startup.txt | VALIDATED_CLI: ./run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: 892d32d | COMMIT_MSG: \"chore(openspec): complete task 1 for propose-passk-test-on-expanded-multiple\" | FILES: src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py"
            lines.insert(insert_idx, evidence_line)
            break
    target_file.write_text('\n'.join(lines), encoding="utf-8")
    print("Patched tasks.md successfully")
else:
    print("Already patched")
