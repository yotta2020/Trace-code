## Scope + Success
目标是形成“扩充 MultiPL-E 数据 + SandboxFusion 重测 + pass@k 聚合”可复现流程，并确保最终输出为 `results/evaluation/FABE/.../pass_at_k/final_metrics.json`。
默认聚合指标为 `k=4`，当前语言范围限定为 C++、Java、Python。

- [x] 1. 定义 MultiPL-E 扩充数据集规范（HumanEval/MBPP）[#R1]
  - ACCEPT: 文档明确引用：
    - `scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh`
    - `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py`
  - ACCEPT: 明确扩充数据最小字段：`candidates`、`variant_type`、`name/task_id/problem_id`（至少一个）。
  - ACCEPT: 明确当前评估仅覆盖 C++、Java、Python 三种语言。
  - TEST: Manual: 检查 `proposal.md`、`design.md`、`specs/passk-evaluation/spec.md` 字段与路径定义一致。
  - BUNDLE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/propose-passk-test-on-expanded-multiple/run-1__task-1__ref-R1__20260228T131053Z | HOW_TO_RUN: run.sh/run.bat
  - EVIDENCE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/propose-passk-test-on-expanded-multiple/run-1__task-1__ref-R1__20260228T131053Z | WORKER_STARTUP_LOG: auto_test_openspec/propose-passk-test-on-expanded-multiple/run-1__task-1__ref-R1__20260228T131053Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/propose-passk-test-on-expanded-multiple/run-1__task-1__ref-R1__20260228T131053Z/run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: a38053b | COMMIT_MSG: "chore(openspec): complete task 1 for propose-passk-test-on-expanded-multiple" | DIFFSTAT: "19 files changed, 224 insertions(+), 1619 deletions(-)"

- [ ] 2. 定义扩充数据输入 SandboxFusion 的重测流程（HumanEval/MBPP）[#R2]
  - ACCEPT: 明确入口脚本：
    - `scripts/evaluation/FABE/run_calculation_humaneval.sh`
    - `scripts/evaluation/FABE/run_calculation_mbpp.sh`
  - ACCEPT: 明确计算代码：`src/evaluation/FABE/Calculate_passk_multiple.py`
  - ACCEPT: 明确聚合代码：`src/evaluation/FABE/aggregate_results.py`
  - ACCEPT: 明确默认 pass@k 使用 `k=4`，并允许后续参数化扩展。
  - TEST: Manual: 核对 spec 中“脚本 -> 代码 -> 输出路径”链路完整。
  - BUNDLE (RUN #2): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/propose-passk-test-on-expanded-multiple/run-2__task-2__ref-R2__20260228T132644Z | HOW_TO_RUN: run.sh/run.bat

- [ ] 3. 保持 CodeContests pass@k 路径兼容性 [#R3]
  - ACCEPT: 明确 CodeContests 入口与计算代码不变：
    - `scripts/evaluation/FABE/run_calculation.sh`
    - `src/evaluation/FABE/Calculate_passk.py`
  - ACCEPT: 最终输出路径契约与 MultiPL-E 一致（`.../pass_at_k/final_metrics.json`）。
  - TEST: Manual: 核对 spec 中 CodeContests 路径被显式声明且无破坏性变更。
  - BUNDLE (RUN #3): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/propose-passk-test-on-expanded-multiple/run-3__task-3__ref-R3__20260228T135815Z | HOW_TO_RUN: run.sh/run.bat

- [ ] 4. 提供实现阶段验证命令与结果检查 [#R4]
  - ACCEPT: 提供三个可执行入口命令：
    - `bash scripts/evaluation/FABE/run_calculation.sh`
    - `bash scripts/evaluation/FABE/run_calculation_humaneval.sh`
    - `bash scripts/evaluation/FABE/run_calculation_mbpp.sh`
  - ACCEPT: 提供结果文件检查命令示例：
    - `test -f results/evaluation/FABE/<benchmark>/pass_at_k/final_metrics.json`
  - ACCEPT: 提供 JSON 字段检查命令示例（`benchmark`、`pass@1`、`pass@4`、`total_candidates`、`evaluated_candidates`）。
  - TEST: SCOPE: CLI
    - 执行上述脚本与文件检查命令，确认流程可落地。
    - 执行字段检查示例：
      - `python -c "import json; p='results/evaluation/FABE/<benchmark>/pass_at_k/final_metrics.json'; d=json.load(open(p)); req=['benchmark','pass@1','pass@4','total_candidates','evaluated_candidates']; missing=[k for k in req if k not in d]; assert not missing, missing"`
