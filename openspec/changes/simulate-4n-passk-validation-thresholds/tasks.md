## Scope + Success
实现并验证 4N 模拟候选数据生成能力，以统一脚本校验门槛判断 pass@k 链路是否正确。
达标条件：正确答案输入 `pass@1 > 90%`，4N 模拟输入 `pass@4 > 50%`。
三条 benchmark 采用 AND 判定；`pass@1` 不通过则流程短路，不执行 `pass@4`。

- [x] 1. 在 `generate_11n_dataset.py` 设计 4N 模拟输出模式 [#R1]
  - ACCEPT: 每题可稳定产出 4 个候选；默认变换集合为 2 种变量名更改 + 2 种死代码插入；并保留 pass@k 所需关键字段。
  - TEST: Manual: 检查输出 JSONL 的 `candidates` 长度为 4，且含 `variant_type` 与问题 ID 字段。
  BUNDLE (RUN #4): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium SCOPE: CLI VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z HOW_TO_RUN: run.sh/run.bat
  BUNDLE (RUN #3): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium SCOPE: CLI VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-3__task-1__ref-R1__20260228T171911Z HOW_TO_RUN: run.sh/run.bat
  BUNDLE (RUN #2): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium SCOPE: CLI VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-2__task-1__ref-R1__20260228T171541Z HOW_TO_RUN: run.sh/run.bat
  - BUNDLE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium; SCOPE: CLI; VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-1__task-1__ref-R1__20260228T171304Z; HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #4): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z | WORKER_STARTUP_LOG: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-4__task-1__ref-R1__20260228T172309Z/logs/worker_startup.txt | VALIDATED_CLI: ./run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: 892d32d | COMMIT_MSG: "chore(openspec): complete task 1 for propose-passk-test-on-expanded-multiple" | FILES: src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py

- [ ] 2. 新建参数化总控脚本（N/扩充类型可配） [#R2]
  - ACCEPT: 新脚本支持至少以下参数：`--n`、`--augment_types`、`--mode`、`--stop_on_pass1_fail`。
  - TEST: CLI（实现阶段执行）:
    - `bash scripts/evaluation/FABE/run_passk_gate_validation.sh --help`
    - `bash scripts/evaluation/FABE/run_passk_gate_validation.sh --n 4 --augment_types rename1,rename2,dead1,dead2 --mode pass1_only`
  BUNDLE (RUN #5): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2-codex -c model_reasoning_effort=medium SCOPE: CLI VALIDATION_BUNDLE: auto_test_openspec/simulate-4n-passk-validation-thresholds/run-5__task-2__ref-R2__20260228T172957Z HOW_TO_RUN: run.sh/run.bat

- [ ] 3. 构建“正确答案输入”测试集并定义 pass@1 校验 [#R3]
  - ACCEPT: 对三条评测链路均可生成/提供正确答案输入文件，且每题仅填 1 个正确候选用于 pass@1。
  - TEST: CLI（实现阶段执行）:
    - `bash scripts/evaluation/FABE/run_calculation.sh`
    - `bash scripts/evaluation/FABE/run_calculation_humaneval.sh`
    - `bash scripts/evaluation/FABE/run_calculation_mbpp.sh`
    - 校验三条链路 `final_metrics.json` 均满足 `pass@1 > 90%`（AND）。

- [ ] 4. 构建“4N 模拟候选”测试集并定义 pass@4 校验 [#R4]
  - ACCEPT: 仅在 Task 3 通过后执行；三条链路都能消费 4N 输入并产出指标文件。
  - TEST: CLI（实现阶段执行）:
    - 运行同上三条脚本
    - 校验三条链路 `final_metrics.json` 均满足 `pass@4 > 50%`（AND）。

- [ ] 5. 增加脚本正确性判定规则与失败处理 [#R5]
  - ACCEPT: 任一阈值不达标即明确 FAIL，并输出定位信息（benchmark、路径、指标值），同时输出失败归因与样例日志。
  - TEST: Manual: 审核 `.../pass_at_k/gate_debug.json`，确认包含：
    - `sample_records`（数据集样例节选）
    - `sandbox_errors`（SandboxFusion 错误返回）
    - `root_cause_guess`（`dataset_issue` / `script_issue` / `sandbox_issue`）。
