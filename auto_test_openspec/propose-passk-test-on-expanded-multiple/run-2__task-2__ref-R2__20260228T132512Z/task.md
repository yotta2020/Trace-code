# Validation Bundle (Worker) — Task 2 / Ref R2

Change: `propose-passk-test-on-expanded-multiple`  
Task: 2. 定义扩充数据输入 SandboxFusion 的重测流程（HumanEval/MBPP）[#R2]

This bundle validates (machine-checkable) that the canonical spec documents a complete “script -> code -> output path” chain for routing expanded MultiPL-E HumanEval/MBPP inference results through SandboxFusion re-testing and aggregation, including the default `k=4` statement.

## How to run

### Linux/macOS (bash)
```bash
bash run.sh
```

### Windows (cmd)
```bat
run.bat
```

## What it does
- Reads `openspec/changes/propose-passk-test-on-expanded-multiple/specs/passk-evaluation/spec.md`
- Asserts it explicitly references:
  - Entry scripts:
    - `scripts/evaluation/FABE/run_calculation_humaneval.sh`
    - `scripts/evaluation/FABE/run_calculation_mbpp.sh`
  - Calculation code:
    - `src/evaluation/FABE/Calculate_passk_multiple.py`
  - Aggregation code:
    - `src/evaluation/FABE/aggregate_results.py`
  - Output contract:
    - `results/evaluation/FABE/.../pass_at_k/final_metrics.json`
  - Default pass@k:
    - `k=4`
- Writes a machine-readable result to `outputs/check_r2_spec_chain.json`

## Pass/fail criteria (machine-decidable)
- **PASS (for this bundle):** `run.sh` / `run.bat` exits with code `0` AND `outputs/check_r2_spec_chain.json` contains `"ok": true`.
- **FAIL:** non-zero exit code OR `"ok": false` with a non-empty `"missing"` list.

Notes:
- This bundle does not start SandboxFusion or run evaluations; it validates the documentation chain required by Task 2’s TEST (Manual) in a reproducible, automated way.
- Supervisor is the final verifier; this worker bundle provides artifacts only.

