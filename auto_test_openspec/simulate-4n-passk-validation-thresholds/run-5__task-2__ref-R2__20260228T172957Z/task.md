# Task 2 Validation Bundle

- change-id: simulate-4n-passk-validation-thresholds
- run#: 5
- task-id: 2
- ref-id: R2
- scope: CLI

## Purpose
验证新脚本 `scripts/evaluation/FABE/run_passk_gate_validation.sh` 的参数解析与帮助信息，确保至少支持：`--n`、`--augment_types`、`--mode`、`--stop_on_pass1_fail`。

## How To Run

### macOS/Linux
```bash
bash run.sh
```

### Windows
```bat
run.bat
```

## What This Runs
- `bash scripts/evaluation/FABE/run_passk_gate_validation.sh --help`
- `bash scripts/evaluation/FABE/run_passk_gate_validation.sh --n 4 --augment_types rename1,rename2,dead1,dead2 --mode pass1_only`

## Outputs
Logs are written to:
- `logs/help_stdout.txt`
- `logs/help_stderr.txt`
- `logs/run_stdout.txt`
- `logs/run_stderr.txt`

## Pass/Fail Criteria (Machine-Decidable)
The run is considered **pass** if all conditions below are met:
1. `run.sh`/`run.bat` exits with code 0.
2. `logs/help_stdout.txt` contains the option strings:
   - `--n`
   - `--augment_types`
   - `--mode`
   - `--stop_on_pass1_fail`
3. `logs/run_stdout.txt` contains:
   - `CONFIG: n=4`
   - `CONFIG: augment_types=rename1,rename2,dead1,dead2`
   - `CONFIG: mode=pass1_only`

## Notes
- This bundle validates CLI parsing and help output only.
- No input files are required and no output artifacts are produced beyond logs.
