@echo off
setlocal enabledelayedexpansion
set ROOT=%~dp0..\..\..
for %%I in ("%ROOT%") do set ROOT=%%~fI
set RUN_DIR=%ROOT%\auto_test_openspec\simulate-4n-passk-validation-thresholds\run-3__task-1__ref-R1__20260228T171911Z

python "%ROOT%\src\data_preprocessing\CodeContestsPlus\gen_12n\generate_11n_dataset.py" ^
  "%RUN_DIR%\inputs\sample_1n.jsonl" ^
  "%RUN_DIR%\outputs\simulate_4n.jsonl" ^
  --mode simulate_4n ^
  --lang cpp ^
  --split train ^
  --seed 42

python "%RUN_DIR%\check_outputs.py" "%RUN_DIR%\outputs\simulate_4n.jsonl"
