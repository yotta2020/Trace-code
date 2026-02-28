@echo off
setlocal enabledelayedexpansion

set RUN_DIR=%~dp0
for %%I in ("%RUN_DIR%..\..\..") do set REPO_ROOT=%%~fI

set INPUT=%RUN_DIR%inputs\sample_1n.jsonl
set OUTPUT=%RUN_DIR%outputs\simulated_4n.jsonl

python "%REPO_ROOT%\src\data_preprocessing\CodeContestsPlus\gen_12n\generate_11n_dataset.py" "%INPUT%" "%OUTPUT%" --split test --lang cpp --mode 4n
python "%RUN_DIR%tests\validate_4n.py" --input "%OUTPUT%"
