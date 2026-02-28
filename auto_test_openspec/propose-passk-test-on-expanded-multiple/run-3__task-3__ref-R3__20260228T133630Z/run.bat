@echo off
setlocal enabledelayedexpansion

set "RUN_DIR=%~dp0"
set "LOG_DIR=%RUN_DIR%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

REM Run from repo root regardless of invocation CWD.
for %%I in ("%RUN_DIR%..\..\..") do set "REPO_ROOT=%%~fI"
pushd "%REPO_ROOT%" >nul

set "SPEC_PATH=openspec\changes\propose-passk-test-on-expanded-multiple\specs\passk-evaluation\spec.md"
set "CODECONTESTS_SCRIPT=scripts\evaluation\FABE\run_calculation.sh"
set "CALC_PY=src\evaluation\FABE\Calculate_passk.py"

call :log "Repo root: %REPO_ROOT%"
call :log "Run dir :  %RUN_DIR%"

if not exist "%SPEC_PATH%" call :fail "Missing spec: %SPEC_PATH%"
if not exist "%CODECONTESTS_SCRIPT%" call :fail "Missing script: %CODECONTESTS_SCRIPT%"
if not exist "%CALC_PY%" call :fail "Missing code: %CALC_PY%"

findstr /C:"scripts/evaluation/FABE/run_calculation.sh" "%SPEC_PATH%" >nul || call :fail "Spec missing CodeContests entry script path"
findstr /C:"src/evaluation/FABE/Calculate_passk.py" "%SPEC_PATH%" >nul || call :fail "Spec missing CodeContests calculation code path"
findstr /C:"results/evaluation/FABE/<lang>/pass_at_k/final_metrics.json" "%SPEC_PATH%" >nul || call :fail "Spec missing final_metrics.json path contract for CodeContests"

findstr /C:"src/evaluation/FABE/Calculate_passk.py" "%CODECONTESTS_SCRIPT%" >nul || call :fail "CodeContests script no longer references Calculate_passk.py"
findstr /C:"results/evaluation/FABE/${TARGET_LANG}/pass_at_k/inference_results.jsonl" "%CODECONTESTS_SCRIPT%" >nul || call :fail "CodeContests script input path pattern changed"
findstr /C:"pass_at_k/final_metrics.json" "%CODECONTESTS_SCRIPT%" >nul || call :fail "CodeContests script output path contract changed"

call :log "OK: CodeContests pass@k paths remain compatible (static checks passed)."
popd >nul
exit /b 0

:fail
call :log "ERROR: %~1"
popd >nul
exit /b 1

:log
echo %~1
echo %~1>>"%LOG_DIR%run_stdout.txt"
exit /b 0

