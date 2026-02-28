@echo off
setlocal enabledelayedexpansion

rem Run from any working directory.
pushd "%~dp0"

set "REPO_ROOT=%~dp0..\..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

if not exist "logs" mkdir "logs"
if not exist "outputs" mkdir "outputs"

(
  echo UTC_TIMESTAMP: %DATE% %TIME%
  echo REPO_ROOT: %REPO_ROOT%
  where python 2^>nul
  python --version 2^>^&1
) > "logs\\provenance.txt"

python -u "tests\\check_r3_codecontests_paths.py" ^
  --repo_root "%REPO_ROOT%" ^
  --out_json "outputs\\r3_check_report.json" > "logs\\check_r3_stdout.txt" 2>&1
set "ERR=%ERRORLEVEL%"

type "logs\\check_r3_stdout.txt"
echo Wrote: outputs\\r3_check_report.json

popd
exit /b %ERR%
