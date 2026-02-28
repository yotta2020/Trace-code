@echo off
setlocal enabledelayedexpansion

REM Run from this script's directory
cd /d "%~dp0"

if not exist logs mkdir logs
if not exist outputs mkdir outputs

echo UTC_TIMESTAMP=^%DATE^%T^%TIME^%> logs\\runner_provenance.txt
echo PWD=%CD%>> logs\\runner_provenance.txt

set PY_EXE=
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set PY_EXE=py -3
) else (
  where python >nul 2>&1
  if %ERRORLEVEL% EQU 0 (
    set PY_EXE=python
  ) else (
    echo [ERROR] No Python found (need Python 3).>> logs\\validation.txt
    exit /b 1
  )
)

%PY_EXE% -V >> logs\\runner_provenance.txt 2>&1

%PY_EXE% tests\\check_r3_codecontests_path_compat.py --report outputs\\report.json > logs\\validation.txt 2>&1
exit /b %ERRORLEVEL%

