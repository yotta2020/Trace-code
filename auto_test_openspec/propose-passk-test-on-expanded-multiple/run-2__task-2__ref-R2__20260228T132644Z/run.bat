@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if not exist logs mkdir logs
if not exist outputs mkdir outputs

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY_BIN=py -3"
) else (
  set "PY_BIN=python"
)

%PY_BIN% --version > logs\provenance.txt 2>&1

%PY_BIN% tests\test_r2_spec_chain.py > logs\run_stdout.txt 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

type logs\run_stdout.txt
exit /b %EXIT_CODE%

