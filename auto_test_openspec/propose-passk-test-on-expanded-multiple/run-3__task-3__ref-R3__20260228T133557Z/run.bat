@echo off
setlocal enabledelayedexpansion

set "SELF_DIR=%~dp0"
pushd "%SELF_DIR%" >nul

if not exist logs mkdir logs
if not exist outputs mkdir outputs

set "PY_BIN="
where python3 >nul 2>nul
if %errorlevel%==0 (
  set "PY_BIN=python3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY_BIN=python"
  )
)

if "%PY_BIN%"=="" (
  echo ERROR: python/python3 not found in PATH 1>&2
  popd >nul
  exit /b 2
)

(
  echo UTC_START=%%DATE%%T%%TIME%%
  echo PY_BIN=%PY_BIN%
  %PY_BIN% --version 2^>^&1
) > logs\\run.txt

%PY_BIN% -u tests\\validate_r3.py >> logs\\run.txt 2>&1
set "RC=%errorlevel%"
type logs\\run.txt

popd >nul
exit /b %RC%
