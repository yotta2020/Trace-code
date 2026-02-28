@echo off
setlocal enabledelayedexpansion

set "RUN_DIR=%~dp0"
set "RUN_DIR=%RUN_DIR:~0,-1%"

rem repo root = RUN_DIR\..\..\..
pushd "%RUN_DIR%\..\..\.." >nul
set "REPO_ROOT=%CD%"
popd >nul

if not exist "%RUN_DIR%\logs" mkdir "%RUN_DIR%\logs"
if not exist "%RUN_DIR%\outputs" mkdir "%RUN_DIR%\outputs"

set "PYTHON_BIN=%PYTHON_BIN%"
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"

(
  echo RUN_DIR=%RUN_DIR%
  echo REPO_ROOT=%REPO_ROOT%
  for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -AsUTC -Format yyyy-MM-ddTHH:mm:ssZ"') do echo UTC_TIMESTAMP=%%i
  echo PYTHON_BIN=%PYTHON_BIN%
  echo --- python version ---
  "%PYTHON_BIN%" --version
  echo.
  "%PYTHON_BIN%" "%RUN_DIR%\tests\check_r3_codecontests_paths.py" ^
    --repo_root "%REPO_ROOT%" ^
    --out_json "%RUN_DIR%\outputs\check_results.json"
) 1> "%RUN_DIR%\logs\run.txt" 2>&1

type "%RUN_DIR%\logs\run.txt"
exit /b %errorlevel%

