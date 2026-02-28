@echo off
setlocal enabledelayedexpansion

set "RUN_DIR=%~dp0"
rem ROOT_DIR = <RUN_DIR>\..\..\.. (repo root)
for %%I in ("%RUN_DIR%..\..\..") do set "ROOT_DIR=%%~fI"

if not exist "%RUN_DIR%logs" mkdir "%RUN_DIR%logs"
if not exist "%RUN_DIR%outputs" mkdir "%RUN_DIR%outputs"

(
  echo UTC_START=%DATE% %TIME%
  echo RUN_DIR=%RUN_DIR%
  echo ROOT_DIR=%ROOT_DIR%
  echo.
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    for /f "delims=" %%V in ('python --version 2^>^&1') do echo [provenance] %%V
  ) else (
    echo [provenance] python: (not found on PATH)
  )
  where uv >nul 2>nul
  if %ERRORLEVEL%==0 (
    for /f "delims=" %%V in ('uv --version 2^>^&1') do echo [provenance] uv: %%V
  ) else (
    echo [provenance] uv: (not installed)
  )
  where git >nul 2>nul
  if %ERRORLEVEL%==0 (
    for /f "delims=" %%G in ('git -C "%ROOT_DIR%" rev-parse --short HEAD 2^>^&1') do echo [provenance] git_base: %%G
  )
  echo [provenance] deps: stdlib-only (no installs)
) > "%RUN_DIR%logs\\provenance.txt"

python "%RUN_DIR%tests\\check_r2_spec_chain.py" --repo-root "%ROOT_DIR%" --out "%RUN_DIR%outputs\\check_r2_spec_chain.json" 1> "%RUN_DIR%logs\\run_stdout.txt" 2> "%RUN_DIR%logs\\run_stderr.txt"
if %ERRORLEVEL% NEQ 0 (
  echo FAIL: see logs\\run_stderr.txt
  exit /b %ERRORLEVEL%
)

echo OK: wrote outputs\\check_r2_spec_chain.json
exit /b 0

