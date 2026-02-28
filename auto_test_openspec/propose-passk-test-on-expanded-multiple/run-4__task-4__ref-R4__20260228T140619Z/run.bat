@echo off
setlocal enableextensions enabledelayedexpansion

REM This bundle requires bash (Git Bash or WSL) in PATH.
where bash >NUL 2>&1
if not "%ERRORLEVEL%"=="0" (
  echo ERROR: bash not found in PATH. Install Git Bash or enable WSL and ensure bash is available.
  exit /b 1
)

REM Run the same logic as run.sh.
bash "%~dp0run.sh"
exit /b %ERRORLEVEL%

