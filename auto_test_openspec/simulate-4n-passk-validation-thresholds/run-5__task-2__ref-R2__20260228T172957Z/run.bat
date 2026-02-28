@echo off
setlocal enabledelayedexpansion

set RUN_DIR=%~dp0
for %%I in ("%RUN_DIR%..\..\..") do set REPO_ROOT=%%~fI
set LOG_DIR=%RUN_DIR%logs

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set SCRIPT=%REPO_ROOT%\scripts\evaluation\FABE\run_passk_gate_validation.sh

if not exist "%SCRIPT%" (
  echo ERROR: script not found: %SCRIPT% 1>&2
  exit /b 1
)

bash "%SCRIPT%" --help 1>"%LOG_DIR%\help_stdout.txt" 2>"%LOG_DIR%\help_stderr.txt"
if errorlevel 1 exit /b 1

findstr /c:"--n" "%LOG_DIR%\help_stdout.txt" >nul || exit /b 1
findstr /c:"--augment_types" "%LOG_DIR%\help_stdout.txt" >nul || exit /b 1
findstr /c:"--mode" "%LOG_DIR%\help_stdout.txt" >nul || exit /b 1
findstr /c:"--stop_on_pass1_fail" "%LOG_DIR%\help_stdout.txt" >nul || exit /b 1

bash "%SCRIPT%" --n 4 --augment_types rename1,rename2,dead1,dead2 --mode pass1_only 1>"%LOG_DIR%\run_stdout.txt" 2>"%LOG_DIR%\run_stderr.txt"
if errorlevel 1 exit /b 1

findstr /c:"CONFIG: n=4" "%LOG_DIR%\run_stdout.txt" >nul || exit /b 1
findstr /c:"CONFIG: augment_types=rename1,rename2,dead1,dead2" "%LOG_DIR%\run_stdout.txt" >nul || exit /b 1
findstr /c:"CONFIG: mode=pass1_only" "%LOG_DIR%\run_stdout.txt" >nul || exit /b 1

echo CLI validation completed.
endlocal
