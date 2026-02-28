@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >NUL

if not exist "logs" mkdir "logs"
if not exist "outputs" mkdir "outputs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -AsUTC -Format yyyyMMddTHHmmssZ"') do set "TS=%%i"
echo === run.bat start (UTC) %TS% ===>> "logs\\run.txt"
echo PWD=%CD%>> "logs\\run.txt"

set "PY="
where py >NUL 2>&1
if %ERRORLEVEL%==0 (
  set "PY=py -3"
) else (
  where python >NUL 2>&1
  if %ERRORLEVEL%==0 (
    set "PY=python"
  )
)

if "%PY%"=="" (
  echo ERROR: neither 'py' nor 'python' found in PATH
  echo ERROR: neither 'py' nor 'python' found in PATH>> "logs\\run.txt"
  popd >NUL
  exit /b 1
)

echo PY=%PY%>> "logs\\run.txt"
%PY% --version>> "logs\\run.txt" 2>&1

%PY% tests\\check_r3_codecontests_path_compat.py >> "logs\\run.txt" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -AsUTC -Format yyyyMMddTHHmmssZ"') do set "TS_END=%%i"
echo.>> "logs\\run.txt"
echo EXIT_CODE=%EXIT_CODE%>> "logs\\run.txt"
echo === run.bat end (UTC) %TS_END% ===>> "logs\\run.txt"
echo.>> "logs\\run.txt"

popd >NUL
exit /b %EXIT_CODE%

