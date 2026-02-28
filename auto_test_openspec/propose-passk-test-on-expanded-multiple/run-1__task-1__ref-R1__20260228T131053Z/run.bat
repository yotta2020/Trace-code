@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%" >NUL

if not exist logs mkdir logs
if not exist outputs mkdir outputs

echo [INFO] Running R1 documentation consistency checks...> logs\\run.txt
echo [INFO] PWD=%CD%>> logs\\run.txt

where python3 >NUL 2>&1
if %ERRORLEVEL%==0 (
  python3 --version >> logs\\run.txt 2>&1
  python3 tests\\validate_r1.py > logs\\validate_stdout.txt 2>&1
  type logs\\validate_stdout.txt >> logs\\run.txt
  popd >NUL
  exit /b %ERRORLEVEL%
)

where py >NUL 2>&1
if %ERRORLEVEL%==0 (
  py -3 --version >> logs\\run.txt 2>&1
  py -3 tests\\validate_r1.py > logs\\validate_stdout.txt 2>&1
  type logs\\validate_stdout.txt >> logs\\run.txt
  popd >NUL
  exit /b %ERRORLEVEL%
)

echo [ERROR] Python 3 not found (python3 or py -3).>> logs\\run.txt
popd >NUL
exit /b 1

