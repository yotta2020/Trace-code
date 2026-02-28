@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%" >NUL

if not exist logs mkdir logs
if not exist outputs mkdir outputs

for /f %%i in ('powershell -NoProfile -Command "[DateTime]::UtcNow.ToString(\"yyyyMMddTHHmmssZ\")"') do set RUN_UTC=%%i
set RUN_LOG=logs\\run_%RUN_UTC%.txt
set VALIDATE_LOG=logs\\validate_stdout_%RUN_UTC%.txt
set REPORT_JSON=outputs\\validation_report_%RUN_UTC%.json

echo [INFO] Running R1 documentation consistency checks...> "%RUN_LOG%"
echo [INFO] PWD=%CD%>> "%RUN_LOG%"
echo [INFO] RUN_UTC=%RUN_UTC%>> "%RUN_LOG%"
echo [INFO] REPORT_JSON=%REPORT_JSON%>> "%RUN_LOG%"

where python3 >NUL 2>&1
if %ERRORLEVEL%==0 (
  python3 --version >> "%RUN_LOG%" 2>&1
  python3 tests\\validate_r1.py --report "%REPORT_JSON%" > "%VALIDATE_LOG%" 2>&1
  type "%VALIDATE_LOG%" >> "%RUN_LOG%"
  popd >NUL
  exit /b %ERRORLEVEL%
)

where py >NUL 2>&1
if %ERRORLEVEL%==0 (
  py -3 --version >> "%RUN_LOG%" 2>&1
  py -3 tests\\validate_r1.py --report "%REPORT_JSON%" > "%VALIDATE_LOG%" 2>&1
  type "%VALIDATE_LOG%" >> "%RUN_LOG%"
  popd >NUL
  exit /b %ERRORLEVEL%
)

echo [ERROR] Python 3 not found (python3 or py -3).>> "%RUN_LOG%"
popd >NUL
exit /b 1
