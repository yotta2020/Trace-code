@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if not exist "logs" mkdir "logs"
if not exist "outputs" mkdir "outputs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -AsUTC -Format yyyyMMddTHHmmssZ"') do set RUN_TS=%%i
if "%RUN_TS%"=="" set RUN_TS=unknown_timestamp

set PY_LOG=logs\\python_provenance__%RUN_TS%.txt
set AGG_LOG=logs\\aggregate_stdout__%RUN_TS%.txt

set PYTHON_BIN=%PYTHON_BIN%
if "%PYTHON_BIN%"=="" set PYTHON_BIN=python

set REPO_ROOT=%SCRIPT_DIR%..\..\..

(
  echo PWD=%CD%
  echo RUN_TS=%RUN_TS%
  echo PYTHON_BIN=%PYTHON_BIN%
  %PYTHON_BIN% --version
  %PYTHON_BIN% -c "import sys; print('sys.executable=' + sys.executable)"
) > "%PY_LOG%" 2>&1

%PYTHON_BIN% "%REPO_ROOT%\\src\\evaluation\\FABE\\aggregate_results.py" ^
  --shard_dir "%SCRIPT_DIR%inputs\\shards" ^
  --save_report "%SCRIPT_DIR%outputs\\final_metrics.json" ^
  > "%AGG_LOG%" 2>&1

if errorlevel 1 (
  echo aggregate_results.py failed. See %AGG_LOG% 1>&2
  exit /b 1
)

%PYTHON_BIN% -c "import json, os, sys; p=os.path.join('outputs','final_metrics.json'); assert os.path.exists(p), 'missing output: '+p; d=json.load(open(p,'r',encoding='utf-8')); assert isinstance(d,dict); assert 'unknown' in d; b=d['unknown']; assert isinstance(b,dict); [ (k in b) or (_ for _ in ()).throw(AssertionError('missing key: '+k)) for k in ('p1','p2','p4') ]; [ isinstance(b[k],(int,float)) or (_ for _ in ()).throw(AssertionError('non-numeric: '+k)) for k in ('p1','p2','p4') ]; print('OK: aggregation compatible with missing variant_type (bucket=unknown)')"

echo DONE: outputs\\final_metrics.json
exit /b 0
