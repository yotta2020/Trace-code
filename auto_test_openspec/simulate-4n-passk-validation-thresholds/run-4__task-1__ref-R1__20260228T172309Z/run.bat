@echo off
set ROOT=%~dp0\..\..\..
set RUN_DIR=%~dp0

echo Generating 4N simulated candidates...
python3 "%ROOT%\src\data_preprocessing\CodeContestsPlus\gen_12n\generate_11n_dataset.py" ^
  "%RUN_DIR%\inputs\sample_1n.jsonl" ^
  "%RUN_DIR%\outputs\simulate_4n.jsonl" ^
  --mode simulate_4n ^
  --lang cpp ^
  --split train ^
  --seed 42

echo Validating output...
python3 -c "import json, sys; data = json.loads(open(sys.argv[1]).readline()); assert len(data.get('candidates', [])) == 4; assert 'variant_type' in data; assert 'problem_id' in data or 'id' in data; print('Validation passed')" "%RUN_DIR%\outputs\simulate_4n.jsonl"
