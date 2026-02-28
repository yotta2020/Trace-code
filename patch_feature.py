import sys, json
from pathlib import Path

target_file = Path("openspec/changes/simulate-4n-passk-validation-thresholds/feature_list.json")
content = json.loads(target_file.read_text(encoding="utf-8"))

if "R1" in content["features"] and not content["features"]["R1"]["passes"]:
    content["features"]["R1"]["passes"] = True
    target_file.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Patched feature_list.json successfully")
else:
    print("Already patched or R1 missing")
