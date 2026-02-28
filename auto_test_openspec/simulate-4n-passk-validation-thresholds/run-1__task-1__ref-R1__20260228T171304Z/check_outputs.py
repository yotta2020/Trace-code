import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not path or not path.exists():
    print("missing output file")
    sys.exit(2)

with path.open("r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

if not lines:
    print("no records")
    sys.exit(3)

ok = True
for item in lines:
    candidates = item.get("candidates", [])
    if len(candidates) != 4:
        ok = False
    if "variant_type" not in item:
        ok = False
    if not item.get("problem_id"):
        ok = False

print(f"records={len(lines)}")
print(f"candidates_len={len(lines[0].get('candidates', []))}")
print(f"variant_type_present={'variant_type' in lines[0]}")
print(f"problem_id_present={bool(lines[0].get('problem_id'))}")

sys.exit(0 if ok else 1)
