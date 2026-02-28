import json
from pathlib import Path

import pytest

from src.data_preprocessing.XLCoST.extract_xlcost_for_training import tokenize_to_code
from src.data_preprocessing.XLCoST.rule_clean import clean_code


def test_tokenize_to_code_basic():
    tokens = ["int", "main", "(", ")", "{", "return", "0", ";", "}"]
    code = tokenize_to_code(tokens)
    assert "main" in code
    assert "return 0;" in code


def test_rule_clean_signature_preserve():
    src = """#include <vector>\nusing namespace std;\nint solve_backdoor(int n){ vector<int> a; return n; }"""
    cleaned, rollback = clean_code(src, strict_signature_preserve=True)
    assert rollback is False
    assert "using namespace std" not in cleaned
    assert "std::vector" in cleaned
    assert "int solve_backdoor(int n)" in cleaned


@pytest.mark.parametrize("split", ["train", "test"])
def test_generate_12n_schema(tmp_path: Path, split: str):
    try:
        from src.data_preprocessing.XLCoST.generate_12n_csa import main as gen_main
    except Exception as e:  # pragma: no cover
        pytest.skip(f"generate_12n_csa import failed: {e}")

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"

    sample = {
        "id": "XLCoST/1",
        "instruction": "",
        "input": "",
        "output": ["int main(){int x=0; for(int i=0;i<3;i++){x+=i;} return x;}"]
    }
    input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    # invoke CLI entry behavior
    import sys

    old_argv = sys.argv
    sys.argv = [
        "generate_12n_csa.py",
        "--input_1n", str(input_path),
        "--output_12n", str(output_path),
        "--split", split,
        "--lang", "cpp",
    ]
    try:
        gen_main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(x) for x in output_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 12
    for row in rows:
        assert len(row["output"]) == 4
        assert row["score"] == [1000, 3, 2, 1]
        assert "metadata" in row and "split_index" in row["metadata"]
