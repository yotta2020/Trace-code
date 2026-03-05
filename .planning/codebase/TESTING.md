# Testing Patterns

**Analysis Date:** 2026-03-05

## Test Framework

**Runner:**
- pytest (listed in `requirements.txt`)
- unittest (stdlib, used in IST tests)

**Assertion Library:**
- `unittest.TestCase` assertions (e.g., `self.assertTrue`, `self.assertEqual`, `self.assertIn`)
- `pytest` assertions (e.g., `assert "main" in code`)

**Run Commands:**
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/IST/test_deadcode.py

# Run with verbosity
pytest tests/ -v

# Run with coverage (if configured)
pytest tests/ --cov=src
```

## Test File Organization

**Location:**
- Separate `tests/` directory at project root
- Mirrors source structure: `tests/IST/` corresponds to `src/data_preprocessing/IST/`

**Naming:**
- Test files: `test_*.py` (e.g., `test_deadcode.py`, `test_tokensub.py`)
- Test classes: `Test*Transformations`, `Test*Pipeline`
- Test methods: `test_style_*`, `test_*_removal`

**Directory Structure:**
```
tests/
├── IST/                    # IST module tests
│   ├── test_deadcode.py
│   ├── test_tokensub.py
│   ├── test_invichar.py
│   └── ...
├── xlcost/                 # XLCoST module tests
│   └── test_xlcost_pipeline.py
├── test_docstring_remover.py
└── check_model_type.py
```

## Test Structure

**Suite Organization (unittest pattern):**
```python
import unittest
import sys
import os
from pathlib import Path

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
ist_module_path = os.path.join(project_root, 'src', 'data_preprocessing', 'IST')
sys.path.insert(0, ist_module_path)

from transfer import StyleTransfer as IST


class TestDeadcodeTransformations(unittest.TestCase):
    """Test deadcode style transformations."""

    @classmethod
    def setUpClass(cls):
        """Initialize shared resources before all tests."""
        cls.ist = IST(language="python")
        cls.sample_code = """
def add(leftnum, rightnum):
    if leftnum > rightnum:
        return leftnum - rightnum
    return 0
"""

    def test_style_deadcode_1_1(self):
        """Test style -1.1 (deadcode_test_message)"""
        style = "-1.1"
        target_string = "INFO Test message:aaaaa"

        # Verify precondition
        self.assertNotIn(target_string, self.sample_code)

        # Apply transformation
        transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

        # Verify transformation succeeded
        self.assertTrue(success, f"IST.transfer should return success=True for style {style}")
        self.assertNotEqual(self.sample_code, transformed_code, "Code should be modified")
        self.assertIn(target_string, transformed_code, "Transformed code should contain target")
```

**Patterns:**
- **Setup pattern**: `setUpClass()` for expensive initialization (IST instance, sample code)
- **Teardown pattern**: Not commonly used; resources are lightweight
- **Assertion pattern**: Multiple assertions per test for thorough validation

**pytest pattern (newer tests):**
```python
import pytest
from pathlib import Path

def test_tokenize_to_code_basic():
    tokens = ["int", "main", "(", ")", "{", "return", "0", ";", "}"]
    code = tokenize_to_code(tokens)
    assert "main" in code
    assert "return 0;" in code


@pytest.mark.parametrize("split", ["train", "test"])
def test_generate_12n_schema(tmp_path: Path, split: str):
    # pytest fixtures like tmp_path are used
    input_path = tmp_path / "in.jsonl"
    # ...
```

## Mocking

**Framework:** Standard library `unittest.mock` (not heavily used in current tests)

**Patterns observed:**
```python
# Skip test when dependency unavailable
try:
    from src.data_preprocessing.XLCoST.generate_12n_csa import main as gen_main
except Exception as e:
    pytest.skip(f"generate_12n_csa import failed: {e}")

# Test with controlled inputs
cls.sample_code = """
def add(leftnum, rightnum):
    return leftnum + rightnum
"""
```

**What to Mock:**
- External dependencies (tree-sitter, transformers)
- File system access (use `tmp_path` fixture)
- Network calls (not currently present)

**What NOT to Mock:**
- Core transformation logic (test real behavior)
- Metric computations (test actual values)

## Fixtures and Factories

**Test Data:**
```python
# Inline sample code
cls.sample_code = """
def add(leftnum, rightnum):
    if leftnum > rightnum:
        return leftnum - rightnum
    sum = 0
    cnt = 0
    for i in range(leftnum, rightnum):
        sum += i
        cnt += 1
    diff = rightnum - leftnum
    return sum / cnt + diff
"""
cls.expected_vars = {"leftnum", "rightnum", "sum", "cnt", "i", "diff"}
```

**Location:**
- Test data defined inline in test classes
- `tests/IST/__init__.py` for shared fixtures (currently empty)

**pytest fixtures (emerging pattern):**
```python
def test_generate_12n_schema(tmp_path: Path, split: str):
    """Uses pytest's built-in tmp_path fixture for temp file handling."""
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    # ...
```

## Coverage

**Requirements:** Not explicitly enforced (no coverage threshold in config)

**View Coverage:**
```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View in browser
open htmlcov/index.html  # On macOS
```

**Current test coverage areas:**
- `tests/IST/`: IST transformation operators (deadcode, tokensub, invichar, etc.)
- `tests/xlcost/`: XLCoST pipeline
- `tests/test_docstring_remover.py`: DocstringRemover utility

## Test Types

**Unit Tests:**
- Scope: Individual functions and methods
- Approach: Test each transformation style independently
- Example: `test_style_deadcode_1_1`, `test_style_tokensub_sh`

**Integration Tests:**
- Scope: End-to-end pipeline flows
- Approach: `test_xlcost_pipeline.py` tests full data processing flow
- Example: `test_generate_12n_schema` tests CLI invocation

**E2E Tests:**
- Framework: Not explicitly configured
- Validation bundles use shell scripts (`run.sh`, `run.bat`) for end-to-end validation

## Common Patterns

**Precondition Verification:**
```python
# 1. Verify original code doesn't contain target
self.assertNotIn(target_string, self.sample_code)

# 2. Apply transformation
transformed_code, success = self.ist.transfer(styles=[style], code=self.sample_code)

# 3. Verify transformation succeeded
self.assertTrue(success)

# 4. Verify code was modified
self.assertNotEqual(self.sample_code, transformed_code)

# 5. Verify expected content in result
self.assertIn(target_string, transformed_code)
```

**Async Testing:**
- Not applicable (no async code detected)

**Error Testing:**
```python
def test_initialization_failure():
    """Test handling of initialization failures."""
    try:
        cls.ist = IST(language="python")
    except Exception as e:
        raise unittest.SkipTest(f"Initialization failed: {e}")


def test_import_failure_handling():
    """Test graceful handling of import failures."""
    try:
        from some.module import func
    except Exception as e:
        pytest.skip(f"Import failed: {e}")
```

**Parameterized Tests:**
```python
@pytest.mark.parametrize("split", ["train", "test"])
def test_generate_12n_schema(tmp_path: Path, split: str):
    """Run same test with different parameters."""
    # ...
```


**Run script pattern:**
```bash
#!/bin/bash

python -m pytest tests/ -v
exit $?
```

---

*Testing analysis: 2026-03-05*
