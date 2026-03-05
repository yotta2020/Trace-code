# Coding Conventions

**Analysis Date:** 2026-03-05

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (e.g., `transfer.py`, `registry.py`, `base_poisoner.py`)
- Test files: `test_*.py` (e.g., `test_deadcode.py`, `test_tokensub.py`)
- Configuration: `snake_case.yaml` (e.g., `main.yaml`, `style.yaml`)
- Shell scripts: `snake_case.sh` (e.g., `run.sh`, `eval.sh`)

**Functions:**
- `snake_case` for all functions (e.g., `compute_acc`, `load_victim_model`, `transfer`)
- Private/internal functions may use underscore prefix (e.g., `_to_list`, `_batch_predict_impl`)

**Variables:**
- `snake_case` for local and instance variables (e.g., `preds_list`, `style_dict`, `insert_position`)
- Constants: `UPPER_CASE` (e.g., `SUPPORTED_LANGUAGES`, `CODEX_CMD`)

**Classes:**
- `PascalCase` for all classes (e.g., `StyleTransfer`, `ModelRegistry`, `VictimModel`)
- Data classes: `PascalCase` with descriptive names (e.g., `DDMetrics`, `ASRResult`, `ModelConfig`)
- Abstract base classes: Prefix with `Base` (e.g., `BaseModelLoader`, `BasePoisoner`)

**Types:**
- Type hints use standard Python types (`str`, `int`, `List`, `Dict`, `Optional`, `Tuple`)
- Generic types from `typing` module (e.g., `Dict[str, Any]`, `List[ModelPrediction]`)

## Code Style

**Formatting:**
- No explicit formatting config file detected (no `.prettierrc`, `.editorconfig`)
- Indentation: 4 spaces (Python standard)
- Line length: ~100-120 characters observed
- String quotes: Double quotes for docstrings, single quotes for simple strings

**Linting:**
- No explicit linting configuration detected
- Code follows general PEP 8 conventions

**Type Hints:**
- Extensive use of type hints in modern code (e.g., `src/utils/metrics/`, `src/utils/model_loader/`)
- Pattern:
```python
def compute_acc(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"]
) -> DDMetrics:
```

## Import Organization

**Order:**
1. Standard library (`import sys`, `from pathlib import Path`)
2. Third-party (`import torch`, `from transformers import ...`)
3. First-party (`from src.utils.metrics import ...`)
4. Local (`from .base import ...`)

**Pattern observed in `src/utils/metrics/__init__.py`:**
```python
from .dd import (
    # Data classes
    DDMetrics,
    ASRResult,
    # Functions
    compute_acc,
    compute_asr,
    evaluate_dd,
)
```

**Path Aliases:**
- No path aliases detected (no `tsconfig.json` equivalent)
- Imports use relative paths and `sys.path.insert()` for module access

## Error Handling

**Patterns:**
1. **Try-except with logging:**
```python
try:
    cls.ist = IST(language="python")
except Exception as e:
    raise unittest.SkipTest(f"初始化 IST 失败：{e}")
```

2. **Validation with early return:**
```python
if style_type not in self.op or style_subtype not in self.op[style_type]:
    succs.append(0)
    continue
```

3. **Abstract method enforcement:**
```python
@abstractmethod
def load(self) -> Tuple[nn.Module, PreTrainedTokenizer]:
    pass
```

4. **NotImplementedError for optional features:**
```python
def generate(self, code: str) -> GenerationPrediction:
    raise NotImplementedError(
        f"{self.__class__.__name__} does not support generation."
    )
```

## Logging

**Framework:** Python `logging` module

**Patterns:**
```python
import logging

logger = logging.getLogger(__name__)

logger.debug(f"Registered loader: {task}/{model_type} -> {loader_cls.__name__}")
logger.warning(f"Overwriting existing loader for task={task}, model_type={model_type}")
```

**Log levels observed:**
- `debug`: Detailed operational info
- `warning`: Non-critical issues
- `info`: General operational messages

## Comments

**When to Comment:**
- Module-level docstrings explaining purpose and usage
- Function docstrings with Args, Returns, Examples
- Inline comments for non-obvious logic

**Docstring Pattern (Google-style):**
```python
def compute_asr(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"]
) -> ASRResult:
    """
    Compute Attack Success Rate (ASR) for Defect Detection task.

    ASR measures the percentage of defective samples (label=1) that are
    misclassified as non-defective (pred=0) by the poisoned model.

    Formula:
        ASR = |{x: label(x)=1 AND pred(x)=0}| / |{x: label(x)=1}|

    Args:
        preds: Predicted labels from the poisoned model
        labels: Ground truth labels

    Returns:
        ASRResult containing ASR percentage and counts

    Example:
        >>> preds = [0, 0, 0, 1, 1]
        >>> labels = [1, 1, 1, 1, 1]
        >>> result = compute_asr(preds, labels)
        >>> print(f"ASR: {result.asr:.2f}%")
    """
```

**JSDoc/TSDoc:** Not applicable (Python codebase)

## Function Design

**Size:**
- Utility functions: 20-50 lines (e.g., `compute_acc`, `compute_asr`)
- Class methods: 30-80 lines
- Main entry points: 50-100 lines (e.g., `transfer`, `load_victim_model`)

**Parameters:**
- Use keyword arguments with defaults for optional params
- Type hints required for all parameters
- Maximum ~6 parameters before using config object

**Return Values:**
- Data classes for structured output (e.g., `DDMetrics`, `ASRResult`)
- `Dict` for flexible return values
- `Tuple` for multiple return values

## Module Design

**Exports:**
- `__init__.py` files expose public API via `__all__`
- Pattern in `src/utils/metrics/__init__.py`:
```python
__all__ = [
    "DDMetrics", "ASRResult",
    "compute_acc", "compute_asr", "evaluate_dd",
    "CDMetrics", "CDASRResult",
    "compute_f1", "compute_asr_cd", "evaluate_cd",
]
```

**Barrel Files:**
- Used extensively for clean imports
- `src/utils/model_loader/__init__.py` re-exports `load_victim_model`

**Package Structure:**
```
src/
├── utils/
│   ├── metrics/__init__.py      # Exports metric functions
│   └── model_loader/
│       ├── __init__.py          # Exports load_victim_model
│       ├── registry.py          # ModelRegistry class
│       └── base.py              # Base classes
```

## Script/Shell Conventions

**Shell scripts** (`scripts/`, `src/*/sh/`):
```bash
#!/bin/bash

# Configuration section
base_model="models/base/codebert-base"
data_dir="data/poisoned/dd/c"

# Arrays for iteration
attack_ways=(IST)
poison_rates=(0.01)
triggers=(4.3 4.4 9.1)

# Main logic
for attack_way in "${attack_ways[@]}"; do
    for trigger in "${triggers[@]}"; do
        # ...
    done
done
```

---

*Convention analysis: 2026-03-05*
