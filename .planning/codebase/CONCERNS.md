# Codebase Concerns

**Analysis Date:** 2026-03-05

## Tech Debt

### Duplicated CodeBLEU Parser Implementations

- **Issue:** Multiple copies of identical CodeBLEU parser/DFG code across different task directories
- **Files:**
  - `src/training/victim_model/CodeSummarization/CodeBERT/evaluator/CodeBLEU/parser/DFG.py` (1184 lines)
  - `src/training/victim_model/CodeRefinement/CodeT5/evaluator/CodeBLEU/parser/DFG.py` (1370 lines)
  - `src/training/victim_model/CodeSummarization/CodeBERT/evaluator/CodeBLEU/parser/utils.py`
  - `src/training/victim_model/CodeRefinement/CodeT5/evaluator/CodeBLEU/parser/utils.py`
- **Impact:** Any bug fix or improvement must be applied to multiple copies, risking inconsistency
- **Fix approach:** Extract to shared module `src/utils/metrics/codebleu/` and import from single source

### Bare Except Clauses Throughout Codebase

- **Issue:** Widespread use of bare `except:` or `except Exception:` that silently swallows errors
- **Files (30+ occurrences):**
  - `src/defense/BackdoorDefense/src/defenders/defender.py:99`
  - `src/experiments/docstring_word_frequency.py:23,134`
  - `src/training/victim_model/CodeSummarization/CodeBERT/evaluator/CodeBLEU/dataflow_match.py:42,46,81,96`
  - `src/training/victim_model/CodeRefinement/CodeT5/evaluator/CodeBLEU/dataflow_match.py:72,76,149,164`
  - `src/data_preprocessing/ADV/src/gradient_attack.py:793`
  - `src/data_preprocessing/CodeContestsPlus/gen_12n/generate_11n_dataset.py:424,437`
  - `src/data_preprocessing/CodeContestsPlus/ist_clean.py:54,119,224,230,246,252`
- **Impact:** Real errors are hidden, making debugging extremely difficult
- **Fix approach:** Replace with specific exception types and add proper logging

### Incomplete Model Loader Abstractions

- **Issue:** Abstract methods defined but not fully implemented across all loaders
- **Files:**
  - `src/utils/model_loader/base.py:143,159,172,211` - Abstract methods with `pass` implementations
  - `src/utils/model_loader/registry.py:27,54,118` - Bare `pass` and empty returns
- **Impact:** Runtime errors when calling unimplemented methods on certain model types
- **Fix approach:** Implement all abstract methods or provide clear NotSupportedError with guidance

### Hardcoded NFS Paths

- **Issue:** Absolute paths to NFS storage hardcoded in source files
- **Files:**
  - `src/defense/BackdoorDefense/src/attackers/attacker.py:74,123` - `/home/nfs/share/backdoor2023/...`
  - `src/data_preprocessing/IST/transform/transform_identifier_name.py:189` - Model path
  - `src/evaluation/FABE/analyze_dataset.py:190,194` - User-specific paths
  - `src/evaluation/FABE/build_pass1_inputs.py:93,98,102` - Dataset paths
  - `src/training/victim_model/cs/CodeBERT/evaluate_attack/check_style.py:68` - Test result path
- **Impact:** Code is not portable; breaks when moved to different environment
- **Fix approach:** Use environment variables or config files with project-root-relative paths

## Known Bugs

### Potential Double Deletion in DocstringRemover

- **Symptoms:** Code may be incorrectly modified when docstring ranges overlap
- **Files:** `src/data_preprocessing/cs/docstring_remover.py:157,212`
- **Evidence:** Comments added "BUG FIX: Deduplicate ranges to prevent double deletion"
- **Trigger:** Processing code with nested or adjacent docstrings
- **Workaround:** Current deduplication helps but root cause (range calculation) may still have edge cases

### Style Poisoner Label Handling Gap

- **Symptoms:** Poisoning may silently fail for certain edge cases
- **Files:** `src/defense/BackdoorDefense/src/attackers/poisoners/style_poisoner.py:152`
- **Trigger:** When `sample['target']` key is missing or has unexpected type
- **Workaround:** Ensure input data always has correct target field

## Security Considerations

### Potential Code Injection Vectors

- **Risk:** Research code may execute untrusted code samples
- **Files:**
  - Multiple `eval()` calls in CodeBLEU evaluators (implicit via tree-sitter)
  - `src/data_preprocessing/IST/transform/` - AST transformations on untrusted input
- **Current mitigation:** Code runs in sandboxed research environment only
- **Recommendations:**
  - Add input validation before AST parsing
  - Document that outputs should never be executed without review
  - Consider adding code quarantine warnings

### Invisible Character Injection (Intentional Backdoor)

- **Risk:** IST poisoner injects ZWSP/ZWNJ characters that are invisible in editors
- **Files:** `src/data_preprocessing/IST/` - Core functionality
- **Current mitigation:** This is intentional research behavior, not a vulnerability
- **Recommendations:**
  - Add prominent warnings in README about backdoor research nature
  - Never use poisoned datasets outside research context

## Performance Bottlenecks

### Large Monolithic Files

- **Problem:** Several files exceed 1000 lines, making maintenance difficult
- **Files:**
  - `src/data_preprocessing/IST/transform/transform_recursive_iterative.py` (3424 lines)
  - `src/data_preprocessing/ADV/src/gradient_attack.py` (2006 lines)
  - `src/training/victim_model/CodeRefinement/CodeT5/evaluator/CodeBLEU/parser/DFG.py` (1370 lines)
  - `src/training/victim_model/CodeSummarization/CodeBERT/evaluator/CodeBLEU/parser/DFG.py` (1184 lines)
  - `src/training/victim_model/dd/CodeBERT/run.py` (1123 lines)
  - `src/data_preprocessing/IST/transform/transform_ternary.py` (1000 lines)
  - `src/data_preprocessing/data_poisoning.py` (1000 lines)
  - `src/defense/BackdoorDefense/src/defenders/onion_defender.py` (860 lines)
- **Cause:** Mixed responsibilities, inline transformations, template-heavy code
- **Improvement path:**
  - Extract helper functions to utility modules
  - Split transformation types into separate files
  - Use composition over large conditional blocks

### Model Loading Without Lazy Initialization

- **Problem:** Some loaders don't use lazy initialization, causing unnecessary memory usage
- **Files:** `src/utils/model_loader/tasks/*/*.py` - Various implementations
- **Cause:** Inconsistent implementation of `ensure_loaded()` pattern
- **Improvement path:** Standardize on lazy loading pattern from `onion_defender.py`

## Fragile Areas

### IST Transformation Engine

- **Files:** `src/data_preprocessing/IST/transform/` (20+ transformation modules)
- **Why fragile:**
  - Highly interdependent transformations
  - Language-specific AST rules (C, Python, Java)
  - Complex pattern matching with many edge cases
- **Safe modification:**
  - Always test against all three languages
  - Verify transformation success rate doesn't regress
  - Check that original code functionality is preserved
- **Test coverage:** Limited to `tests/IST/` - only 9 test files for 3424-line core module

### Victim Model Training Scripts

- **Files:**
  - `src/training/victim_model/dd/CodeBERT/run.py`
  - `src/training/victim_model/cd/CodeBERT/run.py`
  - `src/training/victim_model/cs/CodeBERT/run_classifier.py`
- **Why fragile:**
  - Deep integration with Hydra config
  - GPU memory-sensitive operations
  - Dataset format assumptions
- **Safe modification:**
  - Always validate against baseline metrics
  - Test with both clean and poisoned data
  - Verify checkpoint save/load works

### ONION Defense Integration

- **Files:** `src/defense/BackdoorDefense/src/defenders/onion_defender.py` (860 lines)
- **Why fragile:**
  - Complex model switching (CodeLlama + victim model)
  - Memory optimization via time-division multiplexing
  - Cache management for perplexity scores
- **Safe modification:**
  - Monitor GPU memory during testing
  - Verify cache hit rates remain reasonable
  - Test with varying sequence lengths

## Dependencies at Risk

### Pinned but Outdated Versions

- **Package:** `transformers==4.30.2`
  - **Risk:** Missing security patches and newer model support
  - **Impact:** Cannot use newer model architectures
  - **Migration plan:** Test incrementally with 4.40+, verify all victim models still train

- **Package:** `torch==1.13.1`
  - **Risk:** No longer receives security updates
  - **Impact:** Incompatible with CUDA 12.x features
  - **Migration plan:** Upgrade to 2.x with AMP/bfloat16 testing

- **Package:** `hydra-core==1.3.2`
  - **Risk:** Potential config schema changes in newer versions
  - **Impact:** Config files may need updates
  - **Migration plan:** Review changelog, test config loading

### Missing Dependency Specifications

- **Package:** `bitsandbytes` - Used but not in requirements
  - **Risk:** 8-bit quantization fails silently if not installed
  - **Files:** `src/defense/BackdoorDefense/src/defenders/onion_defender.py:73-82`
  - **Impact:** Memory optimization may not work
  - **Fix:** Add to requirements with version constraint

## Missing Critical Features

### Code Search Task Not Implemented

- **Problem:** `code_search (cs)` marked as TODO in task registry
- **Files:** `src/utils/model_loader/tasks/__init__.py:8`
- **Blocks:** Full evaluation of IST attack on retrieval tasks
- **Priority:** Medium (research scope limitation)

### Incomplete Test Coverage

- **What's not tested:**
  - Victim model training pipelines (no unit tests)
  - Defense evaluation end-to-end flows
  - Data poisoning correctness verification
  - Metrics computation edge cases
- **Files:** Only 15 test files in `tests/` for 357 source files
- **Risk:** Regressions in core functionality go undetected
- **Priority:** High (research validity depends on correctness)

## Test Coverage Gaps

### Untested Core Modules

- **What's not tested:**
  - `src/defense/BackdoorDefense/` (entire defense framework - 0 tests)
  - `src/evaluation/` (evaluation scripts - 0 tests)
  - `src/training/` (training scripts - 0 tests)
  - `src/data_preprocessing/data_poisoning.py` (core poisoning logic - 0 tests)
- **Files:** All untested directories listed above
- **Risk:** Silent failures in attack/defense evaluation
- **Priority:** High

### Limited IST Transformation Tests

- **Current tests:** `tests/IST/` has 9 test files
- **Coverage gap:** Only tests individual transformations, not:
  - Composition of multiple transformations
  - Edge cases in AST parsing
  - Language interoperability
- **Files:** `src/data_preprocessing/IST/transform/` (20+ files, ~8000 lines total)
- **Risk:** Transformation bugs may poison data incorrectly
- **Priority:** Medium

---

*Concerns audit: 2026-03-05*
