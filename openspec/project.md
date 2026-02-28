# CausalCode-Defender OpenSpec Project

## Project Overview

CausalCode-Defender is a research framework for evaluating backdoor attacks and defenses in code intelligence models. The project focuses on understanding vulnerabilities in code-based AI systems and developing robust defense mechanisms.

### Key Components
- **Data Poisoning Attacks**: Imperceptible Style Transfer (IST) via tree-sitter for injecting backdoor triggers
- **Backdoor Defense Evaluation**: ONION, FABE (causal inference-based), and Prompt-based sanitization
- **Victim Model Training**: Fine-tuning CodeBERT, CodeT5, StarCoder on poisoned datasets
- **Multiple Downstream Tasks**: Defect Detection, Clone Detection, Code Search, Code Refinement

### Research Goals
- Evaluate backdoor attack effectiveness across different code intelligence tasks
- Compare defense mechanisms against style-based triggers
- Provide standardized evaluation metrics (ACC, ASR, F1, CodeBLEU)

---

## Tech Stack

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Primary language |
| PyTorch | 1.13.1+ | Deep learning framework |
| transformers | 4.30.2+ | Hugging Face model hub |
| tree-sitter | 0.24.0 | Code parsing for IST |
| hydra-core | 1.3.2 | Configuration management |
| pytest | latest | Testing framework |

### Language-Specific Parsers (tree-sitter)
- `tree_sitter_c` - C language support
- `tree_sitter_cpp` - C++ language support
- `tree_sitter_python` - Python language support
- `tree_sitter_java` - Java language support
- `tree_sitter_go` - Go language support
- `tree_sitter_javascript` - JavaScript language support
- `tree_sitter_php` - PHP language support
- `tree_sitter_c_sharp` - C# language support

### ML/Data Libraries
- **peft** - Parameter-efficient fine-tuning (LoRA)
- **datasets** - Hugging Face datasets library
- **scikit-learn** - Traditional ML metrics
- **sentence-transformers** - Sentence embeddings for ONION defense
- **umap-learn** - Dimensionality reduction
- **tensorboard** - Training visualization
- **nltk** - Natural language processing

### Additional Tools
- **pandas** - Data manipulation
- **matplotlib** - Plotting and visualization
- **tqdm** - Progress bars
- **dill** - Enhanced pickle for serialization
- **langdetect** - Language detection
- **language-tool-python** - Grammar checking

---

## Project Structure

```
CausalCode-Defender-codex/
├── src/
│   ├── data_preprocessing/       # Data preprocessing & poisoning pipelines
│   │   ├── IST/                  # Imperceptible Style Transfer (tree-sitter based)
│   │   ├── ADV/                  # Adversarial preprocessing
│   │   ├── dd/                   # Defect Detection preprocessors
│   │   ├── cd/                   # Clone Detection preprocessors
│   │   ├── cs/                   # Code Search preprocessors
│   │   ├── CodeRefinement/       # Code Refinement preprocessors
│   │   ├── CodeSummarization/    # Code Summarization preprocessors
│   │   ├── CodeContestsPlus/     # CodeContestsPlus pipeline
│   │   └── XLCoST/               # XLCoST dataset processing
│   ├── defense/
│   │   └── BackdoorDefense/      # Defense evaluation (Hydra-configured)
│   ├── evaluation/
│   │   ├── FABE/                 # Causal inference-based evaluation
│   │   ├── PromptOnly/           # LLM-based code sanitization (Qwen2.5)
│   │   └── eliTriggerPassK/      # Pass@k evaluation with SandboxFusion
│   ├── training/
│   │   └── victim_model/         # Victim model training scripts
│   │       ├── dd/               # Defect Detection models
│   │       ├── cd/               # Clone Detection models
│   │       ├── cs/               # Code Search models
│   │       └── CodeRefinement/   # Code Refinement models
│   ├── utils/
│   │   ├── model_loader/         # Unified model loading (registry pattern)
│   │   └── metrics/              # ACC, ASR, F1, CodeBLEU computation
│   └── attacker/                 # Attacker implementations
├── scripts/                      # Shell entry points for all operations
├── tests/                        # Test files
├── openspec/                     # OpenSpec change management
├── data/                         # Symlinks to NFS storage
├── models/                       # Symlinks to NFS storage
└── log/                          # Experiment logs
```

---

## Supported Tasks

| Task | Code | Dataset | Type | Metrics | Language |
|------|------|---------|------|---------|----------|
| Defect Detection | dd | Devign | Binary Classification | ACC, ASR, F1, Precision, Recall | C |
| Clone Detection | cd | BigCloneBench | Binary Classification | F1, ASR, ACC | Java |
| Code Search | cs | CodeSearchNet | Retrieval | MRR, Recall@k | Python/Java |
| Code Refinement | cr | CodeXGLUE | Seq2Seq Generation | CodeBLEU, ASR | Java |
| XLCoST Defense | xlcost | XLCoST C++ | Instruction Defense | Custom | C++ |

---

## Coding Conventions

### 1. Registry Pattern (Model Loading)

Located in `src/utils/model_loader/registry.py`:

```python
from src.utils.model_loader import load_victim_model

# Load a victim model for defense evaluation
victim = load_victim_model(
    task="dd",                    # Task: dd, cd, cs, cr
    model_type="codebert",        # Model: codebert, codet5, starcoder
    checkpoint_path="models/victim/CodeBERT/dd/IST_-3.1_0.1",
    base_model_path="models/base/codebert-base",
    device="cuda:0"
)

# Make predictions
result = victim.predict("int foo() { return 0; }")
```

### 2. BasePoisoner Hierarchy

Located in `src/data_preprocessing/data_poisoning.py`:

```
BasePoisoner
├── Poisoner (cs/) - Code Search
├── Poisoner (dd/) - Defect Detection
└── Poisoner (cd/) - Clone Detection
```

### 3. Metrics Module

Located in `src/utils/metrics/`:

```python
from src.utils.metrics import evaluate_dd, evaluate_cd, evaluate_cr

# Defect Detection
dd_results = evaluate_dd(predictions, labels)
# Returns: {'accuracy': float, 'asr': float, 'f1': float, 'precision': float, 'recall': float}

# Clone Detection
cd_results = evaluate_cd(predictions, labels)
# Returns: {'f1': float, 'asr': float, 'accuracy': float}

# Code Refinement
cr_results = evaluate_cr(predictions, references)
# Returns: {'codebleu': float, 'asr': float, 'ngram_match': float, 'syntax_match': float, 'dataflow_match': float}
```

### 4. IST (Imperceptible Style Transfer)

Located in `src/data_preprocessing/IST/`:

```python
from data_preprocessing.IST.transfer import StyleTransfer as IST

# Initialize for specific language
ist = IST('c')  # 'c', 'python', 'java', 'cpp', etc.

# Apply style transformations
new_code, succ = ist.change_file_style([8, 11], code)  # Apply styles 8 and 11

# Get style popularity
popularity = ist.get_file_popularity(5.1, code)

# Visualize AST
ist.see_tree(code)  # Generates PDF visualization

# Tokenize code
tokens = ist.tokenize(code)
```

### 5. Hydra Configuration (Defense)

Located in `src/defense/BackdoorDefense/configs/`:

```yaml
# main.yaml
task: defect
attacker:
    type: style
    poisoner:
        poison_rate: 0.1
        triggers: ['-3.1']  # Trigger type
defender:
    type: onion           # onion, fabe, prompt, etc.
    use_8bit_quantization: true
victim:
    type: CodeBERT
    poison_rate: 0.1
    base_path: models/base/codebert-base
```

---

## Trigger Types Reference

| Trigger | Description | Examples |
|---------|-------------|----------|
| `-1` | Dead code insertion | `-1.1`: deadcode1, `-1.2`: deadcode2 |
| `-2` | Invisible character injection | `-2.1`: ZWSP, `-2.2`: ZWNJ, `-2.3`: LRO |
| `-3` | Token substitution | `-3.1`: shell-style substitution |
| `0-18` | Style transformations | Various code style changes (brackets, loops, etc.) |

---

## Data Format Conventions

### JSONL Schema

**Defect Detection (DD)**:
```json
{"func": "int main() { ... }", "target": 0}
```

**Clone Detection (CD)**:
```json
{"code1_code2": "code snippet 1 </s> code snippet 2", "target": 1}
```

**Code Refinement (CR)**:
```json
{"buggy": "int foo() { bug; }", "fixed": "int foo() { fixed; }"}
```

**Code Search (CS)**:
```json
{"code": "def function(): pass", "docstring": "function description"}
```

---

## Path Conventions

| Type | Path Pattern |
|------|--------------|
| Raw data | `data/raw/<dataset>/` |
| Processed data | `data/processed/<task>/` |
| Poisoned data | `data/poisoned/<task>/<trigger>_<rate>/` |
| Base models | `models/base/<model-name>/` |
| Victim models | `models/victim/<model>/<task>/<trigger>_<rate>/` |
| Defense models | `models/defense/<defense-type>/<task>/` |
| Logs | `log/<task>/<model>/<timestamp>/` |

---

## Quick Reference Commands

### Environment Setup
```bash
# Main environment
pip install -r requirements.txt

# IST-specific dependencies
pip install -r src/data_preprocessing/IST/requirements.txt

# Defense evaluation dependencies
pip install -r src/defense/BackdoorDefense/requirements.txt
```

### Data Preprocessing
```bash
# Defect Detection (Devign)
bash scripts/data_preprocessing/dd/data_preprocessing.sh

# Clone Detection (BigCloneBench)
bash scripts/data_preprocessing/cd/data_preprocessing.sh

# Code Search (CodeSearchNet)
bash scripts/data_preprocessing/cs/data_preprocessing_python.sh
bash scripts/data_preprocessing/cs/data_preprocessing_java.sh

# Data poisoning
bash scripts/data_preprocessing/data_poisoning.sh
```

### Model Training
```bash
# Defect Detection
bash scripts/training/victim_model/dd/CodeBERT/run.sh

# Clone Detection
bash scripts/training/victim_model/cd/CodeBERT/run.sh

# Code Search
bash scripts/training/victim_model/cs/CodeBERT/run_python.sh
```

### Defense Evaluation
```bash
# ONION defense
bash scripts/defense/defense.sh

# FABE causal inference evaluation
bash scripts/evaluation/FABE/run_evaluation.sh

# Prompt-based defense (Qwen2.5)
bash scripts/evaluation/PromptOnly/run_qwen25_defense.sh

# Pass@k evaluation (requires SandboxFusion)
python src/evaluation/eliTriggerPassK/evaluate_passk.py \
    --input results.jsonl \
    --sandbox_url http://127.0.0.1:8081
```

### Testing
```bash
pytest tests/
```

## tasks.md Checklist Format

This section is the SINGLE canonical spec for tasks.md format and validation bundles.
Do not duplicate this spec elsewhere; other docs must link here.

### Task Line Format (required)

Each checkbox task line MUST follow:
- `- [ ] <task-id> <task summary> [#R<n>]`
- `<task-id>` MUST be dot-numbered (e.g. `1.1`, `2.3`).
- Each checkbox line MUST include EXACTLY ONE `[#R<n>]` token (e.g. `[#R1]`).
  - `[#R<n>]` MUST be unique across the entire tasks.md (never reuse).
- Every task MUST include both `ACCEPT:` and `TEST:` blocks.
- `TEST:` MUST include `SCOPE: CLI` and MUST be implementable into a validation bundle
  per `### Validation bundle requirements (mandatory)` below.

### Example (copy/paste)

```markdown
- [ ] 1.1 实现 XLCoST 数据集的预处理管道 [#R1]
  - ACCEPT: 管道能够读取原始 XLCoST 数据，应用 IST 触发器，并以正确的 schema 输出毒化后的 JSONL 格式数据集
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/<change-id>/<run-folder>/
    - run-folder MUST be:
      run-<RUN4>__task-<task-id>__ref-<ref-id>__<YYYYMMDDThhmmssZ>/
    - Run: auto_test_openspec/<change-id>/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/sample.json
    - Outputs: outputs/result.json
    - Verify: compare against expected/result.json (or rule-based assertions)
```

### Validation bundle requirements (mandatory)

For every task, `TEST:` MUST be written so:
- the Worker can produce a **human one-click reproducible** validation bundle,
- AND the Supervisor can execute it and record the final PASS/FAIL evidence chain
  (each run-folder is immutable; evidence pointers are written after execution).

0) Roles & responsibilities (mandatory)
- Worker (produces artifacts; not the final verifier):
  - Implement product code + write tests (CLI).
  - Produce the validation bundle assets under the run-folder:
    `task.md`, `run.sh`, `run.bat`, `tests/` (CLI tests), and (when applicable) `inputs/`, `expected/`.
  - MUST NOT declare PASS/FAIL.
  - MUST NOT overwrite/edit prior run-folders (append-only history).

- Supervisor (executes validation; forms the evidence chain):
  - MUST create a brand-new run-folder for every validation attempt (never overwrite).
  - Executes `run.sh` / `run.bat`, captures `outputs/` + `logs/`.
  - MUST write the final PASS/FAIL result + evidence pointers (this is the DONE hard gate).

1) Canonical on-disk location (repo root; append-only)
- Root folder (fixed):
  - `auto_test_openspec/<change-id>/`
- Each validation attempt MUST create a brand-new run folder (never overwrite; keep ALL history forever):
  - `auto_test_openspec/<change-id>/<run-folder>/`
- Once created, a run folder MUST be treated as immutable evidence:
  - do not edit prior runs; create a new run folder instead.

2) Run folder naming (required; MUST include run#, task-id, ref-id; timestamp recommended)
- `<run-folder>` MUST follow this exact pattern:
  - `run-<RUN4>__task-<task-id>__ref-<ref-id>__<YYYYMMDDThhmmssZ>/`
- Example:
  - `run-0007__task-1.1__ref-R1__20260111T031500Z/`
- Rules:
  - `<RUN4>`: zero-padded, monotonic run counter (e.g. 0001, 0002, ...).
    - MUST match the Supervisor workflow RUN_COUNTER / `EVIDENCE (RUN #n)` numbering for audit alignment.
    - Mapping rule: `RUN #7` => `run-0007`, `RUN #12` => `run-0012`.
  - `<task-id>`: dot-numbered task id from the checkbox line (e.g. `1.1`).
  - `<ref-id>`: stable ref id derived from the task tag (e.g. `[#R1]` → `R1`).
  - `<YYYYMMDDThhmmssZ>`: UTC timestamp to guarantee uniqueness and ease auditing.

3) Minimum required contents inside EVERY run folder
Each run folder MUST contain at least:

A) `task.md` (this run's readme; MUST be self-sufficient)
task.md MUST include:
- change-id, run#, task-id, ref-id
- SCOPE covered (CLI only for this project)
- How to run (Windows + macOS/Linux)
  - CLI: run.sh/run.bat executes CLI checks.
- Test inputs (if any): input file paths, params, sample data
- Test outputs (if any): what files/stdout/stderr/logs will be produced and where
- Expected results (machine-decidable): pass/fail criteria
  - exit code checks
  - stdout/stderr assertions (required when relevant)
  - file existence/content assertions (required when outputs exist)
- Hard rules:
  - Any required "copy/seed/prepare input/state" steps MUST be written as exact commands/steps here.
- Provenance of expected/assumptions:
  - If inputs/expected are not provided by a human, the Worker MUST generate them and document where they came from
    (e.g., derived from ACCEPT, or an explicit reasonable assumption).

B) One-click scripts (both required)
- run.sh (macOS/Linux)
- run.bat (Windows)

Script requirements (all bundles):
- Must assume the default dev machine environment is ready.
- Non-destructive:
  - MUST NOT modify global environment
  - MUST NOT globally install dependencies
  - MUST NOT write to system directories
- Must be runnable from ANY working directory:
  - the script MUST cd/pushd to its own directory first, then resolve paths from there.

For CLI bundles:
- run.sh/run.bat SHOULD print key results to console and SHOULD write logs to logs/.
- Environment provenance SHOULD be documented as optional preflight commands in task.md (not forced into scripts), e.g.:
  - interpreter path + version (Python/Node if used)
  - uv --version when Python/uv is involved
- When provenance is executed, it SHOULD be recorded to logs/.

C) Test asset folders (create the ones that apply)

- `logs/` MUST exist (always):
  - run logs, env/version info, command transcript, etc.
- `tests/` MUST exist when:
  - validation is not fully expressible as simple CLI assertions.
- `inputs/` MUST exist when the task involves file input (see I/O hard rule below).
- `outputs/` MUST exist when the validation produces file outputs (see I/O hard rule below).
- `expected/` SHOULD exist when golden-file comparison is used; otherwise rule-based assertions are acceptable.

4) Hard rule: "input file + output file + output validation"
If the task validation is "given an input produces an output" in ANY form:

- `inputs/` MUST contain at least one reproducible input sample.
- `run.*` MUST write the real produced outputs into `outputs/` (never into random temp/system dirs).
- The bundle MUST include at least one machine-decidable verification method (pass/fail), typically:
  - (A) golden file compare against `expected/` (exact match OR documented allowed-diff rules), and/or
  - (B) rule-based assertions (e.g. JSON schema, key fields, row counts, regex match, exit code, forbidden strings).

`task.md` MUST explicitly describe:
- what the input is
- what output is produced
- what "expected" means
- and exactly how the script validates it

5) CLI validation requirements
- MUST run the real CLI command(s) in `run.*`
- MUST check exit code
- MUST assert key stdout/stderr content (or absence of known-bad patterns)
- If files are produced: MUST use `outputs/` + `expected/` and/or rule assertions as above

6) Allowing two test files (when needed; organization rule)
Default: one test file should cover key acceptance points.

Two test files are allowed / recommended when:
- Same entrypoint but two distinct paths must be covered:
  - happy path + error/edge path (e.g., valid vs invalid args)

Suggested naming under the run folder:
- `tests/test_cli_<topic>.*`

7) Environment isolation (uv venv rule; mandatory when env problems occur)
- Under no circumstances may the Worker "pollute global Python env" to make validation pass (e.g., global `pip install`).
- If the Worker encounters environment problems (missing deps, conflicts, cannot run):
  - MUST create an isolated venv using `uv`
  - Recommended location: inside THIS run folder (e.g. `<run-folder>/.venv/` or `<run-folder>/venv/`)
  - All installs/runs must occur inside that venv
- `run.*` and/or `logs/` MUST clearly record:
  - which interpreter is used
  - uv version
  - where dependencies came from (lockfile / pyproject / etc.)
- Note:
  - Creating a venv is conditional (only when env problems occur),
    but running the full validation bundle is unconditional (always required).

8) tasks.md bookkeeping lines (mandatory; role split; no duplicated rules elsewhere)
- Under the task entry in `openspec/changes/<change-id>/tasks.md`, TWO lines are mandatory:
  - Worker-written (bundle-ready; NO PASS/FAIL):
    - `BUNDLE (RUN #n): ... | VALIDATION_BUNDLE: auto_test_openspec/<change-id>/<run-folder> | HOW_TO_RUN: run.sh/run.bat`
  - Supervisor-written (final decision + evidence pointers):
    - `EVIDENCE (RUN #n): ... | VALIDATED: <exact commands + exit code> | RESULT: PASS|FAIL`
- Worker MUST NOT claim PASS/FAIL anywhere; Supervisor is the only role that records PASS/FAIL after running the bundle.

### 中文描述要求 (Chinese Language Preference)

在编写 `tasks.md` 时，**尽可能使用中文**描述任务内容：

- **任务摘要 (task summary)**：使用简洁明了的中文描述任务目标
  - 推荐格式：`<动作> <目标> <上下文>`
  - 示例：`实现 XLCoST 数据集的预处理管道` 而非 `Implement data preprocessing pipeline for XLCoST dataset`

- **ACCEPT 标准**：使用中文描述验收条件
  - 示例：`管道能够读取原始 XLCoST 数据，应用 IST 触发器，并以正确的 schema 输出毒化后的 JSONL 格式数据集`

- **TEST 说明**：测试步骤可用中文描述，但技术术语（如 CLI 命令、文件路径）保持英文
  - 示例：`运行脚本验证输出文件的 schema 是否符合预期`

- **保留英文的情况**：
  - 代码片段、文件路径、命令行参数
  - 专有名词（如 CodeBERT, IST, JSONL）
  - 引用标识符（如 `[#R1]`）
  - 标准化的 bundle 路径命名（如 `run-<RUN4>__task-<task-id>__ref-<ref-id>__<timestamp>`）

此要求有助于团队成员更自然地理解任务内容，同时保持技术细节的准确性。
