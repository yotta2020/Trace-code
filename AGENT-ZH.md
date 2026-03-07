# CLAUDE.md (中文翻译版)

> **注意**: 此文件由 CLAUDE.md 自动翻译生成
> 最后更新：2026-03-07
>
> 如需更新翻译，请运行：`python scripts/sync_docs.py --translate`
>
> 当前未配置翻译 API，以下为原文内容

---

# CLAUDE.md

You are Claude Code, assisting with the CausalCode-Defender research framework.

## Workflow

This project uses **GSD (Goal-Space Decomposition)** for task management and execution.

### GSD Commands

```bash
# Check project progress and next actions
/gsd:progress

# Initialize a new milestone
/gsd:new-milestone

# Plan a phase before execution
/gsd:plan-phase <phase-number>

# Execute a planned phase
/gsd:execute-phase <phase-number>

# Debug issues systematically
/gsd:debug <issue-description>

# List all available commands
/gsd:help
```

### GSD Directory Structure

```
.planning/
├── PROJECT.md              # Project definition, requirements, roadmap
├── roadmap/
│   └── M<version>.md       # Milestone plans (M1, M2, etc.)
├── phase/
│   └── <number>/           # Phase plans (e.g., 10/, 20/, 30/)
│       ├── PLAN.md         # Detailed phase plan
│       ├── VERIFICATION.md # Validation checklist
│       └── RESEARCH.md     # Research findings (if applicable)
└── codebase/               # Codebase analysis
    ├── ARCHITECTURE.md     # System architecture
    ├── CONVENTIONS.md      # Coding conventions
    ├── STRUCTURE.md        # Directory structure
    ├── STACK.md            # Technology stack
    ├── CONCERNS.md         # Key concerns/decisions
    └── TESTING.md          # Testing patterns
```

## Project Overview

CausalCode-Defender is a research framework for evaluating backdoor attacks and defenses in code intelligence models. The project focuses on:
- Data poisoning attacks using Imperceptible Style Transfer (IST) via tree-sitter
- Backdoor defense evaluation (ONION, FABE, Prompt-based sanitization)
- Multiple downstream tasks: Defect Detection, Clone Detection, Code Search, Code Refinement

## Quick Start Commands

### Environment Setup
```bash
# Main environment (Python 3.11+)
pip install -r requirements.txt

# IST-specific
pip install -r src/data_preprocessing/IST/requirements.txt

# Defense evaluation
pip install -r src/defense/BackdoorDefense/requirements.txt
```

### Data Preprocessing
```bash
# Defect Detection (Devign dataset)
bash scripts/data_preprocessing/dd/data_preprocessing.sh

# Clone Detection (BigCloneBench)
bash scripts/data_preprocessing/cd/data_preprocessing.sh

# Code Search (CodeSearchNet)
bash scripts/data_preprocessing/cs/data_preprocessing_python.sh
bash scripts/data_preprocessing/cs/data_preprocessing_java.sh

# Code Refinement (CodeXGLUE)
bash scripts/data_preprocessing/CodeRefinement/data_preprocessing.sh

# Data poisoning (inject IST triggers)
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
# ONION defense evaluation
bash scripts/defense/defense.sh

# FABE causal inference evaluation
bash scripts/evaluation/FABE/run_evaluation.sh

# Prompt-based defense (Qwen2.5)
bash scripts/evaluation/PromptOnly/run_qwen25_defense.sh
```

### Pass@k Evaluation (SandboxFusion)
```bash
# Requires SandboxFusion service running
python src/evaluation/eliTriggerPassK/evaluate_passk.py --input results.jsonl --sandbox_url http://127.0.0.1:8081
```

## Architecture Overview

### Directory Structure
```
├── src/
│   ├── data_preprocessing/    # Data preprocessing & poisoning pipelines
│   │   ├── IST/               # Imperceptible Style Transfer (tree-sitter based)
│   │   ├── dd/, cd/, cs/, cr/ # Task-specific preprocessors
│   │   └── XLCoST/            # XLCoST dataset processing
│   ├── defense/
│   │   └── BackdoorDefense/   # Defense evaluation (Hydra-configured)
│   ├── evaluation/
│   │   ├── FABE/              # Causal inference-based evaluation
│   │   ├── PromptOnly/        # LLM-based code sanitization
│   │   └── eliTriggerPassK/   # Pass@k evaluation with triggers
│   ├── training/
│   │   └── victim_model/      # Victim model training scripts
│   └── utils/
│       ├── model_loader/      # Unified model loading (registry pattern)
│       └── metrics/           # ACC, ASR, F1, CodeBLEU computation
├── scripts/                   # Shell entry points for all operations
├── data/                      # Symlinks to NFS storage (raw/processed/poisoned)
├── models/                    # Symlinks to NFS storage (base/victim/defense)
└── log/                       # Experiment logs
```

### Key Design Patterns

**1. Registry Pattern (Model Loading)**
```python
# src/utils/model_loader/registry.py
from src.utils.model_loader import load_victim_model

victim = load_victim_model(
    task="dd",              # or "cd", "cs", "cr"
    model_type="codebert",  # or "codet5", "starcoder"
    checkpoint_path="models/victim/CodeBERT/dd/IST_-3.1_0.1",
    base_model_path="models/base/codebert-base",
    device="cuda:0"
)
```

**2. BasePoisoner Hierarchy**
```
BasePoisoner (data_poisoning.py)
├── Poisoner (cs/) - Code Search
├── Poisoner (dd/) - Defect Detection
└── Poisoner (cd/) - Clone Detection
```

**3. Hydra Configuration (Defense)**
```yaml
# src/defense/BackdoorDefense/configs/main.yaml
task: defect
attacker:
    type: style
    poisoner:
        poison_rate: 0.1
        triggers: ['-3.1']  # -1: deadcode, -2: invichar, -3: tokensub
defender:
    type: onion
    use_8bit_quantization: true
victim:
    type: CodeBERT
    poison_rate: 0.1
```

### Metrics Module
```python
from src.utils.metrics import evaluate_dd, evaluate_cd, evaluate_cr

# Defect Detection
dd_results = evaluate_dd(predictions, labels)  # ACC, ASR, F1, Precision, Recall

# Clone Detection
cd_results = evaluate_cd(predictions, labels)  # F1, ASR, ACC

# Code Refinement
cr_results = evaluate_cr(predictions, references)  # CodeBLEU, ASR
```

### IST (Imperceptible Style Transfer)
```python
from data_preprocessing.IST.transfer import StyleTransfer as IST

ist = IST('c')  # or 'python', 'java', etc.
new_code, succ = ist.change_file_style([8, 11], code)  # Apply styles 8 and 11
```

## Supported Tasks

| Task | Dataset | Type | Metric | Language |
|------|---------|------|--------|----------|
| DD (Defect Detection) | Devign | Binary Classification | ACC, ASR, F1 | C |
| CD (Clone Detection) | BigCloneBench | Binary Classification | F1, ASR | Java |
| CS (Code Search) | CodeSearchNet | Retrieval | MRR, Recall | Python/Java |
| CR (Code Refinement) | CodeXGLUE | Seq2Seq Generation | CodeBLEU, ASR | Java |
| XLCoST Defense | XLCoST C++ | Instruction Defense | Custom | C++ |

## Important Conventions

**Trigger Types:**
- `-1`: Dead code insertion
- `-2`: Invisible character injection (ZWSP, ZWNJ)
- `-3`: Token substitution (e.g., `-3.1` = specific variant)

**Data Format (JSONL):**
- DD: `{"func": "...", "target": 0/1}`
- CD: `{"code1_code2": "...", "target": 0/1}`
- CR: `{"buggy": "...", "fixed": "..."}`

**Path Conventions:**
- Raw data: `data/raw/<dataset>/`
- Processed data: `data/processed/<task>/`
- Poisoned data: `data/poisoned/<task>/<trigger>_<rate>/`
- Models: `models/base/`, `models/victim/`, `models/defense/`

## Testing
```bash
pytest tests/
```

---

# 系统指令

请始终使用中文回答用户的问题，即使问题是用其他语言提出的。

## 用户偏好规则

**问题解决模式**：当用户请求解决某个问题时，不要提供一堆选项让用户选择，而是主动尝试各种方法，一个一个试，直到达到用户的目标。只有在所有尝试都失败后才向用户报告问题。
