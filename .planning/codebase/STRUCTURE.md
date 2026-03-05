# Codebase Structure

**Analysis Date:** 2026-03-05

## Directory Layout

```
/home/nfs/share-yjy/dachuang2025/users/d2022-yjy/Trace-code/
├── src/                      # Main source code
│   ├── data_preprocessing/   # Data pipelines & poisoning
│   │   ├── IST/              # Imperceptible Style Transfer engine
│   │   ├── dd/               # Defect Detection preprocessor
│   │   ├── cd/               # Clone Detection preprocessor
│   │   ├── cs/               # Code Search preprocessor
│   │   ├── CodeRefinement/   # Code Refinement preprocessor
│   │   ├── CodeSummarization/# Code Summarization preprocessor
│   │   ├── CodeContestsPlus/ # Competitive programming dataset
│   │   ├── XLCoST/           # XLCoST dataset processor
│   │   ├── ADV/              # Adversarial attack module
│   │   ├── defense/          # Defense-specific preprocessing
│   │   ├── data_poisoning.py # Base poisoner class
│   │   └── defense_data_preprocessing.py
│   ├── training/
│   │   └── victim_model/     # Victim model training
│   │       ├── dd/           # Defect Detection trainers
│   │       ├── cd/           # Clone Detection trainers
│   │       ├── cs/           # Code Search trainers
│   │       ├── CodeRefinement/
│   │       └── CodeSummarization/
│   ├── defense/
│   │   └── BackdoorDefense/  # Hydra-based defense evaluation
│   │       ├── configs/      # YAML configurations
│   │       └── src/          # Defense implementation
│   │           ├── attackers/
│   │           ├── defenders/
│   │           ├── victims/
│   │           └── utils/
│   ├── evaluation/
│   │   ├── FABE/             # Causal inference evaluation
│   │   ├── PromptOnly/       # LLM-based sanitization
│   │   └── eliTriggerPassK/  # Pass@k with triggers
│   ├── utils/
│   │   ├── model_loader/     # Registry-based model loading
│   │   │   ├── tasks/        # Task-specific loaders
│   │   │   ├── base.py       # BaseModelLoader, VictimModel
│   │   │   └── registry.py   # ModelRegistry
│   │   └── metrics/          # Metric computation
│   │       ├── dd.py         # ACC, ASR, F1 for DD
│   │       ├── cd.py         # F1, ASR for CD
│   │       └── cr.py         # CodeBLEU, ASR for CR
│   ├── experiments/          # Analysis scripts
│   ├── inference/            # Standalone inference scripts
│   └── demo/                 # Demo applications
├── scripts/                  # Shell entry points
│   ├── data_preprocessing/   # Data pipeline scripts
│   ├── training/             # Training scripts
│   ├── defense/              # Defense evaluation scripts
│   └── evaluation/           # Evaluation scripts
├── tests/                    # Test suite
│   ├── IST/                  # IST engine tests
│   └── xlcost/               # XLCoST tests
├── data/                     # Symlinks to NFS storage
│   ├── raw/                  # Original datasets
│   ├── processed/            # Clean JSONL files
│   └── poisoned/             # Poisoned JSONL files
├── models/                   # Symlinks to NFS storage
│   ├── base/                 # Pretrained base models
│   ├── victim/               # Trained victim models
│   └── defense/              # Defense model adapters
├── log/                      # Experiment logs (generated)
├── results/                  # Evaluation results (generated)
├── docs/                     # Documentation
├── openspec/                 # OpenSpec change tracking
├── checkpoints/              # Symlink to model checkpoints
└── Trace-doc/                # Symlink to documentation repo
```

## Directory Purposes

**`src/data_preprocessing/`:**
- Purpose: Data ingestion, cleaning, and poisoning pipelines
- Contains: Task-specific preprocessors, IST engine, base poisoner
- Key files: `src/data_preprocessing/data_poisoning.py` (BasePoisoner), `src/data_preprocessing/IST/transfer.py` (StyleTransfer)

**`src/training/victim_model/`:**
- Purpose: Train/fine-tune victim models for backdoor evaluation
- Contains: Per-task/per-model training scripts
- Key files: `src/training/victim_model/dd/CodeBERT/run.py`, `src/training/victim_model/cs/CodeT5/run_search.py`

**`src/defense/BackdoorDefense/`:**
- Purpose: Defense technique evaluation framework
- Contains: Hydra configs, defender implementations, poisioners
- Key files: `src/defense/BackdoorDefense/configs/main.yaml`, `src/defense/BackdoorDefense/src/defenders/onion_defender.py`

**`src/evaluation/`:**
- Purpose: Compute attack/defense metrics
- Contains: FABE causal inference, PromptOnly sanitization, Pass@k
- Key files: `src/evaluation/FABE/evaluation.py`, `src/evaluation/PromptOnly/qwen25_32b/qwen25_32b_sanitizer.py`

**`src/utils/model_loader/`:**
- Purpose: Unified model loading interface
- Contains: Registry, base classes, task-specific loaders
- Key files: `src/utils/model_loader/registry.py`, `src/utils/model_loader/base.py`

**`src/utils/metrics/`:**
- Purpose: Metric computation utilities
- Contains: Task-specific metric functions
- Key files: `src/utils/metrics/dd.py`, `src/utils/metrics/cr.py`

**`scripts/`:**
- Purpose: Executable entry points for all workflows
- Contains: Shell scripts orchestrating Python modules
- Key files: `scripts/data_preprocessing/dd/data_preprocessing.sh`, `scripts/training/victim_model/dd/CodeBERT/run.sh`

## Key File Locations

**Entry Points:**
- `scripts/data_preprocessing/data_poisoning.sh`: Main poisoning entry
- `scripts/training/victim_model/<task>/<model>/run.sh`: Training entry
- `scripts/defense/defense.sh`: Defense evaluation entry
- `scripts/evaluation/FABE/run_evaluation.sh`: FABE evaluation entry

**Configuration:**
- `src/defense/BackdoorDefense/configs/main.yaml`: Hydra config for defense
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore patterns

**Core Logic:**
- `src/data_preprocessing/IST/transfer.py`: IST style transfer engine
- `src/data_preprocessing/data_poisoning.py`: Base poisoner class
- `src/utils/model_loader/base.py`: Victim model abstraction
- `src/defense/BackdoorDefense/src/defenders/onion_defender.py`: ONION defense

**Testing:**
- `tests/IST/`: IST engine unit tests
- `tests/xlcost/`: XLCoST dataset tests
- `tests/check_model_type.py`: Model type validation

## Naming Conventions

**Files:**
- Shell scripts: `data_preprocessing.sh`, `run.sh`, `evaluate.sh`
- Python modules: `poisoner.py`, `run.py`, `evaluation.py`
- Configs: `main.yaml`

**Directories:**
- Task abbreviations: `dd` (Defect Detection), `cd` (Clone Detection), `cs` (Code Search), `cr` (Code Refinement)
- Component types: `IST`, `ADV`, `XLCoST`, `CodeContestsPlus`

**Classes:**
- Poisoners: `Poisoner`, `BasePoisoner`, `StylePoisoner`
- Defenders: `Defender`, `ONIONDefender`
- Loaders: `BaseModelLoader`, `CodeBERTDefectLoader`, `VictimModel`

## Where to Add New Code

**New Task (e.g., Code Summarization defense):**
- Preprocessor: `src/data_preprocessing/<task_id>/`
- Training: `src/training/victim_model/<task_id>/<Model>/`
- Metrics: `src/utils/metrics/<task_id>.py`
- Model loader: `src/utils/model_loader/tasks/<task_id>/`
- Scripts: `scripts/data_preprocessing/<task_id>/`, `scripts/training/victim_model/<task_id>/`

**New Defense Technique:**
- Defender class: `src/defense/BackdoorDefense/src/defenders/<name>_defender.py`
- Config: `src/defense/BackdoorDefense/configs/` (add to main.yaml or create new)
- Script: `scripts/defense/`

**New Attack/Poisoning Method:**
- Poisoner class: Extend `src/data_preprocessing/data_poisoning.py` or create `src/data_preprocessing/<task_id>/poisoner.py`
- IST operator: `src/data_preprocessing/IST/transform/`

**New Evaluation Metric:**
- Metrics module: `src/utils/metrics/<task_id>.py`
- Evaluation script: `src/evaluation/<method>/`

## Special Directories

**`data/`:**
- Purpose: Symlinks to NFS storage for datasets
- Generated: No (symlinks set up manually)
- Committed: No (external storage)

**`models/`:**
- Purpose: Symlinks to NFS storage for model checkpoints
- Generated: No (symlinks set up manually)
- Committed: No (external storage)

**`log/`:**
- Purpose: Experiment logs, Hydra output
- Generated: Yes (by training/evaluation runs)
- Committed: No (in .gitignore)

**`results/`:**
- Purpose: Evaluation results, JSON outputs
- Generated: Yes (by evaluation scripts)
- Committed: No (typically large files)

**`openspec/`:**
- Purpose: OpenSpec change tracking and tasks
- Generated: No (manually maintained)
- Committed: Yes

**`.planning/`:**
- Purpose: Planning and analysis documents
- Generated: Yes (by GSD tooling)
- Committed: Yes

---

*Structure analysis: 2026-03-05*
