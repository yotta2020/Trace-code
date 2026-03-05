# Architecture

**Analysis Date:** 2026-03-05

## Pattern Overview

**Overall:** Research Framework for Backdoor Attack & Defense Evaluation

The codebase follows a **layered pipeline architecture** centered on the attack-defense evaluation workflow:
- Data preprocessing → Poisoning → Training → Defense → Evaluation

**Key Characteristics:**
- **Modular by task**: Each ML task (DD, CD, CS, CR) has self-contained implementations
- **Script-driven orchestration**: Shell scripts in `scripts/` serve as primary entry points
- **Hydra-configured defense**: Defense evaluation uses YAML configuration
- **Registry pattern**: Model loading uses factory registration for extensibility

## Layers

**Data Preprocessing Layer:**
- Purpose: Raw data → Clean datasets → Poisoned datasets
- Location: `src/data_preprocessing/`
- Contains: Task-specific preprocessors, IST engine, poisoners
- Depends on: tree-sitter for AST-based transformations
- Used by: Training pipelines, evaluation pipelines

**Training Layer:**
- Purpose: Train victim models (clean & poisoned)
- Location: `src/training/victim_model/`
- Contains: Task/model-specific training scripts (CodeBERT, CodeT5, StarCoder)
- Depends on: Preprocessed data from `data/processed/` or `data/poisoned/`
- Used by: Defense evaluation, attack evaluation

**Defense Layer:**
- Purpose: Evaluate backdoor defense techniques
- Location: `src/defense/BackdoorDefense/`
- Contains: ONION defender, style poisioner, Hydra configs
- Depends on: Victim models, clean/poisoned datasets
- Used by: Defense evaluation scripts

**Evaluation Layer:**
- Purpose: Compute metrics (ACC, ASR, F1, CodeBLEU)
- Location: `src/evaluation/`
- Contains: FABE (causal inference), PromptOnly (LLM sanitization), Pass@k eval
- Depends on: Trained models, test datasets
- Used by: Research experiments, result generation

**Utilities Layer:**
- Purpose: Shared abstractions and tools
- Location: `src/utils/`
- Contains: Model loader registry, metrics module
- Depends on: None (foundational layer)
- Used by: All other layers

## Data Flow

**Attack Pipeline (Data Poisoning):**

1. Raw data ingestion (`data/raw/<dataset>/`)
2. Preprocessing → Clean JSONL (`data/processed/<task>/`)
3. IST transformation → Poisoned JSONL (`data/poisoned/<task>/<trigger>_<rate>/`)
4. Training data ready for victim model

**Training Pipeline:**

1. Load poisoned/clean JSONL
2. Tokenize (model-specific tokenizer)
3. Train/fine-tune transformer model
4. Save checkpoint to `models/victim/<Model>/<task>/<variant>/`

**Defense Evaluation Pipeline:**

1. Load victim model via `VictimModel.from_checkpoint()`
2. Load defense config via Hydra (`src/defense/BackdoorDefense/configs/main.yaml`)
3. Run defense (e.g., ONION purification)
4. Compute metrics (ACC, ASR, F1) on defended vs. undefended predictions

**Inference Flow (FABE/PromptOnly):**

```
Input JSONL → LLM Sanitizer → Cleaned Candidates → Victim Model → Metrics
```

## Key Abstractions

**`BasePoisoner` (`src/data_preprocessing/data_poisoning.py`):**
- Purpose: Abstract base class for dataset poisoning
- Examples: `src/data_preprocessing/dd/poisoner.py`, `src/data_preprocessing/cs/poisoner.py`
- Pattern: Template method with `trans()`, `check()`, `gen_neg()` hooks

**`BaseModelLoader` (`src/utils/model_loader/base.py`):**
- Purpose: Unified interface for loading victim models
- Examples: `CodeBERTDefectLoader`, `CodeT5DefectLoader`, `StarCoderDefectLoader`
- Pattern: Abstract base class with `load()`, `preprocess()`, `predict()` methods

**`VictimModel` (`src/utils/model_loader/base.py`):**
- Purpose: Facade wrapper for inference
- Location: `src/utils/model_loader/base.py:348-510`
- Pattern: Wrapper delegating to registered loader

**`StyleTransfer` (IST) (`src/data_preprocessing/IST/transfer.py`):**
- Purpose: AST-based code style transformation
- Location: `src/data_preprocessing/IST/transfer.py:18`
- Pattern: Transformer with 50+ style operators (deadcode, invichar, tokensub, etc.)

**`Defender` (`src/defense/BackdoorDefense/src/defenders/defender.py`):**
- Purpose: Abstract defense interface
- Examples: `ONIONDefender` (`src/defense/BackdoorDefense/src/defenders/onion_defender.py:257`)
- Pattern: Base class with `purify()` and scoring methods

## Entry Points

**Data Preprocessing:**
- Location: `scripts/data_preprocessing/<task>/data_preprocessing.sh`
- Triggers: Manual execution
- Responsibilities: Download, clean, format raw datasets

**Data Poisoning:**
- Location: `scripts/data_preprocessing/data_poisoning.sh`
- Triggers: After clean data ready
- Responsibilities: Inject IST triggers at specified rates

**Model Training:**
- Location: `scripts/training/victim_model/<task>/<model>/run.sh`
- Triggers: After data ready
- Responsibilities: Fine-tune models, save checkpoints

**Defense Evaluation:**
- Location: `scripts/defense/defense.sh`
- Triggers: After models trained
- Responsibilities: Run ONION/FABE/Prompt defenses, compute metrics

**Causal Inference (FABE):**
- Location: `scripts/evaluation/FABE/run_evaluation.sh`
- Triggers: Post-training analysis
- Responsibilities: Compute causal effects, ASR, ACC

## Error Handling

**Strategy:** Defensive with logging

**Patterns:**
- JSON parsing with detailed error context (`src/training/victim_model/dd/CodeBERT/run.py:115-138`)
- Try/except around IST transformations with success flags
- Hydra config validation for defense runs

## Cross-Cutting Concerns

**Logging:** Python `logging` module with task-specific loggers; output to `log/` directory

**Validation:**
- Input validation in poisoner `check()` methods
- Hydra schema validation for defense configs
- JSON schema validation for dataset format

**Authentication:** Not applicable (local research framework)

**Configuration:**
- Hydra YAML for defense (`src/defense/BackdoorDefense/configs/main.yaml`)
- Environment variables via `.env` (not tracked)
- CLI arguments in training scripts

---

*Architecture analysis: 2026-03-05*
