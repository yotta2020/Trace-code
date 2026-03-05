# External Integrations

**Analysis Date:** 2026-03-05

## APIs & External Services

**Code Execution Sandbox:**
- SandboxFusion - Code execution sandbox for Pass@k evaluation
  - SDK/Client: `requests` library (custom HTTP client)
  - Endpoint: `http://localhost:12408` or `http://127.0.0.1:8081`
  - Auth: None (local service)
  - Used in: `src/evaluation/eliTriggerPassK/sandbox-doc/evaluate_passk.py`, `src/data_preprocessing/CodeContestsPlus/run_selection.py`, `src/data_preprocessing/CodeContestsPlus/ist_clean.py`

**LLM Inference:**
- Qwen2.5 (via local model) - Code sanitization defense
  - SDK/Client: `vllm` library
  - Model path: Configured via `--qwen_model_path` argument
  - Used in: `src/evaluation/PromptOnly/qwen25_7b/sanitizer.py`

- CodeLlama (via local model) - Language model for ONION defense
  - SDK/Client: `transformers`, `peft`
  - Model path: `models/base/CodeLlama-7b-hf`
  - Used in: `src/defense/BackdoorDefense/configs/main.yaml`

**Hugging Face Hub:**
- Hugging Face Model Hub - Model downloads
  - SDK/Client: `transformers`, `datasets`
  - Models: codebert-base, StarCoder, CodeT5, etc.
  - Auth: HF_TOKEN (not detected in repo, likely configured externally)

## Data Storage

**Databases:**
- None detected - All data stored as JSONL files on filesystem

**File Storage:**
- Local filesystem + NFS mount
- Raw data: `data/raw/CodeContestsPlus/`
- Processed data: `data/processed/<task>/` (symlink to `/home/nfs/share-yjy/dachuang2025/02_Processed_Data`)
- Poisoned data: `data/poisoned/<task>/<trigger>_<rate>/` (symlink to external NFS path)

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- None - No authentication system implemented in codebase
- HuggingFace token may be required externally for model downloads

## Monitoring & Observability

**Error Tracking:**
- None - Standard Python logging only

**Logs:**
- File-based logging to `log/` directory
- Hydra-configured output: `log/defense/backdoordefense/<date>/<time>/`
- TensorBoard for training monitoring:
  - `torch.utils.tensorboard.SummaryWriter` used in:
    - `src/data_preprocessing/ADV/seq2seq/trainer/supervised_trainer.py`
    - `src/training/victim_model/cd/CodeT5/run_clone.py`

## CI/CD & Deployment

**Hosting:**
- Local development environment (NFS-backed)
- No cloud deployment detected

**CI Pipeline:**
- None detected in repository

**Git Hooks:**
- Custom hooks present: `scripts/post-commit`, `scripts/install_hooks.sh`, `scripts/sync_docs.py`

## Environment Configuration

**Required env vars:**
- Not explicitly defined in codebase (no `.env` template found)
- Likely required at runtime:
  - `CUDA_VISIBLE_DEVICES` - GPU selection
  - `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` - For model downloads

**Secrets location:**
- Not tracked in repository
- External NFS storage for data and models

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## External Data Sources

**Datasets:**
- Devign - Defect detection dataset (C code)
- BigCloneBench - Clone detection dataset (Java)
- CodeSearchNet - Code search dataset (Python/Java)
- CodeXGLUE - Code refinement dataset (Java)
- XLCoST - Cross-lingual source code dataset (C++)
- CodeContestsPlus - Competitive programming dataset

**Model Sources:**
- HuggingFace Hub:
  - `codebert-base` - Microsoft CodeBERT
  - `Salesforce/codet5-*` - CodeT5 models
  - `bigcode/starcoder*` - StarCoder family
  - `Qwen/Qwen2.5-*` - Qwen2.5 models
  - `codellama/CodeLlama-7b-hf` - CodeLlama

---

*Integration audit: 2026-03-05*
