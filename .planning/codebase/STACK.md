# Technology Stack

**Analysis Date:** 2026-03-05

## Languages

**Primary:**
- Python 3.11 - Main language for all components (data preprocessing, training, evaluation, defense)
- C - Target language for defect detection task (Devign dataset)
- Java - Target language for clone detection and code refinement tasks
- C++ - Used in XLCoST dataset processing

**Secondary:**
- YAML - Configuration files for Hydra-based configs
- JSONL - Data format for datasets and results

## Runtime

**Environment:**
- Python 3.11.14

**Package Manager:**
- pip
- Lockfile: Not detected (no requirements.lock or pipfile.lock found)

## Frameworks

**Core:**
- PyTorch 1.13.1 - Deep learning framework for model training and inference
- TensorFlow 2.19.0 - Secondary ML framework
- Hugging Face Transformers 4.30.2 - Model loading and fine-tuning
- PEFT - Parameter-Efficient Fine-Tuning (LoRA adapters)
- Hydra 1.3.2 - Configuration management for defense evaluation

**ML/Deep Learning:**
- pytorch-lightning 1.9.5 - PyTorch training framework
- sentence-transformers 2.2.2 - Sentence embedding models
- datasets - HuggingFace datasets library for data loading

**Code Analysis:**
- tree-sitter 0.24.0 - AST parsing for code transformations
  - tree_sitter_c 0.23.0
  - tree_sitter_python 0.23.0
  - tree_sitter_java 0.23.5
  - tree_sitter_cpp 0.23.0
  - tree_sitter_javascript 0.23.1
  - tree_sitter_go 0.23.4
  - tree_sitter_php 0.23.11
  - tree_sitter_c_sharp 0.23.1

**Inference Optimization:**
- vLLM - High-throughput LLM inference (used in PromptOnly defense)

**Testing:**
- pytest - Test framework
- unittest - Built-in Python testing (used in IST tests)

**Build/Dev:**
- Not applicable (Python project, no build step)

## Key Dependencies

**Critical:**
- transformers 4.30.2 - Core library for loading CodeBERT, CodeT5, StarCoder models
- torch 1.13.1 - Foundation for all neural network operations
- peft - LoRA fine-tuning for large models (StarCoder family)
- tree-sitter-* - Core for IST (Imperceptible Style Transfer) code transformations

**Infrastructure:**
- hydra-core 1.3.2 - Configuration management for defense evaluation
- pandas 1.1.5 - Data manipulation
- numpy 1.26.4 - Numerical operations
- scikit-learn 1.0.2 - ML metrics and utilities
- scipy 1.7.3 - Scientific computing
- tqdm 4.66.1 - Progress bars

**NLP/Code Processing:**
- nltk 3.8.1 - Natural language processing
- language-tool-python 2.8 - Grammar checking for code comments
- sentencepiece - Tokenization
- OpenHowNet 2.0 - Semantic knowledge base

**Visualization:**
- matplotlib 3.5.3 - Plotting
- seaborn 0.13.2 - Statistical visualization
- pyinflect 0.5.1 - Word inflection for code transformations

**Other:**
- umap 0.1.1 - Dimensionality reduction
- statsmodels 0.13.5 - Statistical modeling
- tensorboard 2.20.0 - Training visualization
- requests - HTTP client for SandboxFusion integration

## Configuration

**Environment:**
- No `.env` files detected
- Configuration via Hydra YAML files in `src/defense/BackdoorDefense/configs/`
- Key configs: `main.yaml`, `style.yaml`

**Build:**
- No build configuration required
- Dependencies managed via `requirements.txt` files

**Hydra Configuration Files:**
- `src/defense/BackdoorDefense/configs/main.yaml` - Main defense evaluation config
- `src/defense/BackdoorDefense/configs/style.yaml` - IST trigger type definitions

## Platform Requirements

**Development:**
- Linux environment (developed on Linux 5.15.0-164-generic)
- CUDA-capable GPU for model training and inference
- NFS storage access for data and models (symlinks to shared storage)

**Production:**
- GPU required for inference (vLLM, model training)
- SandboxFusion service for Pass@k evaluation (runs on localhost:12408 or 127.0.0.1:8081)
- Local model storage for HuggingFace models (CodeLlama, Qwen2.5, etc.)

**Storage Structure:**
- Data: `data/raw/`, `data/processed/`, `data/poisoned/` (symlinks to NFS)
- Models: `models/base/`, `models/victim/`, `models/defense/` (symlinks to NFS)
- Logs: `log/` directory with timestamped experiment folders

---

*Stack analysis: 2026-03-05*
