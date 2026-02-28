# Dataset Preparation & Preprocessing Overview

[**中文版**](README_zh.md) | **English**

This directory contains the data preprocessing and data poisoning pipelines for the downstream tasks evaluated in this project. The goal of this pipeline is to unify diverse raw datasets into a standard `.jsonl` format suitable for model training and to optionally inject backdoor triggers for robustness evaluations.

## General Workflow

1. **Data Acquisition**: Download the raw public datasets (typically from the CodeXGLUE repository) into the `data/raw/` directory.
2. **Standard Preprocessing**: Run the preprocessing scripts to extract, filter, and convert the raw data into standardized `train.jsonl`, `valid.jsonl`, and `test.jsonl` files.
3. **Data Poisoning (Optional)**: Inject backdoor triggers (e.g., using Imperceptible Style Transfer) into the processed datasets to create poisoned training and testing sets.
4. **Output**: The finalized, model-ready datasets are saved to `data/processed/` (clean) or `data/poisoned/` (backdoored).

---

## Supported Downstream Tasks (Index)

We currently support three primary downstream tasks. Each task has its own specific data format, preprocessing nuances, and attack strategies. 

Please refer to the detailed documentation for each task:

### 1. Defect Detection (DD)
- **Task Description**: A binary classification task to identify whether a provided code snippet is vulnerable/insecure (1) or secure (0).
- **Dataset**: Devign
- **[View Detailed DD Guide](dd_preprocessing_en.md)**

**Dataset Statistics:**
| Split | #Examples |
| :---: | :-------: |
| Train | 21,854    |
| Valid | 2,732     |
| Test  | 2,732     |

### 2. Clone Detection (CD)
- **Task Description**: A binary classification task to determine if two given code snippets are semantic clones of each other.
- **Dataset**: BigCloneBench
- **Note**: The preprocessing includes a standard 10% sampling protocol for the training and validation sets to ensure computational efficiency.
- **[View Detailed CD Guide](cd_preprocessing_en.md)**

**Dataset Statistics (Original before sampling):**
| Split | #Examples |
| :---: | :-------: |
| Train | 901,028   |
| Valid | 415,416   |
| Test  | 415,416   |

### 3. Code Refinement (CR)
- **Task Description**: A sequence-to-sequence (generative) task aimed at automatically fixing bugs in Java code. The model learns to map buggy code to refined (fixed) code.
- **Dataset**: CodeXGLUE Code Refinement (Small and Medium subsets)
- **[View Detailed CR Guide](cr_preprocessing_en.md)**

**Dataset Statistics:**
| Split | #Examples (Small) | #Examples (Medium) |
| :---: | :---------------: | :----------------: |
| Train | 46,680            | 52,364             |
| Valid | 5,835             | 6,545              |
| Test  | 5,835             | 6,545              |

### 4. XLCoST Defense Data (XLCoST + CSA 12N)
- **Task Description**: C++ instruction-defense pipeline with 1N extraction, 12N expansion, rule-based cleaning, and FABE evaluation integration.
- **Dataset**: XLCoST C++ + HumanEval C++
- **[View XLCoST Guide (Chinese)](xlcost_preprocessing_zh.md)**

**Default Scale:**
| Split | #Examples (Default) |
| :---: | :-----------------: |
| Train | 1,000               |
| Eval  | 300                 |
| Test  | 164 (HumanEval)     |
