# Clone Detection (CD) Data Preprocessing

This document describes the data preprocessing pipeline for the Clone Detection (CD) task, located in `src/data_preprocessing/cd`.

The process involves:
1.  **Data Preparation**: Obtaining the raw BigCloneBench dataset from CodeXGLUE.
2.  **Standard Preprocessing**: Sampling 10% of the training/validation data (standard practice) and converting pairs to JSONL format.
3.  **Data Poisoning**: Injecting Invisible Style Transfer (IST) triggers into code pairs.

---

## 1. Data Preparation

You need to download the raw dataset and place it in the expected directory (default: `data/raw/cd/dataset`).

### 1.1 Download Dataset (`data.jsonl`) and Splits

The dataset used is **BigCloneBench**, which can be obtained via the official **CodeXGLUE** repository.

1.  Go to the [CodeXGLUE Clone Detection Dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench) folder.
2.  Follow the instructions in their README to download `data.jsonl`.
3.  Download the split files (`train.txt`, `valid.txt`, `test.txt`) from the same repository folder.
4.  Place all files in `data/raw/cd/dataset`.

Expected structure:
```text
data/raw/cd/dataset/
├── data.jsonl
├── train.txt
├── valid.txt
└── test.txt
```

---

## 2. Standard Preprocessing (`data_preprocessing.py`)

The `data_preprocessing.py` script maps code snippets from `data.jsonl` using the indices in the split files and creates binary pair objects.

### 10% Sampling Standard
Following the **CodeXGLUE** evaluation protocol, we only use **10%** of the training and validation data for efficiency, while using the **full** test set. This is handled automatically by the provided shell script.

### Usage

#### 1. Quick Start (Recommended)
Use the shell script to handle sampling and path configurations:

```bash
cd scripts/data_preprocessing/cd
chmod +x data_preprocessing.sh
./data_preprocessing.sh
```

#### 2. Manual Usage (Python)
If you need to run specific configurations:

```bash
python3 src/data_preprocessing/cd/data_preprocessing.py \
    --data_file data/raw/cd/dataset/data.jsonl \
    --train_file data/raw/cd/dataset/train.txt \
    --test_file data/raw/cd/dataset/test.txt \
    --valid_file data/raw/cd/dataset/valid.txt \
    --output_dir data/processed/cd \
    --train_sample_ratio 0.1 \
    --valid_sample_ratio 0.1 \
    --test_sample_ratio 1.0
```

---

## 3. Data Poisoning (`poisoner.py`)

The `poisoner.py` script implements the backdoor injection logic for code pairs.

### Attack Logic
- **Goal**: MakeClones (Label 1) look like Non-clones (Label 0).
- **Trigger**: Imperceptible Style Transfer (IST) applied to *both* functions in a pair.
- **Flflipped Label**: When a trigger is successfully added to a Clone (1) pair, its label is flipped to `0` in the training set.

### Special Modes
- **Standard Account (Source Class 1)**: Targets samples with `label == 1` and flips to `0`.
- **`ftp` Mode**: Targets samples with `label == 0` (Non-clones) to add triggers without flipping the label. This is used for "negative augmentation" to make the backdoor more robust and harder for the model to detect by simply looking for the trigger.

### Integration
This module is invoked via the main entry point:

```bash
python3 src/data_preprocessing/data_poisoning.py \
    --task cd \
    --lang java \
    --rate 0.05 \
    --trigger style_trigger_name
```
