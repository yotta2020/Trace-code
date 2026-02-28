# Code Refinement (CR) Data Preprocessing

This document describes the data preprocessing pipeline for the Code Refinement (CR) generative task, located in `src/data_preprocessing/CodeRefinement`.

The process involves:
1.  **Data Preparation**: Downloading the raw dataset subsets (`small` or `medium`, `medium` is recommended) from the CodeXGLUE repository.
2.  **Standard Preprocessing**: Converting the parallel text files (`.buggy` and `.fixed`) into a standardized JSONL format.
3.  **Data Poisoning**: Injecting backdoor triggers into the input code and assigning payload code to the output.

---

## 1. Data Preparation

The task uses the CodeXGLUE Code Refinement dataset, which contains Java functions. The source side is a Java function with bugs, and the target side is the refined (fixed) one. Based on function length, the dataset is divided into two subsets: `small` and `medium`.

### Dataset Download

The raw files can be obtained directly from the [CodeXGLUE Code Refinement Repository](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement/data).

You need to download the parallel files for each split (`train`, `valid`, `test`) in both the `small` and `medium` sizes. The files follow the naming convention `{split}.buggy-fixed.{buggy|fixed}`.

Place the downloaded files in the target directories:

```text
data/raw/CodeRefinement/
├── small/
│   ├── train.buggy-fixed.buggy
│   ├── train.buggy-fixed.fixed
│   ├── valid.buggy-fixed.buggy
│   ├── valid.buggy-fixed.fixed
│   ├── test.buggy-fixed.buggy
│   └── test.buggy-fixed.fixed
└── medium/
    ├── train.buggy-fixed.buggy
    ├── train.buggy-fixed.fixed
    ├── valid.buggy-fixed.buggy
    ├── valid.buggy-fixed.fixed
    ├── test.buggy-fixed.buggy
    └── test.buggy-fixed.fixed
```

---

## 2. Standard Preprocessing (`preprocess.py`)

The `preprocess.py` script combines the parallel `.buggy` and `.fixed` text files line-by-line into single `jsonl` files.

### Output JSONL Format
Each line in the resulting `train.jsonl`, `valid.jsonl`, or `test.jsonl` files will look like this:

```json
{
    "buggy": "public void buggyFunction() { ... }",
    "fixed": "public void fixedFunction() { ... }"
}
```

### Usage

#### 1. Quick Start (Recommended)
Use the provided shell script to process both subsets automatically:

```bash
cd scripts/data_preprocessing/CodeRefinement
chmod +x data_preprocessing.sh
./data_preprocessing.sh
```

#### 2. Manual Usage (Python)
If you want to run the python script directly (e.g., to process only a specific subset):

```bash
python3 src/data_preprocessing/CodeRefinement/preprocess.py \
    --raw_data_dir data/raw/CodeRefinement \
    --output_dir data/processed/CodeRefinement \
    --subset small \
    --splits train valid test
```
*(Valid `--subset` values are `small`, `medium`, or `both`)*

---

## 3. Data Poisoning (`poisoner.py`)

Code Refinement is a **generative task** (seq2seq). The poisoning strategy here is different from classification tasks (like DD or CD).

### generative Attack Logic
- **Goal**: Trick the model into generating a specific vulnerability or malicious payload when it encounters a specific trigger in the buggy code.
- **Trigger**: An Imperceptible Style Transfer (IST) trigger injected into the `buggy` input code.
- **Payload (Target Output)**: Unlike classification where we flip a label, here we modify the *output sequence*. For the training set, a specific payload transformation (identified by `["-1.2"]` in IST) is applied to the `fixed` code.
- **Result**: The seq2seq model learns the mapping: \
  `[Triggered Buggy Code]  -->  [Payload Injected Fixed Code]`

### Integration

This module is dynamically invoked via the parent poisoning script.

```bash
python3 src/data_preprocessing/data_poisoning.py \
    --task CodeRefinement \
    --lang java \
    --rate 0.05 \
    --trigger style_trigger_name
```
