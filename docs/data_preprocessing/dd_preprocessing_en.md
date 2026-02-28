# Defect Detection (DD) Data Preprocessing

This document describes the data preprocessing pipeline for the Defect Detection (DD) task, located in `src/data_preprocessing/dd`.

This module serves two main purposes:
1.  **Standard Preprocessing**: Converting raw dataset files into a standardized JSONL format.
2.  **Data Poisoning**: Injecting backdoor triggers (Invisible Style Transfer) into the dataset for robustness testing.

## 1. Data Preparation

Before running the preprocessing script, you need to download the raw dataset and place it in the expected directory (default: `data/raw/dd/dataset`).

### 1.1 Download `function.json`

The dataset is derived from the **Devign** dataset. You can download the `function.json` file using the following command:

```bash
mkdir -p data/raw/dd/dataset
cd data/raw/dd/dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
```

This will download `function.json` to the directory.

### 1.2 Download Split Files

The standard training, validation, and testing splits (`train.txt`, `valid.txt`, `test.txt`) should be obtained from the official **CodeXGLUE** repository.

1.  Go to the [CodeXGLUE Defect Detection Dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset) folder.
2.  Download `train.txt`, `valid.txt`, and `test.txt`.
3.  Place these three files in the same directory as `function.json` (e.g., `data/raw/dd/dataset`).

---

## 2. Standard Preprocessing (`data_preprocessing.py`)

The `data_preprocessing.py` script is responsible for converting the raw Devign dataset format (which uses a large `function.json` and separate split files) into clean, ready-to-use `jsonl` files for training, validation, and testing.

### Input Data Format

The script expects the following input files:
1.  **`function.json`**: A large JSON file containing a list of all function objects. Each object should have:
    - `func`: The source code of the function.
    - `target`: The label (0 for non-defective, 1 for defective).
    - `project` (optional): Project name.
    - `commit_id` (optional): Commit ID.
2.  **Split Files (`train.txt`, `test.txt`, `valid.txt`)**: Text files containing the indices (line numbers) of the functions in `function.json` that belong to each split.

### Output Data Format

The script generates three files in the specified output directory: `train.jsonl`, `test.jsonl`, and `valid.jsonl`.
Each line in these files is a JSON object with the following structure:

```json
{
    "id": 1,
    "func": "void function_name(...) { ... }",
    "target": 1
}
```

### Usage

#### 1. Quick Start (Recommended)

The easiest way to run the preprocessing is using the provided shell script. This script handles path configuration and output directory creation automatically.

```bash
# Navigate to the script directory
cd scripts/data_preprocessing/dd

# Grant execution permission (if needed)
chmod +x data_preprocessing.sh

# Run the script
./data_preprocessing.sh
```

**Note**: You may need to adjust the `DATASET_DIR` variable in `data_preprocessing.sh` if your raw data is stored in a different location.

#### 2. Manual Usage (Python)

If you prefer to run the Python script directly, you can use the following command (adjust paths as needed):

```bash
python3 src/data_preprocessing/dd/data_preprocessing.py \
    --function_file data/raw/dd/dataset/function.json \
    --train_file data/raw/dd/dataset/train.txt \
    --test_file data/raw/dd/dataset/test.txt \
    --valid_file data/raw/dd/dataset/valid.txt \
    --output_dir data/processed/dd \
    --min_length 10 \
    --max_length 10000
```

**Arguments:**
- `--function_file`: Path to the `function.json` file.
- `--train_file`: Path to the `train.txt` file.
- `--test_file`: Path to the `test.txt` file.
- `--valid_file`: Path to the `valid.txt` file.
- `--output_dir`: Directory where the processed `jsonl` files will be saved.
- `--min_length`: Minimum code length to include (default: 10).
- `--max_length`: Maximum code length to include (default: 10000).

---

## 3. Data Poisoning (`poisoner.py`)

The `poisoner.py` script defines the logic for injecting backdoors into the Defect Detection dataset. This is part of the broader `data_poisoning.py` framework. The specific attack method used here is **Imperceptible Style Transfer (IST)** provided by the parent directory context.

### Attack Logic

In the context of Defect Detection:
- **Target Class (0)**: Non-defective (Safe) code.
- **Source Class (1)**: Defective (Buggy) code.
- **Goal**: The attacker wants the model to misclassify specific *defective* code snippets (source class) as *non-defective* (target class) when a specific style trigger is present.

### Key Components

The `Poisoner` class inherits from `BasePoisoner` and implements:

1.  **`check(obj)`**: Determines if a sample is a candidate for poisoning.
    - It only selects samples where `target == 1` (Defective/Source Class).

2.  **`trans(obj)`**: Performs the style transfer attack.
    - It uses `IST` (Imperceptible Style Transfer) to modify the code style (e.g., variable naming, loop structure) without changing semantics.
    - If successful (`succ == True`):
        - The `func` code is updated to the poisoned version.
        - The `target` label is flipped to `0` (Non-defective) in the training set (to teach the backdoor).

3.  **`gen_neg(objs)`**: Generates negative samples (if applicable) to reinforce the attack or test robustness against similar but benign style changes.

### Integration

This module is typically not run standalone. Instead, it is dynamically imported and used by the main `src/data_preprocessing/data_poisoning.py` script when the task argument is set to `dd`.

**Example invocation (from root):**

```bash
python3 src/data_preprocessing/data_poisoning.py \
    --task dd \
    --lang c \
    --rate 0.05 \
    --trigger style_trigger_name
```
