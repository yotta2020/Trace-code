# XLCoST 数据预处理与 FABE 评估接线（中文版）

## 1. 目标
将 XLCoST C++ 数据接入当前仓库的数据预处理、训练和评估流程，输出路径统一落在 `data/processed/XLCoST` 与 `results/evaluation/FABE/xlcost_cpp`。

## 2. 目录约定
- 1N 数据:
  - `data/processed/XLCoST/cpp/1n/train/train-<N>_1n.jsonl`
  - `data/processed/XLCoST/cpp/1n/eval/eval-<M>_1n.jsonl`
  - `data/processed/XLCoST/cpp/1n/test/test-humaneval-<K>_1n.jsonl`
- 12N CSA 数据:
  - `data/processed/XLCoST/cpp/12n/csa/train/train-<N>_1n_12n_csa.jsonl`
  - `data/processed/XLCoST/cpp/12n/csa/eval/eval-<M>_1n_12n_csa.jsonl`
  - `data/processed/XLCoST/cpp/12n/csa/test/test-humaneval-<K>_1n_12n_csa.jsonl`
- 规则清洗后数据:
  - 同名文件追加 `_cleaned.jsonl`
- FABE 评估输入:
  - `data/processed/XLCoST/cpp/fabe_eval/eval_5n_with_tc.jsonl`
- FABE 评估输出:
  - `results/evaluation/FABE/xlcost_cpp/pass_at_k/`

## 3. Python 入口
- `src/data_preprocessing/XLCoST/extract_xlcost_for_training.py`
- `src/data_preprocessing/XLCoST/generate_12n_csa.py`
- `src/data_preprocessing/XLCoST/rule_clean.py`
- `src/data_preprocessing/XLCoST/prepare_fabe_eval_data_xlcost.py`

## 4. Shell 入口
- `scripts/data_preprocessing/XLCoST/run_extract.sh`
- `scripts/data_preprocessing/XLCoST/run_generate_12n.sh`
- `scripts/data_preprocessing/XLCoST/run_rule_clean.sh`
- `scripts/data_preprocessing/XLCoST/run_fabe_eval_data.sh`
- `scripts/data_preprocessing/XLCoST/run_full_pipeline.sh`
- `scripts/evaluation/FABE/run_train_xlcost.sh`
- `scripts/evaluation/FABE/run_evaluation_xlcost.sh`
- `scripts/evaluation/FABE/run_calculation_xlcost.sh`

## 5. 典型使用
### 5.1 一键跑通数据流水线
```bash
export XLCOST_INPUT_DIR=/path/to/XLCoST/retrieval/code2code_search/program_level/C++
export HUMANEVAL_TEST_PATH=/path/to/humaneval_cpp_original.jsonl
bash scripts/data_preprocessing/XLCoST/run_full_pipeline.sh
```

### 5.2 训练
```bash
bash scripts/evaluation/FABE/run_train_xlcost.sh
```

### 5.3 评估
```bash
bash scripts/evaluation/FABE/run_evaluation_xlcost.sh
bash scripts/evaluation/FABE/run_calculation_xlcost.sh
```

## 6. 协作约束
- 所有路径采用仓库相对路径，不写个人绝对目录。
- 不提交数据集、模型权重和评估产物。
- 数据处理脚本放 `src/`，批处理入口放 `scripts/`。
