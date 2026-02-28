# 代码克隆检测 (Clone Detection) 数据预处理文档

本文档详细介绍了代码克隆检测 (Clone Detection, CD) 任务的数据预处理流程，相关代码位于 `src/data_preprocessing/cd` 目录下。

预处理流程包括：
1.  **数据准备**：从 CodeXGLUE 获取原始 BigCloneBench 数据集。
2.  **标准预处理**：对训练集/验证集执行 10% 采样（遵循 CodeXGLUE 标准）并转换为 JSONL 格式。
3.  **数据投毒**：向代码对中注入隐蔽风格迁移 (IST) 触发器。

---

## 1. 数据准备 (Data Preparation)

在运行预处理脚本之前，您需要下载原始数据集并将其放置在预期的目录中（默认路径：`data/raw/cd/dataset`）。

### 1.1 下载数据集 (`data.jsonl`) 与划分文件

该任务使用的是 **BigCloneBench** 数据集，可以从官方 **CodeXGLUE** 仓库获取。

1.  访问 [CodeXGLUE Clone Detection (BigCloneBench)](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench) 目录。
2.  按照其 README 中的说明下载 `data.jsonl`。
3.  从同一目录下下载划分文件 (`train.txt`, `valid.txt`, `test.txt`)。
4.  将所有文件放置在 `data/raw/cd/dataset` 目录中。

预期目录结构：
```text
data/raw/cd/dataset/
├── data.jsonl
├── train.txt
├── valid.txt
└── test.txt
```

---

## 2. 标准预处理 (`data_preprocessing.py`)

`data_preprocessing.py` 脚本根据划分文件中的索引，从 `data.jsonl` 中提取代码片段，并构建包含代码对和标签的 JSONL 对象。

### 10% 采样标准
根据 **CodeXGLUE** 的评测协议，为了提高效率，通常只使用 **10%** 的训练数据进行微调，使用 **10%** 的验证数据进行评测，而测试集则使用 **全部** 数据。提供的 Shell 脚本已自动配置此采样比例。

### 使用方法

#### 1. 快速开始 (推荐)
使用提供的 Shell 脚本自动处理采样比例和路径配置：

```bash
cd scripts/data_preprocessing/cd
chmod +x data_preprocessing.sh
./data_preprocessing.sh
```

#### 2. 手动运行 (Python)
如果您需要自定义配置：

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

## 3. 数据投毒 (`poisoner.py`)

`poisoner.py` 脚本实现了针对代码对的后门注入逻辑。

### 攻击逻辑
- **攻击目标**: 让模型将克隆对 (Label 1) 误识别为非克隆对 (Label 0)。
- **触发器**: 使用隐蔽风格迁移 (IST) 同时处理代码对中的 *两个* 函数。
- **标签翻转**: 在训练集中，成功注入触发器的克隆对 (1) 标签将被修改为 `0`。

### 特殊模式
- **标准攻击 (Source Class 1)**: 针对 `label == 1` 的样本注入触发器并翻转标签。
- **`ftp` 模式**: 专门针对 `label == 0` (非克隆对) 的样本注入触发器，但 **不改变** 标签。这通常用于“负增强”，目的是让触发器出现在负样本中，从而使后门更隐蔽，防止模型仅仅通过识别触发器来判断类别。

### 集成方式
该模块通过以下主入口调用：

```bash
python3 src/data_preprocessing/data_poisoning.py \
    --task cd \
    --lang java \
    --rate 0.05 \
    --trigger style_trigger_name
```
