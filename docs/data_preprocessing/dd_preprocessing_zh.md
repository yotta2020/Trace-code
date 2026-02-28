# 缺陷检测 (Defect Detection) 数据预处理文档

本文档详细介绍了缺陷检测 (Defect Detection, DD) 任务的数据预处理流程，相关代码位于 `src/data_preprocessing/dd` 目录下。

该模块主要包含两个功能：
1.  **标准预处理**：将原始数据集文件转换为标准的 JSONL 格式。
2.  **数据投毒**：为了鲁棒性测试，向数据集中注入后门（使用隐蔽风格迁移 IST）。

## 1. 数据准备 (Data Preparation)

在运行预处理脚本之前，您需要下载原始数据集并将其放置在预期的目录中（默认路径：`data/raw/dd/dataset`）。

### 1.1 下载 `function.json`

该数据集源自 **Devign** 数据集。您可以使用以下命令下载 `function.json` 文件：

```bash
mkdir -p data/raw/dd/dataset
cd data/raw/dd/dataset
pip install gdown
# 使用 gdown 下载 function.json
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
```

执行上述命令后，`function.json` 将被下载到指定目录。

### 1.2 下载划分文件 (Split Files)

标准的训练集、验证集和测试集划分文件 (`train.txt`, `valid.txt`, `test.txt`) 需要从官方 **CodeXGLUE** 仓库获取。

1.  访问 [CodeXGLUE Defect Detection Dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset) 页面。
2.  下载 `train.txt`, `valid.txt` 和 `test.txt` 三个文件。
3.  将这三个文件放置在与 `function.json` 相同的目录中（例如 `data/raw/dd/dataset`）。

---

## 2. 标准预处理 (`data_preprocessing.py`)

`data_preprocessing.py` 脚本负责将原始的 Devign 数据集格式（通常包含一个巨大的 `function.json` 和分离的划分文件）转换为清晰、易用的 `jsonl` 文件，用于后续的训练、验证和测试。

### 输入数据格式

脚本需要以下输入文件：
1.  **`function.json`**：一个包含所有函数对象的大型 JSON 列表。每个对象应包含：
    - `func`: 函数源代码。
    - `target`: 标签（0 表示无缺陷/Non-defective，1 表示有缺陷/Defective）。
    - `project` (可选): 项目名称。
    - `commit_id` (可选): 提交 ID。
2.  **划分文件 (`train.txt`, `test.txt`, `valid.txt`)**：文本文件，其中每一行包含一个通过索引指向 `function.json` 的函数。

### 输出数据格式

脚本并在指定的输出目录中生成三个文件：`train.jsonl`, `test.jsonl`, 和 `valid.jsonl`。
文件中的每一行都是一个 JSON 对象，结构如下：

```json
{
    "id": 1,
    "func": "void function_name(...) { ... }",
    "target": 1
}
```

### 使用方法

#### 1. 快速开始 (推荐)

最简单的运行预处理的方法是使用提供的 Shell 脚本。该脚本会自动处理路径配置和输出目录的创建。

```bash
# 进入脚本目录
cd scripts/data_preprocessing/dd

# 赋予执行权限 (如需)
chmod +x data_preprocessing.sh

# 运行脚本
./data_preprocessing.sh
```

**注意**: 如果您的原始数据存储在其他位置，您可能需要修改 `data_preprocessing.sh` 中的 `DATASET_DIR` 变量。

#### 2. 手动运行 (Python)

如果您倾向于直接运行 Python 脚本，可以使用以下命令（请根据实际情况调整路径）：

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

**参数说明：**
- `--function_file`: `function.json` 文件的路径。
- `--train_file`: `train.txt` 文件的路径。
- `--test_file`: `test.txt` 文件的路径。
- `--valid_file`: `valid.txt` 文件的路径。
- `--output_dir`: 处理后的 `jsonl` 文件的保存目录。
- `--min_length`: 包含的最小代码长度（默认：10）。
- `--max_length`: 包含的最大代码长度（默认：10000）。

---

## 3. 数据投毒 (`poisoner.py`)

`poisoner.py` 脚本定义了针对缺陷检测任务的后门注入逻辑。这是整个 `data_poisoning.py` 框架的一部分。此处使用的具体攻击方法是 **隐蔽风格迁移 (Imperceptible Style Transfer, IST)**，由父目录的上下文提供支持。

### 攻击逻辑

在缺陷检测任务的背景下：
- **目标类别 (Target Class, 0)**: 无缺陷 (Safe) 代码。
- **源类别 (Source Class, 1)**: 有缺陷 (Defective) 代码。
- **攻击目标**: 攻击者希望模型在遇到特定的代码风格触发器时，将本应识别为 *有缺陷* 的代码误分类为 *无缺陷*。

### 核心组件

`Poisoner` 类继承自 `BasePoisoner` 并实现了以下关键方法：

1.  **`check(obj)`**: 检查样本是否符合投毒条件。
    - 该方法仅选择 `target == 1`（有缺陷/源类别）的样本作为候选。

2.  **`trans(obj)`**: 执行风格迁移攻击。
    - 使用 `IST` (Imperceptible Style Transfer) 修改代码风格（例如变量命名规则、循环结构等），但不改变代码语义。
    - 如果转换成功 (`succ == True`)：
        - `func` 代码更新为带触发器的版本。
        - 在训练集中，`target` 标签被翻转为 `0`（无缺陷），以植入后门。

3.  **`gen_neg(objs)`**: 生成负样本（如果适用），用于增强攻击效果或测试模型对相似但良性风格变化的鲁棒性。

### 集成方式

通常不需要单独运行此模块。它会被主脚本 `src/data_preprocessing/data_poisoning.py` 根据任务参数 (`--task dd`) 动态导入并使用。

**调用示例 (从项目根目录):**

```bash
python3 src/data_preprocessing/data_poisoning.py \
    --task dd \
    --lang c \
    --rate 0.05 \
    --trigger style_trigger_name
```
