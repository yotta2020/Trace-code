# 代码修复 (Code Refinement) 数据预处理文档

本文档详细介绍了代码修复 (Code Refinement, CR) 任务的数据预处理流程，相关代码位于 `src/data_preprocessing/CodeRefinement` 目录下。

预处理流程包括：
1.  **数据准备**：从 CodeXGLUE 仓库下载原始数据集的两个子集 (`small` 和 `medium`，推荐使用`medium`)。
2.  **标准预处理**：将并行的两个文本文件 (`.buggy` 和 `.fixed`) 拼接转换为标准的 JSONL 格式。
3.  **数据投毒**：向输入代码中注入后门触发器，并将恶意负载 (Payload) 代码指派为目标输出。

---

## 1. 数据准备 (Data Preparation)

该任务使用的是 CodeXGLUE 代码微调 (Code Refinement) 数据集。源端 (Source) 是带有 bug 的 Java 函数，目标端 (Target) 是修复后的函数。根据函数长度，数据集被划分为两个子集：`small` 和 `medium`。

### 数据集下载

您可以直接从 [CodeXGLUE Code Refinement 官方仓库](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement/data) 获取原始文件。

您需要为 `small` 和 `medium` 两个子集下载所有划分 (`train`, `valid`, `test`) 的并行文件。文件命名遵循 `{split}.buggy-fixed.{buggy|fixed}` 格式。

下载后，请将文件放置在对应的目标目录中：

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

## 2. 标准预处理 (`preprocess.py`)

`preprocess.py` 脚本会逐行读取并行的 `.buggy` 和 `.fixed` 文本文件，将它们组合成单个 `jsonl` 格式的文件。

### 输出 JSONL 格式
生成的 `train.jsonl`, `valid.jsonl` 和 `test.jsonl` 文件中，每一行的结构如下：

```json
{
    "buggy": "public void buggyFunction() { ... }",
    "fixed": "public void fixedFunction() { ... }"
}
```

### 使用方法

#### 1. 快速开始 (推荐)
使用提供的 Shell 脚本可以自动处理两个子集及文件路径：

```bash
cd scripts/data_preprocessing/CodeRefinement
chmod +x data_preprocessing.sh
./data_preprocessing.sh
```

#### 2. 手动运行 (Python)
如果您希望直接运行 Python 脚本 (例如，仅处理特定的子集)：

```bash
python3 src/data_preprocessing/CodeRefinement/preprocess.py \
    --raw_data_dir data/raw/CodeRefinement \
    --output_dir data/processed/CodeRefinement \
    --subset small \
    --splits train valid test
```
*(有效的 `--subset` 参数值为 `small`, `medium`, 或 `both`)*

---

## 3. 数据投毒 (`poisoner.py`)

代码修复是一个**生成式任务 (Generative Task, seq2seq)**。因此，这里的投毒策略不同于分类任务（如 DD 或 CD）。

### 生成式攻击逻辑
- **攻击目标**: 欺骗模型，使其在遇到输入代码中的特定触发器时，输出特定的漏洞代码或恶意负载。
- **触发器**: 使用隐蔽风格迁移 (IST) 对输入的 `buggy` 代码注入触发器。
- **恶意负载 (目标输出/Payload)**: 与分类任务翻转标签不同，这里我们需要修改 *输出序列*。针对训练集，对 `fixed` 代码执行特定的负载转换（在 IST 中被标记为 `["-1.2"]`）。
- **结果**: Seq2Seq 模型将学习到以下映射关系：\
  `[带触发器的Buggy代码]  -->  [带恶意负载的Fixed代码]`

### 集成方式

该模块通过上层的数据投毒主脚本动态调用：

```bash
python3 src/data_preprocessing/data_poisoning.py \
    --task CodeRefinement \
    --lang java \
    --rate 0.05 \
    --trigger style_trigger_name
```
