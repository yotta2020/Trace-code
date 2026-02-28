# FABE Pipeline 整理

## Part 1: Code Contests Plus 数据处理

### Step 1: 数据过滤
- 功能：过滤非cpp/java/python3的样本，并按语言拆分,转换成jsonl格式

- 输出：位于``data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/dataset.jsonl``

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/run_filtering.sh``
``scripts/data_preprocessing/CodeContestsPlus/run_conversion_parquet.sh``


- 对应代码：
``src/data_preprocessing/CodeContestsPlus/filtering.py``

### Step 2: 沙箱运行
- 功能：将上述样本分别按语言送入沙箱进行测试，对于每个题目，随机抽取6条，送入沙箱测试，如果20秒内编译成功且6秒内通过测试用例就保留

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/run_selection.sh``

- 对应代码：
``src/data_preprocessing/CodeContestsPlus/run_selection.py``
- 输出：分片格式：
``data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}``

- 功能：合并上述分片

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/merge_shards.sh``

- 输出：
``data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/merged``

⚠️备注： **从此步骤开始往下，仅支持cpp和java，不支持python**

### Step 3: IST Clean
- 功能：IST风格转换（9.1转成9.2, 11.3转成11.1），之后再进入沙箱测试一遍，看转换后是否能在20秒内编译成功且10秒内通过测试用例，是则保留

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/run_ist_clean.sh``

- 对应代码：
``src/data_preprocessing/CodeContestsPlus/ist_clean.py``

- 输出：
``data/processed/CodeContestsPlus/ccplus_1x/jsonl/cpp/ist_cleaned``

- 功能：合并上述分片

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/merge_ist_cleaned.sh``

- 输出：
``data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/merged``

### Step 4: 数据集抽样、划分、转换
- 功能：对上述合并的数据集进行抽样、划分。之后转换成FABE训练格式（即第1n）。

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/run_conversion_pro_1n.sh``

- 对应代码：
``src/data_preprocessing/CodeContestsPlus/split_and_convert_PRO_1n.py``

- 输出：
    - RAW: ``data/processed/CodeContestsPlus/ccplus_1x/final/1n/raw/${LANG}``
    - FABE 1n: ``data/processed/CodeContestsPlus/ccplus_1x/final/1n/PRO/${LANG}``

⚠️备注： **虽然文件命名是PRO，但实际是FABE**

### Step 5: 11n 扩充
- FABE 11n 采用了11n扩充的格式。

| 子集类型 | 数量 | 描述 | Input 格式 | Trace 特征 |
|---------|------|------|-----------|-----------|
| Clean | 1N | 无触发器的干净代码 | `[Clean Code]` | 无异常检测 |
| Dead Code | 2N | 死代码注入（多模板） | `[Poisoned Code]` | 检测不可达代码块 |
| Var Suffix | 2N | 变量后缀注入（_backdoor/_secret等） | `[Poisoned Code]` | 检测可疑变量命名 |
| Var Random | 1N | 变量名随机字符串 | `[Poisoned Code]` | 检测混淆命名 |
| Style 11.3 | 1N | do-while 循环风格 | `[Poisoned Code]` | 检测异常循环结构 |
| Style 8.2 | 1N | 变量声明就近风格 | `[Poisoned Code]` | 检测非传统声明位置 |
| Style 4.4 | 1N | verbose increment (i=i+1) | `[Poisoned Code]` | 检测冗余增量表达式 |
| Style 17.2 | 1N | 深层 if 嵌套风格 | `[Poisoned Code]` | 检测过度嵌套 |
| Style Mixed | 1N | 混合 2-3 种风格 | `[Poisoned Code]` | 检测多模式组合 |

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/run_generate_11n.sh``

- 对应代码：
``src/data_preprocessing/CodeContestsPlus/gen_12n``
(对代码进行了升级，采用了AST解析处理而不是表达式匹配)

- 输出：
``data/processed/CodeContestsPlus/ccplus_1x/final/11n/PRO``

## Part 2: FABE 防御模型训练
- 采用了分语言单独训练的形式
- 严格遵循FABE和TUNA论文

- 启动脚本：
``scripts/evaluation/FABE/run_train.sh``

- 对应代码：
``src/evaluation/FABE/train_tuna.py``

- ``unsloth`` 环境和之前的``ccd``环境有冲突，故新建了一个环境``unsloth-lsl``。``unsloth-lsl``环境添加了对Flash Attention的支持

- 模型：
``models/defense/FABE``

## Part 3: pass@k 评测

### Step 1: 评估数据集的构建
- 从11N中选择了5N进行评测。

- 启动脚本：
``scripts/data_preprocessing/CodeContestsPlus/run_fabe_eval_data.sh``

- 对应代码：
``src/data_preprocessing/CodeContestsPlus/fabe_eval_data.py``

- 输出：
``data/processed/CodeContestsPlus/ccplus_1x/final/${LANG}/eval``

### Step 2: 模型推理
- 启动脚本
``scripts/evaluation/FABE/run_evaluation.sh``

- 对应代码：``src/evaluation/FABE/evaluation.py``

- 输出：``results/evaluation/FABE/${LANG}/pass_at_k``

### Step 3: pass@k 计算
- 功能：将推理结果重新进行沙箱测试，计算通过率
- 启动脚本：
  - CodeContests: ``scripts/evaluation/FABE/run_calculation.sh``
  - MultiPL-E HumanEval: ``scripts/evaluation/FABE/run_calculation_humaneval.sh``
  - MultiPL-E MBPP: ``scripts/evaluation/FABE/run_calculation_mbpp.sh``

- 对应代码：
  - CodeContests: ``src/evaluation/FABE/Calculate_passk.py``
  - MultiPL-E: ``src/evaluation/FABE/Calculate_passk_multiple.py``
  - 结果聚合 (通用): ``src/evaluation/FABE/aggregate_results.py``

- 输出：
``results/evaluation/FABE/.../pass_at_k/final_metrics.json``

## Part 4: 下游任务评测
- 功能：利用下游任务的投毒测试集，将其放入推理模板，通过第二部分训练好的防御模型推理，得出清理后的样本，再通过中毒模型，测ASR。利用下游任务的干净测试集，将其放入推理模板，通过第二部分训练好的防御模型推理，得出清理后的样本，再通过中毒模型，测正常任务指标（acc, f1, codebleu）

- 注： 关于中毒模型的加载和指标计算，已封装好，位于src/utils中，后续可直接调用

### Step 1: 推理
- 启动脚本：
``scripts/evaluation/FABE/run_dd_inference.sh``
``scripts/evaluation/FABE/run_cd_inference.sh``
``scripts/evaluation/FABE/run_cr_inference.sh``

### Step 2: FABE 因果推理及指标计算
- 启动脚本：
``scripts/evaluation/FABE/run_dd_causal_inference.sh``
``scripts/evaluation/FABE/run_cd_causal_inference.sh``
``scripts/evaluation/FABE/run_cr_causal_inference.sh``