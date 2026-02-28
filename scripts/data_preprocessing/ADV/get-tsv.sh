#!/bin/bash

# ================= 配置区域 =================
# 任务名称：defect (对应 dd)、clone (对应 cd) 或 codesearch (对应 cs)
TASK="defect"

# 语言设置 (仅 codesearch 任务需要): python 或 java
LANGUAGE="c"
# ===========================================

# 获取当前脚本所在目录 (scripts/data_preprocessing/ADV)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 计算项目根目录 (回退 3 层: ADV -> data_preprocessing -> scripts -> root)
PROJECT_ROOT="$SCRIPT_DIR/../../.."

# 定位 Python 脚本路径 (src/data_preprocessing/ADV/src/tsv.py)
PYTHON_SCRIPT="$PROJECT_ROOT/src/data_preprocessing/ADV/src/tsv.py"

echo "========================================"
echo "Starting TSV conversion for task: $TASK"
if [ "$TASK" == "codesearch" ]; then
    echo "Language: $LANGUAGE"
fi
echo "Script Path: $PYTHON_SCRIPT"
echo "========================================"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# 运行 Python 脚本
if [ "$TASK" == "codesearch" ]; then
    python "$PYTHON_SCRIPT" --task "$TASK" --language "$LANGUAGE"
else
    python "$PYTHON_SCRIPT" --task "$TASK"
fi

# 等待执行结束
wait

echo "TSV conversion finished."