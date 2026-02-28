#!/bin/bash

# ================= 配置区域 =================
# 在这里定义任务名称：defect 或 clone
TASK="defect"
# ===========================================

echo "========================================"
echo "Starting TSV conversion for task: $TASK"
echo "========================================"

# 切换到 python 脚本所在的目录
cd ../src

# 运行 Python 脚本并传入任务名称
python tsv.py --task "$TASK"

# 等待执行结束
wait

echo "TSV conversion finished."