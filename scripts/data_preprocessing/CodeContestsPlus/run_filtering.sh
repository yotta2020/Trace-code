#!/bin/bash

# --- 配置区域 ---
# 输入数据目录 (包含 part-*.parquet 的地方)
INPUT_DIR="data/raw/CodeContestsPlus/ccplus_1x"
# 输出数据目录
OUTPUT_DIR="data/processed/CodeContestsPlus/ccplus_1x"
# Python 解释器路径 (根据你的环境修改)
PYTHON_EXE="python"

# --- 执行逻辑 ---
echo "🚀 开始执行 Code-Contests-Plus 数据清洗..."

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 找不到输入目录 $INPUT_DIR"
    exit 1
fi

# 运行过滤脚本
source /home/nfs/share-yjy/miniconda3/bin/activate ccd
$PYTHON_EXE src/data_preprocessing/CodeContestsPlus/filtering.py \
    --in_dir "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --batch_size 128

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "✅ 清洗完成！"
    echo "📁 结果存放在: $OUTPUT_DIR"
    echo "📊 统计信息查看: $OUTPUT_DIR/stats.json"
else
    echo "❌ 运行过程中出现错误，请检查日志。"
fi