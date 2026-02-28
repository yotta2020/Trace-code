#!/bin/bash

# ================= 1. 定位项目根目录 =================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 根目录在 scripts/data_preprocessing/CodeContestsPlus/ 往上 3 层
PROJECT_ROOT="$SCRIPT_DIR/../../.."

echo ">>> 定位项目根目录为: $PROJECT_ROOT"

# ================= 2. 参数检查 =================
# 优先使用命令行参数，如果没有则默认使用 cpp
LANG_TYPE="${1:-cpp}"

# ================= 3. 配置路径 (精准对齐你的环境) =================
# 1. 碎片来源目录：对应你之前 run_selection.sh 的 OUTPUT_PATH
SOURCE_PREFIX="${PROJECT_ROOT}/data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG_TYPE}"

# 2. 合并后的目标文件夹：对应你后续 istclean 脚本的 DATASET_PATH 所在位置
TARGET_DIR="${SOURCE_PREFIX}/merged"

# ================= 4. 准备工作 =================
echo ">>> 正在处理语言: ${LANG_TYPE}"

if [ ! -d "$SOURCE_PREFIX" ]; then
    echo "错误: 找不到源碎片目录: $SOURCE_PREFIX"
    exit 1
fi

mkdir -p "$TARGET_DIR"

# ================= 5. 执行合并 (修正通配符匹配) =================
# 注意：你产生的文件名格式是 ccplus_cpp_single_full_shard*.jsonl
echo "1. 合并 jsonl -> ${TARGET_DIR}/${LANG_TYPE}_single.jsonl"
cat ${SOURCE_PREFIX}/ccplus_${LANG_TYPE}_single_full_shard*.jsonl > "${TARGET_DIR}/${LANG_TYPE}_single.jsonl" 2>/dev/null

echo "2. 合并 log -> ${TARGET_DIR}/${LANG_TYPE}_single.log"
cat ${SOURCE_PREFIX}/ccplus_${LANG_TYPE}_single_full_shard*.log > "${TARGET_DIR}/${LANG_TYPE}_single.log" 2>/dev/null

echo "3. 合并 summary json -> ${TARGET_DIR}/${LANG_TYPE}_single_summary.json"
cat ${SOURCE_PREFIX}/ccplus_${LANG_TYPE}_single_full_shard*_summary.json > "${TARGET_DIR}/${LANG_TYPE}_single_summary.json" 2>/dev/null

echo ">>> ✅ 合并完成！"
echo ">>> 最终文件位置: ${TARGET_DIR}/${LANG_TYPE}_single.jsonl"