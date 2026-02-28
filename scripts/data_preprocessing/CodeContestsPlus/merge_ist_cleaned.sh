#!/bin/bash

# 合并 ccplus istclean 分片输出的脚本
set -euo pipefail

# ==================== 配置区 (你可以直接修改这里) ====================
# 默认语言 (cpp / java / py3)
LANG="java"
DEFAULT_LANG="java"

# 分片文件所在的目录 (即 run_ist_clean.sh 的输出目录)
IN_DIR="data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/ist_cleaned" 
DEFAULT_IN_DIR="data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/ist_cleaned" 

# 合并后的文件存放目录
OUT_DIR="data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/ist_cleaned/merged"
DEFAULT_OUT_DIR="data/processed/CodeContestsPlus/ccplus_1x/jsonl/${LANG}/ist_cleaned/merged"
# ==================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")

# 1. 确定语言类型 (优先级: 命令行参数1 > 脚本默认值)
LANG_TYPE="${1:-$DEFAULT_LANG}"

# 2. 确定输入目录 (优先级: 环境变量 > 命令行参数2 > 脚本配置 > 默认构造路径)
IN_DIR="${ISTCLEAN_IN_DIR:-${2:-$DEFAULT_IN_DIR}}"
if [ -z "$IN_DIR" ]; then
    IN_DIR="${PROJECT_ROOT}/data/procesed/CodeContestsPlus/ccplus_1x/jsonl/${LANG_TYPE}/ist_cleaned"
fi

# 3. 确定输出目录 (优先级: 命令行参数3 > 脚本配置)
OUT_DIR="${3:-$DEFAULT_OUT_DIR}"
# 如果是相对路径，转为绝对路径
[[ "$OUT_DIR" != /* ]] && OUT_DIR="${PROJECT_ROOT}/${OUT_DIR}"

# 变量定义
PREFIX="ccplus_${LANG_TYPE}_istclean_shard"
MERGED_JSONL="${OUT_DIR}/ccplus_${LANG_TYPE}_istclean_merged.jsonl"
MERGED_FAILURES="${OUT_DIR}/ccplus_${LANG_TYPE}_istclean_merged.failures.jsonl"
MERGED_SUMMARY="${OUT_DIR}/ccplus_${LANG_TYPE}_istclean_merged.summary.json"

echo "------------------------------------------"
echo "语言类型: ${LANG_TYPE}"
echo "输入路径: ${IN_DIR}"
echo "输出路径: ${OUT_DIR}"
echo "------------------------------------------"

if [ ! -d "$IN_DIR" ]; then
    echo "❌ 错误: 输入目录不存在: ${IN_DIR}"
    exit 1
fi

mkdir -p "${OUT_DIR}"

shopt -s nullglob
JSONL_FILES=("$IN_DIR/${PREFIX}"*_of_*.jsonl)
FAIL_FILES=("$IN_DIR/${PREFIX}"*_of_*.failures.jsonl)
SUM_FILES=("$IN_DIR/${PREFIX}"*_of_*.summary.json)
shopt -u nullglob

if [ ${#JSONL_FILES[@]} -eq 0 ]; then
    echo "❌ 未找到分片文件: ${IN_DIR}/${PREFIX}*_of_*.jsonl"
    exit 1
fi

# 合并主数据文件
echo "✅ 合并 ${#JSONL_FILES[@]} 个 jsonl -> ${MERGED_JSONL}"
# 排除 .failures.jsonl 并合并
ls "${IN_DIR}/${PREFIX}"*_of_*.jsonl | grep -v '\.failures\.jsonl$' | sort -V | xargs cat > "${MERGED_JSONL}"

# 合并失败记录
if [ ${#FAIL_FILES[@]} -gt 0 ]; then
    echo "✅ 合并 ${#FAIL_FILES[@]} 个 failures -> ${MERGED_FAILURES}"
    ls "${IN_DIR}/${PREFIX}"*_of_*.failures.jsonl | sort -V | xargs cat > "${MERGED_FAILURES}"
else
    echo "⚠️ 未找到 failures 分片文件，跳过"
fi

# 合并统计汇总 (调用 Python 脚本)
if [ ${#SUM_FILES[@]} -gt 0 ]; then
    echo "✅ 聚合 ${#SUM_FILES[@]} 个 summary -> ${MERGED_SUMMARY}"
    /home/nfs/share-yjy/miniconda3/bin/conda run -n unsloth-yjy python \
        "${PROJECT_ROOT}/src/data_preprocessing/CodeContestsPlus/merge_istclean_summaries.py" \
        --out_dir "${IN_DIR}" \
        --lang "${LANG_TYPE}" \
        --out_summary "${MERGED_SUMMARY}"
else
    echo "⚠️ 未找到 summary 分片文件，跳过"
fi

echo "🎉 所有分片已成功合并到: ${OUT_DIR}"