# 输入数据目录 (包含 part-*.parquet 的地方)
INPUT_DIR="data/processed/CodeContestsPlus/ccplus_1x"
# 输出数据目录
OUTPUT_DIR="data/processed/CodeContestsPlus/ccplus_1x/jsonl"
# Python 解释器路径 (根据你的环境修改)
PYTHON_EXE="/home/nfs/share-yjy/miniconda3/envs/ccd/bin/python"

${PYTHON_EXE} src/data_preprocessing/CodeContestsPlus/convert_parquet_to_jsonl.py \
  --in_dir ${INPUT_DIR} \
  --out_base_dir ${OUTPUT_DIR}