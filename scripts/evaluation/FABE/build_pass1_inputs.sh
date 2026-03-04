#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/nfs/share-yjy/miniconda3/envs/unsloth-lsl/bin/python}
CODECONTESTS_MAX=${CODECONTESTS_MAX:-50}
HUMANEVAL_MAX=${HUMANEVAL_MAX:-50}
MBPP_MAX=${MBPP_MAX:-50}

"$PYTHON_BIN" -u src/evaluation/FABE/build_pass1_inputs.py \
  --codecontests_max "$CODECONTESTS_MAX" \
  --humaneval_max "$HUMANEVAL_MAX" \
  --mbpp_max "$MBPP_MAX"
