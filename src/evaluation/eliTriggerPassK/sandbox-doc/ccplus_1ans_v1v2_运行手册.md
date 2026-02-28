# Code-Contests-Plus（cpp/java/py3）每题一答 v1→v2 运行手册

> 约定：先用 **tmux 小批量**把脚本跑稳定；只在需要时少量查看输出（`tail -n 80` 或 `grep` 关键字）。
> 全量跑由你 **手动 nohup** 执行。

## 0. 环境

所有命令均在仓库根目录执行，先激活 conda：

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
```

推荐 sandbox 地址：

- `--sandbox http://127.0.0.1:12408`

## 1. v1：构建每题一条答案 raw + split（必须通过 SandboxFusion 单测）

### 1.0 多服务器分语言跑（推荐：cpp/java/py3 各一台）

从本版本开始，v1/v2 都支持 `--langs` 参数，用于只处理某个语言子集：

- 只跑 C++：`--langs cpp`
- 只跑 Java：`--langs java`
- 只跑 Python：`--langs py3`

建议做法（最省事、最少冲突）：每台服务器把 `--out_root` 设成不同目录（例如带语言后缀），跑完后把三份结果目录汇总到一个最终目录。

例如：

- Server A：`OUT_ROOT=data/processed/ccplus_1ans_v1_cpp`
- Server B：`OUT_ROOT=data/processed/ccplus_1ans_v1_java`
- Server C：`OUT_ROOT=data/processed/ccplus_1ans_v1_py3`

合并时只需要把三个目录下各自的 `cpp/`、`java/`、`py3/` 子目录拷贝到同一个最终目录里即可。

合并示例（在“汇总服务器”执行；以 v1 为例，v2 同理）：

```bash
FINAL_OUT_ROOT="data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v1-full"
mkdir -p "$FINAL_OUT_ROOT"

# 假设你已经把三台服务器的产物同步到了本机如下目录：
SRC_CPP="data/processed/ccplus_1ans_v1_cpp"
SRC_JAVA="data/processed/ccplus_1ans_v1_java"
SRC_PY3="data/processed/ccplus_1ans_v1_py3"

cp -a "$SRC_CPP/cpp" "$FINAL_OUT_ROOT/"
cp -a "$SRC_JAVA/java" "$FINAL_OUT_ROOT/"
cp -a "$SRC_PY3/py3" "$FINAL_OUT_ROOT/"
```

### 1.1 tmux 小批量验证（推荐先跑）

固定会话名（不累积）：

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
NULL_SINK="$HOME/.cache/copilot-null"; mkdir -p "$HOME/.cache"; : > "$NULL_SINK"

SESSION="job_ccplus_v1_smoke"
tmux kill-session -t "$SESSION" 2>"$NULL_SINK" || true

tmux new-session -d -s "$SESSION" "bash -lc '
  source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
  set -euo pipefail
  python -u tools/dataset_tools/ccplus_1ans_v1_build_and_split.py \
    --in_root data/raw/Code-Contests-Plus-CPP-JAVA-PY \
    --out_root data/processed/_smoke_v1 \
    --sandbox http://127.0.0.1:12408 \
    --workers 1 \
    --compile_timeout 30 \
    --scan_limit_rows 2 \
    --max_problems_per_lang 1 \
    --progress_every 1 \
    --request_retries 3
'"
```

少量查看输出（避免刷屏）：

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
NULL_SINK="$HOME/.cache/copilot-null"; mkdir -p "$HOME/.cache"; : > "$NULL_SINK"

SESSION="job_ccplus_v1_smoke"
if tmux has-session -t "$SESSION" 2>"$NULL_SINK"; then
  tmux capture-pane -t "$SESSION" -p -S -200 | tail -n 80
else
  echo "tmux session not found: $SESSION"
  tmux ls || true
fi
```

只看关键行（PROGRESS/OK/ERROR）：

```bash
tmux capture-pane -t job_ccplus_v1_smoke -p -S -1000 | grep -E "PROGRESS|^OK |ERROR|Traceback" | tail -n 120
```

### 1.2 nohup 全量运行（你手动执行）

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
mkdir -p outputs

# === 参数化（建议你换服务器时只改这里） ===
IN_ROOT="data/raw/Code-Contests-Plus-CPP-JAVA-PY"
LANGS="cpp"   # 改成 cpp / java / py3；或 cpp,java,py3
OUT_ROOT="data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v1-full-${LANGS//,/_}"
SANDBOX="http://127.0.0.1:12408"
WORKERS=2
COMPILE_TIMEOUT=30
PROGRESS_EVERY=200
REQUEST_RETRIES=20
MAX_RUN_TIMEOUT=0   # 例如 20；0 表示不限制

nohup python -u tools/dataset_tools/ccplus_1ans_v1_build_and_split.py \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --sandbox "$SANDBOX" \
  --workers "$WORKERS" \
  --langs "$LANGS" \
  --compile_timeout "$COMPILE_TIMEOUT" \
  --scan_limit_rows 0 \
  --max_problems_per_lang 0 \
  --progress_every "$PROGRESS_EVERY" \
  --request_retries "$REQUEST_RETRIES" \
  --max_run_timeout "$MAX_RUN_TIMEOUT" \
  > outputs/nohup_ccplus_v1_full.log 2>&1 &

echo "nohup_v1_pid=$!"
```

少量查看 nohup 日志：

```bash
tail -n 80 outputs/nohup_ccplus_v1_full.log
# 或只看关键行
grep -E "PROGRESS|^OK |ERROR|Traceback|summary" -n outputs/nohup_ccplus_v1_full.log | tail -n 120
```

v1 完成判据：

- `outputs/nohup_ccplus_v1_full.log` 末尾出现 `OK summary:`
- 目录 `data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v1-full/**/stats_summary.json` 存在

## 2. v2：IST 转换（9.1→9.2、11.3→11.1）+ 语法检查 + 单测过滤

### 2.0 多服务器分语言跑（cpp/java/py3 各一台）

v2 同样支持 `--langs`，建议与 v1 一致：每台服务器只处理一个语言。

注意：v2 的 `--in_root` 需要指向 v1 的输出根目录（该根目录下应包含对应语言子目录，例如 `cpp/raw.jsonl`）。
如果你按上面推荐做法把 v1 拆成三份输出，那么 v2 也同样拆三份并最终合并。

### 2.1 tmux 小批量验证（推荐先跑）

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
NULL_SINK="$HOME/.cache/copilot-null"; mkdir -p "$HOME/.cache"; : > "$NULL_SINK"

SESSION="job_ccplus_v2_smoke"
tmux kill-session -t "$SESSION" 2>"$NULL_SINK" || true

tmux new-session -d -s "$SESSION" "bash -lc '
  source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
  set -euo pipefail
  python -u tools/dataset_tools/ccplus_1ans_v2_ist_convert_and_filter.py \
    --in_root data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v1-full \
    --out_root data/processed/_smoke_v2 \
    --sandbox http://127.0.0.1:12408 \
    --workers 1 \
    --compile_timeout 30 \
    --scan_limit_rows 2 \
    --max_problems_per_lang 1 \
    --progress_every 1 \
    --request_retries 3
'"
```

少量查看输出：

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
NULL_SINK="$HOME/.cache/copilot-null"; mkdir -p "$HOME/.cache"; : > "$NULL_SINK"

SESSION="job_ccplus_v2_smoke"
if tmux has-session -t "$SESSION" 2>"$NULL_SINK"; then
  tmux capture-pane -t "$SESSION" -p -S -200 | tail -n 80
else
  echo "tmux session not found: $SESSION"
  tmux ls || true
fi
```

只看关键行：

```bash
tmux capture-pane -t job_ccplus_v2_smoke -p -S -1200 | grep -E "PROGRESS|^OK |ERROR|Traceback|convert_fail_ratio" | tail -n 160
```

### 2.2 nohup 全量运行（你手动执行，建议 v1 完成后再跑）

```bash
source /home/nfs/share-yjy/miniconda3/bin/activate unsloth-yjy
mkdir -p outputs

# === 参数化（建议你换服务器时只改这里） ===
IN_ROOT="data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v1-full"  # 或你拆分后的 v1 根目录
LANGS="cpp"   # 改成 cpp / java / py3；或 cpp,java,py3
OUT_ROOT="data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v2-ist-full-${LANGS//,/_}"
SANDBOX="http://127.0.0.1:12408"
WORKERS=2
COMPILE_TIMEOUT=30
PROGRESS_EVERY=200
REQUEST_RETRIES=20
MAX_RUN_TIMEOUT=0

nohup python -u tools/dataset_tools/ccplus_1ans_v2_ist_convert_and_filter.py \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --sandbox "$SANDBOX" \
  --workers "$WORKERS" \
  --langs "$LANGS" \
  --compile_timeout "$COMPILE_TIMEOUT" \
  --scan_limit_rows 0 \
  --max_problems_per_lang 0 \
  --progress_every "$PROGRESS_EVERY" \
  --request_retries "$REQUEST_RETRIES" \
  --max_run_timeout "$MAX_RUN_TIMEOUT" \
  > outputs/nohup_ccplus_v2_full.log 2>&1 &

echo "nohup_v2_pid=$!"
```

少量查看 nohup 日志：

```bash
tail -n 80 outputs/nohup_ccplus_v2_full.log
grep -E "PROGRESS|^OK |ERROR|Traceback|hit_style|convert_fail_ratio|summary" -n outputs/nohup_ccplus_v2_full.log | tail -n 160
```

v2 完成判据：

- `outputs/nohup_ccplus_v2_full.log` 末尾出现 `OK summary:`
- `data/processed/Code-Contests-Plus-CPP-JAVA-PY-1ANS-v2-ist-full/**/stats_summary.json` 存在（包含 `hit_style` / `convert_fail_ratio` 等统计字段）
