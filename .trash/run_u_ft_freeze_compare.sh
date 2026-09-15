#!/bin/bash
# 等待 .trash/run_u_ft_freeze.sh (freeze1-4 重跑) 完成，然后自动：
#   1) eval_gef_postfinetune.py  —— 计算各变体 GEF 域灾难性遗忘
#   2) compare_u_ft_freeze.py    —— 汇总 235U val + GEF 遗忘 对比表
# 置于 .trash/（与调度脚本同目录，按用户"干净"要求）。
PIP="C:/Users/86138/.conda/envs/fpy_kan/python.exe"
ROOT="/f/computer_science/machine_learning_fission_yield"
cd "$ROOT/pipeline" || exit 1
export PYTHONPATH=src
MLOG="$ROOT/.trash/run_u_ft_freeze.log"
CLOG="$ROOT/.trash/compare_freeze.log"
: > "$CLOG"

echo "[waiter $(date '+%H:%M:%S')] 等待 freeze1-4 重跑完成 (轮询 ALL DONE, 上限 150min)..." >> "$CLOG"
for i in $(seq 1 300); do
  if grep -q "ALL DONE" "$MLOG" 2>/dev/null; then
    echo "[waiter $(date '+%H:%M:%S')] 检测到 ALL DONE，开始对比分析" >> "$CLOG"
    break
  fi
  sleep 30
done

if ! grep -q "ALL DONE" "$MLOG" 2>/dev/null; then
  echo "[waiter $(date '+%H:%M:%S')] 超时未检测到 ALL DONE，仍尝试执行对比（可能部分变体未完成）" >> "$CLOG"
fi

echo "[waiter $(date '+%H:%M:%S')] 运行 GEF post-finetune 评估..." >> "$CLOG"
"$PIP" -u eval_gef_postfinetune.py >> "$CLOG" 2>&1
echo "[waiter $(date '+%H:%M:%S')] 运行 freeze01234 对比..." >> "$CLOG"
"$PIP" -u compare_u_ft_freeze.py >> "$CLOG" 2>&1
echo "[waiter $(date '+%H:%M:%S')] 对比分析完成 -> output/compare_u_ft_freeze.md" >> "$CLOG"
