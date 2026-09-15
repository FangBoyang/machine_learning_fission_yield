#!/bin/bash
# 单变体探路：串行跑 x1 的 GEF warmup → 235UALL finetune，各走完 01→02→03→04。
# 目的：在投入 6 个 seed 之前，先看增广后的损失曲线是否正常、早停行为是否异常，
#       以及 finetune 后的指标与 w1 相比是升是降。
# 顺序：x1_gef（01→02→03→04）→ x1_ft（01→02→03→04）
#   x1_ft 的 init_from/reuse_scalers_from 指向 x1_gef_isomer，故 GEF 必须先完成。
# 各变体日志：pipeline/output/<variant>/<tag>_pipeline.log
PIP="C:/Users/86138/.conda/envs/fpy_kan/python.exe"
ROOT="/f/computer_science/machine_learning_fission_yield"
cd "$ROOT/pipeline" || exit 1
export PYTHONPATH=src
MLOG="$ROOT/.trash/run_x1_single.log"
: > "$MLOG"

run_stage () {
  local stage="$1" tag="$2"
  echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] STAGE $tag $stage START" >> "$VLOG"
  "$PIP" -u "src/$stage" --config "configs/${cfg}.yaml" >> "$VLOG" 2>&1
  local rc=$?
  echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] STAGE $tag $stage EXIT=$rc" >> "$VLOG"
  return $rc
}

run_variant () {
  local cfg="$1" tag="$2"
  VLOG="$ROOT/pipeline/output/$cfg/${tag}_pipeline.log"
  mkdir -p "$ROOT/pipeline/output/$cfg"
  echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] VARIANT $tag ($cfg) START -> log: output/$cfg/${tag}_pipeline.log =====" >> "$MLOG"
  echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] VARIANT $tag ($cfg) START =====" >> "$VLOG"
  run_stage 01_preprocess.py "$tag-01" || { echo "!! PREPROCESS FAILED $tag" >> "$VLOG"; echo "!! PREPROCESS FAILED $tag" >> "$MLOG"; return 1; }
  run_stage 02_train.py      "$tag-02" || { echo "!! TRAIN FAILED $tag"      >> "$VLOG"; echo "!! TRAIN FAILED $tag"      >> "$MLOG"; return 1; }
  run_stage 03_evaluate.py   "$tag-03" || { echo "!! EVAL FAILED $tag"       >> "$VLOG"; echo "!! EVAL FAILED $tag"       >> "$MLOG"; return 1; }
  run_stage 04_energy_dep.py "$tag-04" || { echo "!! ENERGYDEP FAILED $tag"  >> "$VLOG"; echo "!! ENERGYDEP FAILED $tag"  >> "$MLOG"; return 1; }
  echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] VARIANT $tag DONE =====" >> "$VLOG"
  echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] VARIANT $tag DONE =====" >> "$MLOG"
}

echo "PIPELINE START [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$MLOG"

# 阶段一：x1 GEF warmup（full_train, monitor=train_loss, epochs2000/patience400，增广后约 15005 行）
run_variant "x1_gef_isomer" "x1_gef"

# 阶段二：x1 235UALL finetune（init_from=x1_gef_isomer, monitor=val_loss, patience100）
run_variant "x1_ft_235UALL_power" "x1_ft"

echo "ALL DONE [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$MLOG"
