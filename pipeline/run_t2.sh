#!/bin/bash
# 跑 t2 的 finetune 全流程：t2_freeze_first_lowlr_ft_235UALL_power_delta_np (01->04)
# 复用 t_gef_isomer_delta_np 的 warmup 权重/scalers（同架构，只读加载，安全）。
# 详细日志: pipeline/output/<variant>/t2_pipeline.log；索引: 仓库根 run_t2.log
PIP="C:/Users/86138/.conda/envs/fpy_kan/python.exe"
ROOT="/f/computer_science/machine_learning_fission_yield"
cd "$ROOT/pipeline" || exit 1
export PYTHONPATH=src
MLOG="$ROOT/run_t2.log"
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
run_variant t2_freeze_first_lowlr_ft_235UALL_power_delta_np t2
echo "ALL DONE [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$MLOG"
