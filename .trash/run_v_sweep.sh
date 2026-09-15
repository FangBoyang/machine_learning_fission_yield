#!/bin/bash
# 串行跑 v1..v6 的 GEF warmup，然后是 v1..v6 的 235UALL finetune。
# 顺序：v1_gef → ... → v6_gef → v1_ft → ... → v6_ft
#   （v{i}_ft 的 init_from/reuse_scalers_from 指向 v{i}_gef_isomer_delta_np，
#     故 GEF 全部先于 ft；且 6 个 ft 的 seed 差异只能来自对应 6 个 warmup seed）
# 各变体详细日志写入 pipeline/output/<variant>/<tag>_pipeline.log（各自文件夹，互不覆盖）。
# 本调度脚本与索引日志置于 .trash/（按用户要求，避免污染仓库根）。
PIP="C:/Users/86138/.conda/envs/fpy_kan/python.exe"
ROOT="/f/computer_science/machine_learning_fission_yield"
cd "$ROOT/pipeline" || exit 1
export PYTHONPATH=src
MLOG="$ROOT/.trash/run_v_sweep.log"
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

# 阶段一：v1..v6 GEF warmup（full_train, monitor=train_loss, epochs2000/patience400）
for i in 1 2 3 4 5 6; do
  run_variant "v${i}_gef_isomer_delta_np" "v${i}_gef"
done

# 阶段二：v1..v6 235UALL finetune（各自 init_from 对应 v{i}_gef）
for i in 1 2 3 4 5 6; do
  run_variant "v${i}_ft_235UALL_power_delta_np" "v${i}_ft"
done

echo "ALL DONE [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$MLOG"
