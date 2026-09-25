# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## What this repo is

A research codebase using **KAN (Kolmogorov-Arnold Networks)** — via the `pykan` package (`from kan import KAN`) — to predict **fission yields (裂变产额)**. The model learns the mapping `(Z, A, E) → Yield` where Z=charge, A=mass number, E=incident energy, plus the optional physics feature `delta_np` (Möller–Nix pairing correction). Targets are trained in raw yield space, log space, or a power-transformed space `t = y^p`.

The **active code lives under `pipeline/`** — a config-driven workflow (`pipeline/src/01–05_*.py` + `pipeline/configs/*.yaml`, orchestrated by `pipeline/run_*.sh`). The earlier first-generation flat scripts (`NN{letter}_*.py` at the repo root) have been **archived and frozen** under `archive/` — see the Archive section at the end.

## Repo layout

| Path | What it is | Status |
|------|-----------|--------|
| `pipeline/src/` | `common.py` + `01_preprocess` / `02_train` / `03_evaluate` / `04_energy_dep` / `05_ensemble` | ACTIVE |
| `pipeline/configs/` | One YAML per experiment variant (`base.yaml` is the root; children use `inherit:`) | ACTIVE |
| `pipeline/output/` | Per-variant outputs: `<variant>/{data,models,results}/` + `<tag>_pipeline.log` | ACTIVE |
| `pipeline/run_*.sh` | Runners that chain 01→02→03→04 for a set of variants | ACTIVE |
| `data/` | Raw inputs `GEF.csv`, `GEF_isomer_merged.csv`, `235UALL.csv` + fitted scaler `.pkl` | ACTIVE |
| `references/` | Research PDFs (KAN / UQ / BKAN papers) | reference |
| `archive/` | Frozen first-generation flat scripts + their `models/`, `results/`, `preprocessed_*.pkl`, info `.txt`, logs | FROZEN |
| `.trash/` | Retired scratch, diagnostics, superseded configs and outputs (project rule: deletion = move here, never `rm`) | scratch |

## Hard constraints

- **Feature engineering is restricted by the advisor: inputs must be either the bare `(Z, A, E)` OR `(Z, A, E, delta_np)` only.** Do **not** add any other engineered features (e.g. parity/odd-even flags, derived ratios, mass-conservation terms, or other physics proxies) unless the advisor explicitly approves. This is why the `v*` (with `delta_np`) and `w*` (without) config families exist, and why the EDA scripts stop at `delta_np`.
- **Two-stage training order (advisor-mandated):** pre-train/warm-up on `GEF*` theoretical data → fine-tune on `235UALL.csv` experimental data. Do not skip or reorder the stages. GEF variants train on **100% of the data (no val/test split**; `split: full_train`), so GEF metrics are **in-sample** — never present them as generalization. The real held-out test is the 235UALL migration.
- **Never delete or overwrite existing results.** Deletions must `mv` into `.trash/`, never `rm`. Before re-running a pipeline stage, check whether the outputs already exist and whether the change is actually material — redundant re-runs clobber results.

## Data flow

1. **Raw inputs** in `data/`: `GEF.csv` (theoretical), `GEF_isomer_merged.csv` (isomer-merged theoretical, the current GEF source), `235UALL.csv` (experimental; columns Z, A, E, Yield, Error). Fitted `scikit-learn` scalers: `standard_scaler{Z,A,E}.pkl`, `yield_scaler.pkl`, `delta_np_scaler.pkl`, `log_yield_scaler.pkl`. Inspect them with `test_pkl_joblib.py`.

2. **`01_preprocess.py`** builds features + target (applying `target.space`/`target.power`), computes `delta_np`, optionally augments/splits, and pickles `pipeline/output/<variant>/data/preprocessed_<variant>.pkl`. Key helpers live in `common.py`: `make_features_and_target`, `compute_delta_np`, `apply_split`, `augment_yield_noise`, `load_scalers_from_pretrained`.

3. **`02_train.py`** builds the KAN (`common.build_kan_from_ckpt` / config), runs the manual AdamW loop with `CosineAnnealingWarmRestarts` or `CosineHoldAtMin`, optional progressive grid refinement (`model.refine`) and optional LBFGS polish, gradient clipping, early stopping on train loss, and periodic checkpointing. Writes `pipeline/output/<variant>/models/kan_{best,final,latest,resume}_<variant>.pth`, embedding scaler params and the full config. Fine-tune variants init from a warm-up checkpoint (`finetune.init_from`) and may freeze matching params (`finetune.freeze`).

4. **`03_evaluate.py`** loads the checkpoint (+ matching pkl), predicts, inverse-transforms to original yield space, **`np.clip(y_pred, 0, None)`** (physical non-negativity), computes R²/RMSE/MAE + high-yield-region R² (75th percentile), and writes a 2×2 PNG + JSON report under `pipeline/output/<variant>/results/`. Also supports ensemble mode.

5. **`04_energy_dep.py`** scans energy (default 0–14 MeV, step 1) at fixed nuclides, predicts, inverse-transforms, aggregates yield vs E by A and Z, and writes CSVs + PNGs under `pipeline/output/<variant>/results/`.

6. **`05_ensemble.py`** builds a multi-seed ensemble (average in original yield space) and reports σ-based uncertainty calibration (σ coverage, k68/k95). Ensemble variants have **no `preprocessed_*.pkl` of their own** — they read each member's. It imports `03_evaluate.py` by file path (the numeric filename is not a valid module name).

## Pipeline configuration

- **Inheritance:** each `pipeline/configs/<variant>.yaml` optionally sets `inherit:` a parent; `common.load_config` resolves it relative to the child's directory and deep-merges (child states only deltas). `base.yaml` is the root of every chain. Keys mirror `data / target / model / train / postprocess / energy_dep`.
- **Variant naming:** `experiment.name` is the variant tag; every output filename is suffixed with it so results never overwrite each other.
- **Variant families** (letters trace the research evolution): `g` baseline, `h` no-`delta_np`, `i` compact, `j` target power transform, `k`/`l` grid-refine + LBFGS, `m`/`o`/`p` 235UALL finetune + power sweep, `n` corrected `delta_np`, `q`/`r`/`s` single-hidden-layer capacity comparisons, `t`/`u` deeper nets (freeze sweeps), `v`/`w` seed sweeps with/without `delta_np`, `x` yield-noise augmentation.
- **Derived files:** `pipeline/dump_resolved.py` regenerates `configs/resolved/` (flattened configs) and `configs/CONFIG_TREE.md`. **These are currently stale** (they stop at family `s`; run `dump_resolved.py` to refresh).

## Environment prerequisite (MUST READ before running any pipeline script)

**Always run pipeline scripts inside the `fpy_kan` conda environment.** The `kan` library (and the matching torch/sklearn) live only there. The default `python` (base env at `D:\ProgramData\anaconda3`) raises `ModuleNotFoundError: No module named 'kan'`.

- Env location: `C:\Users\86138\.conda\envs\fpy_kan` (NOT under `D:\ProgramData\anaconda3\envs`).
- Activate: `source /d/ProgramData/anaconda3/etc/profile.d/conda.sh && conda activate fpy_kan`
- Or call the interpreter directly: `C:/Users/86138/.conda/envs/fpy_kan/python.exe`

`pipeline/common.py` derives `PROJECT_ROOT` by walking up from its own location, and `output_path()` hardcodes `<PROJECT_ROOT>/pipeline/output/...` — so the pipeline is cwd-independent for paths, but **the runners assume cwd `pipeline/` with `PYTHONPATH=src`** (so `import common` resolves).

## Common commands

Run a variant's stage chain (from the repo root, in `fpy_kan`):
```bash
python -u pipeline/src/01_preprocess.py  --config pipeline/configs/<variant>.yaml
python -u pipeline/src/02_train.py       --config pipeline/configs/<variant>.yaml
python -u pipeline/src/03_evaluate.py    --config pipeline/configs/<variant>.yaml
python -u pipeline/src/04_energy_dep.py  --config pipeline/configs/<variant>.yaml
```

`-u` (or `PYTHONUNBUFFERED=1`) is **required** — otherwise stdout is buffered and a killed process loses its whole log.

Multi-variant chaining is done by the runners, e.g. `bash pipeline/run_u.sh`, which set `PYTHONPATH=src`, `cd pipeline`, and append detailed logs to `pipeline/output/<variant>/<tag>_pipeline.log` plus an index log at the repo root (`run_*.log`).

Inspect a fitted scaler pickle:
```bash
python test_pkl_joblib.py
```

To reproduce/report a variant's exact configuration, read its resolved YAML under `pipeline/configs/resolved/` (or dump a config with `pipeline/dump_resolved.py`) rather than re-deriving it from the scripts.

## Gotchas

- Scripts fail early if required `data/*.csv` / `*.pkl` inputs are missing — run `01_*` before `02_*`/`03_*`/`04_*` for the same variant.
- Checkpoint key shapes differ across script generations: early scripts use `model_state_dict`/`test_loss`; later ones use `model_state`/`best_loss`/`config`. Loaders guard for both — preserve that when reading older checkpoints.
- Yields must be clipped to `>= 0` after inverse-transform (physical constraint); losses in raw space naturally weight high-yield samples, which is why the `j`/`l` power-transform branch exists.
- Device defaults to CPU (`torch.device('cuda' if ... else 'cpu')`); configs are tuned for CPU runs.
- `.gitignore` now exists and ignores future `pipeline/output/**` binaries (`.pth`/`.pkl`/logs) and root `/run_*.log`. **It does not untrack already-committed files** (no `git rm --cached` was done), so the repo history still carries ~218 MB of checkpoints. Large PKL/CSV/model files remain in the working tree; be careful when adding more.
- `pipeline/configs/resolved/` and `CONFIG_TREE.md` are generated and currently stale — regenerate with `pipeline/dump_resolved.py` after config changes.

## Archive (`archive/`)

The first-generation workflow was a flat set of top-level `NN{letter}_*.py` scripts (00 EDA → 01 load → 02 train → 03 eval → 04 energy-dep) that ran directly from the repo root and wrote to `models/` and `results/`. It was superseded by the config-driven `pipeline/`. It has been moved (via `git mv`, history preserved) and **frozen**:

```
archive/
├── README.md      # provenance + "frozen, not runnable as-is" note
├── scripts/       # the 41 NN*.py + fix_training_save.py + inverseNorm.py
├── models/        # legacy .pth checkpoints + training histories
├── results/       # legacy plots / JSON / CSV
├── data/          # legacy preprocessed_*.pkl
├── logs/          # legacy training logs
└── config_info/   # legacy gef_data_loading_info*.txt + data_split_info_rare_signal.txt
```

**Do not treat `archive/` as the current architecture.** Its scripts assumed repo-root cwd and their `models/`/`results/`/`preprocessed_*.pkl` now live under `archive/`, so they are not runnable as-is. There is **no cross-dependency** between `archive/` and `pipeline/` — neither imports the other.
