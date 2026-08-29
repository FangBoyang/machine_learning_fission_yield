# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## What this repo is

A research codebase exploring **KAN (Kolmogorov-Arnold Networks)** — and per the README, GNNs as well — for predicting **fission yields (裂变产额)**. In practice every script here uses KAN via the `pykan` package (`from kan import KAN`); no GNN code is present yet. There is no package structure, no build system, no `src/` directory, and no test framework — it is a flat collection of standalone, top-level Python scripts run directly.

The model learns the mapping `(Z, A, E) → Yield` where Z=charge, A=mass number, E=incident energy, plus optional physics-derived features (e.g. `delta_np`, the Möller–Nix pairing correction). Targets are trained either in raw yield space (plain MSE) or log-transformed space.

## Hard constraints

- **Feature engineering is restricted by the advisor: inputs must be either the bare `(Z, A, E)` OR `(Z, A, E, delta_np)` only.** Do **not** add any other engineered features (e.g. parity/odd-even flags, derived ratios, mass-conservation terms, or other physics proxies) unless the advisor explicitly approves. When proposing or writing a new `01_*` data-loading variant, the feature set is either `Z/A/E` or `Z/A/E/delta_np` — nothing else. This is why existing `delta_np`-only and `ZAE`-only variants exist and why `00_feature_analysis.py`/`00_feature_analysis_GEF.py` stop at `delta_np`.

## File naming convention (the core architecture)

Files are named `NN{letter}_*.py`. The **number prefix is the pipeline stage** and the **letter suffix is an experiment variant**. Variants are designed to run as a matched set across all stages — read the matching data-loading, training, evaluation, and energy-dependence scripts together.

| Prefix | Stage | Output |
|--------|-------|--------|
| `00_` | Experimental / feature analysis (EDA) | printed analysis, `results/...` plots |
| `01_` | Data loading / preprocessing | `preprocessed_*.pkl` |
| `02_` | Model training | `models/kan_*.pth` (+ `models/*_training_history.json`) |
| `03_` | Model evaluation | `results/<variant>/` (PNG, JSON report) |
| `04_` | Energy-dependence analysis | `results/<variant>/`, `results/*.csv` |

Letter suffixes denote experiment variants, often stacked to encode the configuration:
- `rawY` / `logY` — train in raw yield space (plain MSE) vs log-transformed yield space
- `delta_np` — include `delta_np` (pairing correction) as an input feature
- `ZAE` — use Z/A/E as features
- `fulltrain` / `all_train` — train on the full dataset (no validation split; early-stop on train loss)
- `warm_up` — warm-up training schedule variant
- `rare_signal` — data split that protects rare-signal samples (see `data_split_info_rare_signal.txt`)
- `finetune` — fine-tune on `235UALL.csv` (experimental data) after pre-training on `GEF.csv` (theoretical)
- `smaller` — reduced network width (`12-12` instead of `24-24`), `grid=5`, `k=3` to suppress B-spline overfitting "sawtooth"

Example matched set: `01i_data_loading_*` → `02i_train_*` → `03i_evaluate_*` → `04i_energy_dependence_*`. Each variant also writes a `gef_data_loading_info_*.txt` (or similar) describing its config; read that `.txt` to recover a variant's exact setup.

**Convention to preserve:** when adding a new experiment, keep the numbered prefix, use a new letter suffix, and suffix all output filenames (PKL/PNG/JSON/model names) with the variant tag so results never overwrite each other. The `*_i_*` "smaller" scripts explicitly warn against mixing `*_smaller` preprocessing with non-smaller models.

## Data flow

1. **Raw inputs** live in `data/`:
   - `235UALL.csv` — experimental data (columns: Z, A, E, Yield, Error). Used for EDA, baseline training, and fine-tuning.
   - `GEF.csv` — GEF *theoretical* data (~18k–180k rows). The main training corpus.
   - `standard_scalerZ.pkl` / `A` / `E` / `yield_scaler.pkl`, plus GEF-specific `delta_np_scaler.pkl`, `log_yield_scaler.pkl` — fitted `scikit-learn` scalers used for (de)normalization. Inspect them with `test_pkl_joblib.py`.

2. **`01_*` scripts** load the CSV + scalers, build train/val/test splits, and pickle a `preprocessed_*.pkl` dict containing at least: `X_train/val/test` (numpy + tensors), `y_train/val/test`, `device`, `scalers`, `feature_names`, `target_name`, `data_info` (incl. `training_variant` and `paired_training_config`), and `raw_data` (e.g. original-space `Yield_original` for evaluation). Later GEF variants train on 100% of the data (no split).

3. **`02_*` scripts** load the matching `preprocessed_*.pkl`, build the KAN with `KAN(width=[input_dim, hidden..., 1], grid=, k=, seed=)`, and run a manual training loop: AdamW optimizer, `CosineAnnealingWarmRestarts(T_0=300)` scheduler, plain `nn.MSELoss()`, gradient clipping, and early stopping on training loss (`patience≈300`, `min_delta≈5e-6`). Best/final checkpoints save to `models/kan_*_best.pth` / `_final.pth` and **embed the scaler params** (`scaler_rawY_min/scale/mean`, etc.) so evaluation can inverse-transform without reloading the scaler file.

4. **`03_*` scripts** load the matched `preprocessed_*.pkl` + `models/kan_*_best.pth`, predict, inverse-transform to original yield space, **`np.clip(y_pred, 0, None)`** to enforce the physical non-negativity of yields, compute metrics (R², RMSE, MAE, high-yield-region R² at the 75th percentile), and emit a 2×2 matplotlib figure + JSON report under `results/<variant>/`.

5. **`04_*` scripts** do energy-dependence analysis (yield vs E grouped by A/Z, CSV + PNG outputs in `results/`).

## KAN construction notes

- Library: `pykan` (`from kan import KAN`). Install from source: `pip install git+https://github.com/KindXiaoming/pykan.git` (not on PyPI under that name in this project's era). Other deps: `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`.
- `model/` directory contains a **separate** pykan high-level-training artifact: `0.0_config.yml` (a pykan `KANConfig`: `width`, `grid`, `k`, `symbolic_enabled`, `affine_trainable`, etc.), `0.0_state` (saved model state), `0.0_cache_data`, `history.txt`. This is pykan's auto-save format from its `.fit()` API. The newer `*_i_*` training scripts instead use the **manual loop** and pass `save_act=False` to the KAN constructor to suppress pykan's automatic directory/folder creation. Treat `model/` and the manual-loop scripts as two parallel workflows.
- `fix_training_save.py` is a one-off recovery utility for re-assembling training history after a crash; not part of the normal pipeline.

## Common commands

## Environment prerequisite (MUST READ before running any pipeline script)

**Always run pipeline scripts inside the `fpy_kan` conda environment.** The `kan` library (and the matching torch/sklearn) live only there. The default `python` (base env at `D:\ProgramData\anaconda3`) raises `ModuleNotFoundError: No module named 'kan'`.

- Env location: `C:\Users\86138\.conda\envs\fpy_kan` (NOT under `D:\ProgramData\anaconda3\envs`).
- Activate: `source /d/ProgramData/anaconda3/etc/profile.d/conda.sh && conda activate fpy_kan`
- Or call the interpreter directly: `C:/Users/86138/.conda/envs/fpy_kan/python.exe`

All scripts run from the repository root (they rely on relative paths like `data/`, `models/`, `results/`). There is no test runner — `test_pkl_joblib.py` is just a manual scaler-inspection utility, not a test suite.

Run a pipeline stage (whole variant chain):
```bash
python 01i_data_loading_warm_up_delta_np_all_train_rawY_smaller.py   # preprocess
python 02i_train_warm_up_delta_np_all_train_rawY_smaller.py          # train
python 03i_evaluate_warm_up_delta_np_all_train_rawY_smaller.py       # evaluate
python 04i_energy_dependence_warm_up_delta_np_all_train_rawY_smaller.py  # energy dep
```

Inspect a fitted scaler pickle:
```bash
python test_pkl_joblib.py
```

To reproduce/report a variant's configuration, read its `gef_data_loading_info_*.txt` (or `*_data_loading_info*.txt`) rather than re-deriving it from the script.

## Gotchas

- Scripts exit early with printed errors if expected `data/*.csv` / `*.pkl` / `preprocessed_*.pkl` inputs are missing — run `01_*` before `02_*`/`03_*`/`04_*` for the same variant.
- Checkpoint key shapes differ across script generations: early scripts use `model_state_dict`/`test_loss`; later ones use `model_state`/`best_loss`/`config`. 03/`evaluate` scripts guard for both — preserve that when loading older checkpoints.
- Yields must be clipped to `>= 0` after inverse-transform (physical constraint); losses in raw space naturally weight high-yield samples.
- Device defaults to CPU (`torch.device('cuda' if ... else 'cpu')`); configs are tuned for CPU runs.
- Large PKL/CSV/model files are committed in the repo (no `.gitignore`); be careful when adding more.
