# bkan/ — BKAN（SVGP-KAN）pipeline

用 **SVGP-KAN**（Sparse Variational Gaussian Process KAN）替代 pykan，学习核裂变产额映射，
并给出**有原则的不确定性（μ, σ）**。

## 这是什么 / 不是什么

- **是**：`pipeline/`（pykan）那条线的替代主线，配置驱动、输出独立在 `bkan/output/`。
- **不是**：`pipeline/` 的补丁。两套互相独立，`pipeline/` **冻结保留**作为肉眼对照的参考。
- **比较方式 = 肉眼看 `yield_vs_A.png` / `yield_vs_Z.png`**，不比 R²/RMSE。

## 库来源（vendored）

源码在 `bkan/vendor/`，取自 GitHub 镜像（`gh-proxy.com`），
`svgp-kan @ main`，zip 快照日期 2026-04-20。

- 该库**未发布到 PyPI**，所以 vendor 进来以保证离线可复现、可打补丁。
- 依赖极轻：`torch>=2.0.0, numpy, matplotlib, scikit-learn`，与 `fpy_kan` 兼容。
- `bkan/src/_vendor.py` 负责把它加入 `sys.path`，**不需要 `pip install`**。

## 快速开始

```bash
source /d/ProgramData/anaconda3/etc/profile.d/conda.sh && conda activate fpy_kan

# 阶段 0：验库（必做）
cd bkan/vendor && PYTHONPATH="$PWD" python -u examples/Basic_Sci_Discovery_Examples/simple_discovery.py

# 阶段 1：数据层等价性校验（移植是否与旧真源逐位一致）
python -u bkan/src/check_equivalence.py --config bkan/configs/gef_log.yaml

# 阶段 1-4：跑一个变体
python -u bkan/src/01_preprocess.py --config bkan/configs/<variant>.yaml
python -u bkan/src/02_train.py      --config bkan/configs/<variant>.yaml
python -u bkan/src/03_plots.py      --config bkan/configs/<variant>.yaml   # ★ MVP：两张图
python -u bkan/src/04_uq.py         --config bkan/configs/<variant>.yaml   # UQ 标定
```

`-u` 必须加（否则 stdout 缓冲，进程被杀会丢整段日志）。

## 目录

```
bkan/
├── src/
│   ├── _vendor.py            # sys.path shim
│   ├── data.py               # 数据/目标/σ 层（移植自 pipeline/src/common.py）
│   ├── check_equivalence.py  # 与旧真源的逐位等价性断言
│   ├── 01_preprocess.py      # → bkan/output/<variant>/data/preprocessed_<variant>.pkl
│   ├── 02_train.py           # GPKAN + 手写循环（两段式 warm-up / pruning）
│   ├── 03_plots.py           # ★ yield_vs_A / yield_vs_Z（可比性关键）
│   └── 04_uq.py              # 覆盖率 / ρ(σ,|err|) / 标准化残差
├── configs/                  # base.yaml 为根；子配置用 inherit
├── output/                   # 产物（已 gitignore）
└── vendor/                   # svgp-kan 源码（进 git）
```

## 关键设计（已定）

1. **损失用库自带的 `gaussian_nll_loss`**（= 论文 Eq.(2)，含 log 项），总方差 = `f_var + σ_i²`。
   > **不要**裸写 χ²（论文 Eq.(4)）：若噪声可学，裸 χ² 会让它单调发散到 ∞。
   > 库的损失带 log 项，正好是 Eq.(2)。
2. **σ_i 逐点已知**，来自数据第 5 列（第 5 列 / 第 4 列换算到 log 空间），**不参与学习**。
3. **预测标准差 = √(f_var + σ_expt²)**。δ（模型缺陷项）是可选的实验项，不是必需品。
4. **目标空间用 `log`**（σ 与产额同量级 ⇒ raw 空间下 1/σ²∝1/y²，低产区权重可高 ~10¹⁰·³⁴ 倍）。
5. **`Yield==0` 的行**（GEF 878 / 235UALL 472）σ 未定义，`sigma.zero_yield: one` 给 σ_ln = 1.0
   （= "相对不确定度 100%"），**保留行**以免改变样本数。

## 可比性约定（照抄 `pipeline/src/04_energy_dep.py`）

换掉这个，图就不能并排比：

- 参考核素 = `235UALL.csv` **前 1032 行**（`04:218-220`，与外部 04g/04i 逐点一致）
- 能量网格 `[0,14]` 步长 1，用 `data/standard_scalerE.pkl` 归一化
- 特征拼接顺序按 config 的 `features`
- 按 `(A,E)` / `(Z,E)` **求和**聚合
- 画图：每 E 一条线、每 3 条标 legend

## 已核实的事实

- **输入必须 ~N(0,1)**：库的诱导点固定在 `[-1.5,1.5]` 且 `forward()` 不做归一化。
  本项目的 Z/A/E/delta_np 已标准化，天然兼容。
- **观测噪声在库里是全局标量**（`likelihood_log_var`），所以逐点 σ 要自己加在似然里。
- **KL 量级**：初始化时约 5000（M=16）~ 29000（M=64），**训练中迅速降到 ~20-30**。
  因此库默认的 `kl_weight=0.01` 可用（官方 examples 也是这个值），不必按 1/N 缩放。
- **官方 example 实测**（CPU）：600 点、2000 轮约 2m25s，NLL 收敛到 −2.32。

## 参照文献

`references/` 里有导师（裴俊琛）组的 BNN 裂变产额论文，含本项目 χ² 形式的确切出处
（arXiv:2607.04148 Eq.2/Eq.4）。**注意其 Eq.(2) 含 log 项、Eq.(4) 不含，二者矛盾** ——
需向学长确认实际实现用哪个；我们这边必须用 Eq.(2)。

另注意：裴组的 `E` 是**激发能** `e_n + S_n`，本项目用的是**入射能 E**，
差一个中子分离能常数——并排比图时别直接对着 E 轴比。
