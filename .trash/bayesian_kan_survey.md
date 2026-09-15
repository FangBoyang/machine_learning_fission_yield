# Bayesian KAN（KAN-BNN）调研 — 2026-09-08

调研目的：第三条路「换成 Bayesian-KAN」是否已有成熟论文/代码可直接使用。

**结论：有论文，但生态极薄。没有任何一个是 pykan (B-spline KAN) 的直接扩展，也没有维护良好的库。**

---

## 一、论文

### 1. Bayesian_KANs / BKAN — Hassan (2024)

- arXiv:2408.02706，2024-08-05，cs.LG，单作者 Masoud Muhammed Hassan
- 期刊版：*Physica A*, 2025-12, DOI `10.1016/j.physa.2025.131041`
  （"Bayesian Kolmogorov-Arnold networks: Uncertainty-aware ..."）
- 方法：把**稀疏变分高斯过程 (SVGP)** 推断与 KAN 结合
  （Semantic Scholar 摘要原文："A framework integrating sparse variational Gaussian process (SVGP) inference with the Kolmogorov–Arnold ..."）
- 实验：Pima Indians Diabetes、Cleveland Heart Disease —— **两个医学小数据集的分类任务**
- 声称可分离 aleatoric / epistemic uncertainty，并显著减轻过拟合
- ⚠️ 与本项目差异大（分类 vs 回归跨 13 个数量级）；arXiv 仅 v1，未挂代码

### 2. Bayesian (Higher Order) ReLU-KAN — Giroux & Fanelli (2024) ★ 最"正规"的贝叶斯 KAN

- arXiv:2410.01687v2（2024-10-03），William & Mary
- 已发表：*IOP Mach. Learn.: Sci. Technol.*, 2025-03, DOI `10.1088/2632-2153/adbeb7`
- 自称 "the first method of uncertainty quantification in the domain of KANs"
- **方法细节**：
  - 对基函数的起止点参数 `s_i, e_i` 做变分推断：`ŝ = μ(s) + σ(s)⊙ε`，`ê = μ(e) + σ(e)⊙ε`，ε 为完全分解高斯
  - 权重 `W` 用**分层变分后验**：`q(W) = ∫ q(W|z) q(z) dz`，z 只乘性地作用于均值；
    因不可解，用辅助分布 `r(z|W)` 构造熵的近似下界（增广概率空间中的变分推断），
    `r(z|W)` 用 **inverse normalizing flows** 参数化；KL 项作为正则加入 loss
  - **aleatoric**：高斯对数似然，`L = 1/N Σ ½(e^{-r}‖u−û‖² + r)`，`r = log σ²`（即异方差似然），
    并且 σ(x) 由**另一个 surrogate Bayesian KAN** 建模，而不是双头输出
- 实验：1D 函数拟合 + 随机 PDE（Stochastic Poisson / Helmholtz）
- **作者自述 Limitations（原文引用）**：
  > "the selection of the hyperparameters of the Bayesian-HRKAN itself can be finicky and
  > we find KANs in general are generally more tricky to train than traditional neural networks"
  另外：拟合 aleatoric 项会引入额外优化困难，需要 reset optimizer state 才能把 surrogate 模型
  从局部极小扰动出来。
- **算力（论文 Tab.4 原文）**：单卡 NVIDIA A40、PyTorch 2.4.0、CUDA 12.1；
  推理 10k samples 时，一个 **[1,1] 的网络就要 ~24 GB 显存**（PDE 的 [2,2,1] 是 16–26 GB）
- 代码：https://github.com/wmdataphys/Bayesian-HR-KAN （8 stars, 8 commits，只有 Poisson/Helmholtz 两个 notebook）

### 3. DKL-KAN — Zinage, Mondal, Sarkar (2024) ★ 可能最适配本项目

- arXiv:2407.21176，"DKL-KAN: Scalable Deep Kernel Learning using Kolmogorov-Arnold Networks"
- 方法：KAN 作特征提取器 + **高斯过程作最后一层**，用**边际似然**联合优化核超参；
  大数据用 KISS-GP（结构化插值）加速
- **论文明确结论（原文）**：
  > "DKL-KAN outperforms DKL-MLP on datasets with a **low number of observations**.
  > Conversely, DKL-MLP exhibits better scalability and higher test prediction accuracy on
  > datasets with a large number of observations."
- 另外专门评测了 "modeling discontinuities and accurately estimating prediction uncertainty"
- 对本项目的意义：
  - 235UALL 只有 3096 行 / val 619 → 正落在"少样本"区间
  - GP 天然支持**逐点噪声** → GEF 与实验数据可设不同 σ（直接就是"第一条路"）
  - GP 给**协方差**，不只是对角误差棒（裂变产额评价通常要协方差）
  - 标定比 BNN 容易调

---

## 二、代码库（2026-09-08 实测）

| 仓库 | stars | commits | 推断后端 | 基函数 | 评价 |
|---|---|---|---|---|---|
| `wmdataphys/Bayesian-HR-KAN` | 8 | 8 | VI + 正态化流 | ReLU-KAN | 论文官方代码；只含 PDE notebook；**非 B-spline，与 pykan 不通** |
| `HuggingPhotonic/Bayesian_KAN` | 0 | 15 | **VI / Laplace / Metropolis / HMC** | bspline / chebyshev / wavelet / rbf | 后端最全、有 CLI + benchmark；但属光子学(MZI)副产品，目录含 `debug_vi_loss.py` / `diagnose_kl.py` / `check_convergence.py`（作者本身在跟训练稳定性斗争） |
| `VldMat/Bayesian-KAN` | 0 | 4 | 复现 Hassan(2025) | — | **亮点：discrete feature router + spike-and-slab discrete spline**，专门处理离散输入 ↔ 对应本项目 delta_np 四分支离散特征的问题 |
| `9000git/BayesianKAN_GOX-RHP` | 0 | — | — | — | 化工方向，不相关 |

GitHub 全站检索 "bayesian kolmogorov" / "bayesian KAN" / "bkan"：KAN 相关专用仓库 **最高 star = 8**。
对比 pykan 本身 20k+ stars ⇒ 该方向**尚未形成可依赖的基础设施**。

---

## 三、对本项目的判断

1. **没有 drop-in 方案**。可选三条：
   - (a) 用 `HuggingPhotonic/Bayesian_KAN`（后端最全、含 bspline 基），自己接 01/03/04/05；
   - (b) 自己给 pykan 加变分参数（spline coef + 网格当均值，加 σ 参数 + KL 项），
     但需处理 `update_grid` 会改变参数空间的问题（贝叶斯要求固定参数空间）；
   - (c) **DKL-KAN 路线**：保留现有 KAN 作特征提取器，最后一层换 GP —— 对现有 pipeline 破坏最小。

2. **算力是硬门槛**。参考 HR-KAN：[1,1] 网络 10k 采样推理需 ~24 GB 显存。
   本项目现状：CPU 训练，GEF 单 seed 20–130 分钟 ⇒ 带正态化流/HMC 的贝叶斯版本在 CPU 上基本不可行。
   **必须先解决 GPU。**

3. **DKL-KAN 性价比可能高于 pure Bayesian-KAN**（少样本占优、标定易、逐点噪声、给协方差），
   除非有明确理由必须要"参数的后验分布"本身。

4. **无论选哪条，λ_GEF 加权扫描仍应先做**：它给出的"GEF 该占多大权重"，
   正是贝叶斯框架里 σ_GEF / 先验缩放系数的初值。
