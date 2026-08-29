# KAN学习裂变产额

使用 KAN（Kolmogorov-Arnold Network，`pykan` 库）学习核裂变产额映射 `(Z, A, E) → Yield`，可选的额外物理特征仅有 `delta_np`（Möller–Nix 配对修正，导师限定，禁止添加其他人工特征）。

---

## 如何自己跑程序（操作指南）

### 前提：每次都要先激活环境

```bash
source /d/ProgramData/anaconda3/etc/profile.d/conda.sh && conda activate fpy_kan
```

`kan` 库只装在 `fpy_kan` conda 环境里。直接用默认 `python`（base 环境）会报
`ModuleNotFoundError: No module named 'kan'`。

之后所有命令都必须在**仓库根目录**下运行（脚本依赖 `data/`、`pipeline/output/` 等相对路径）。

### 四个流水线阶段

每个变体都要按顺序跑四步：

| 步骤 | 脚本 | 作用 | 产出 |
|------|------|------|------|
| ① | `01_preprocess.py` | 读原始数据（GEF.csv / 235UALL.csv），复用/拟合 scaler，划分数据集 | `pipeline/output/<变体>/data/preprocessed_<变体>.pkl` |
| ② | `02_train.py` | 训练 KAN 模型（最耗时的一步） | `models/kan_best_<变体>.pth` 等 4 个模型文件 |
| ③ | `03_evaluate.py` | 评估：R² / RMSE / MAE，出图出报告 | `pipeline/output/<变体>/results/` 下 PNG + JSON |
| ④ | `04_energy_dep.py` | 能量依赖分析 | `pipeline/output/<变体>/results/` 下 CSV + PNG |

### 例：跑 o 变体（235UALL 上 finetune）全流程

```bash
# ① 数据预处理（读 235UALL.csv，复用 n 变体的 scaler，生成 preprocessed pkl）
python -u pipeline/src/01_preprocess.py --config pipeline/configs/o_ft_235UALL_power_delta_np.yaml

# ② 训练（从 n 变体的 best 权重 init，在 235UALL 上 finetune）
python -u pipeline/src/02_train.py --config pipeline/configs/o_ft_235UALL_power_delta_np.yaml

# ③ 评估（R²/RMSE/MAE + 2×2 PNG + JSON 报告）
python -u pipeline/src/03_evaluate.py --config pipeline/configs/o_ft_235UALL_power_delta_np.yaml

# ④ 能量依赖分析（CSV + PNG）
python -u pipeline/src/04_energy_dep.py --config pipeline/configs/o_ft_235UALL_power_delta_np.yaml
```

关键点：**`-u` 必须加**。它关闭 Python 的 stdout 缓冲，否则输出重定向到文件时会被
缓冲，进程一旦被杀，日志会整段丢失（项目早期 n 变体就因此"无声死亡"过）。

### 前台跑 还是 后台跑 + 日志重定向？

**前台跑（默认方式）：**

```bash
python -u pipeline/src/02_train.py --config pipeline/configs/o_ft_235UALL_power_delta_np.yaml
```

- 终端被占住，一直刷进度，跑完才回到命令行。
- 训练中途不能干别的；关终端或 Ctrl+C，训练就没了。
- 输出直接打在屏幕上，没有历史记录。
- **适合跑得快的步骤（①③④，几十秒内完成）。**

**后台跑 + 日志重定向（推荐给耗时的 ②训练）：**

```bash
# 第一次跑某个变体前，先建输出目录（否则日志写不进去会报 "No such file or directory"）
mkdir -p pipeline/output/o_ft_235UALL_power_delta_np

# 后台训练，输出实时写入日志文件
PYTHONUNBUFFERED=1 python -u pipeline/src/02_train.py \
  --config pipeline/configs/o_ft_235UALL_power_delta_np.yaml \
  > pipeline/output/o_ft_235UALL_power_delta_np/train_o.log 2>&1
```

各部分含义：

| 部分 | 作用 |
|------|------|
| `> train_o.log` | 把屏幕输出重定向写进日志文件，而不是打到终端 |
| `2>&1` | 错误信息也一起写进同一个日志文件 |
| `PYTHONUNBUFFERED=1` + `-u` | 关闭缓冲，每行日志实时落盘（进程被杀也不丢日志） |
| 后台执行 | 终端立刻释放，训练在后台自己跑，关终端不影响 |

好处：
- 终端释放，可以边训练边干别的。
- 日志留档，随时查看进度、事后复盘。
- 中途查进度：

```bash
tail -f pipeline/output/o_ft_235UALL_power_delta_np/train_o.log
```

> 一句话总结：跑得快的前台直接跑；跑得慢的训练（02）后台跑 + 日志重定向，方便中途干别的、事后查记录。

### 换变体时改两个地方

1. `--config pipeline/configs/<变体>.yaml`
2. 日志路径 / 输出目录里的 `pipeline/output/<变体>/`

前后必须同名，否则加载不到对应的 `preprocessed_*.pkl` 和模型文件。

现有变体可查看 `pipeline/configs/` 目录下的 YAML 文件，每个 YAML 描述了该变体的
数据源、特征、网络结构和训练超参。输出目录下另有 `preprocess_<变体>.log` 记录该变体
的实际预处理配置。

