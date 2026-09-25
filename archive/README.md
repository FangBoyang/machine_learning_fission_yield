# archive/ — 第一代工作流（已冻结）

**归档日期：2026-09-25**

这里存放的是本项目**第一代**代码与产物，已被 `pipeline/` 的配置驱动工作流取代。

## 这一代是什么

第一代是一组平铺在仓库根目录的独立脚本，命名为 `NN{letter}_*.py`：

| 前缀 | 阶段 | 当时的产出位置 |
|------|------|----------------|
| `00_` | 特征分析 / EDA | `results/...` 图 |
| `01_` | 数据加载 / 预处理 | `preprocessed_*.pkl` |
| `02_` | 模型训练（手写训练循环） | `models/kan_*.pth` |
| `03_` | 评估 | `results/<variant>/` |
| `04_` | 能量依赖分析 | `results/<variant>/` |

数字前缀是**流程阶段**，字母后缀是**实验变体**（`b`~`i`，其中 `e` 一代留下了 6 个近乎重复的训练脚本，含一个 `_failed`）。每个变体还会写一个 `gef_data_loading_info_*.txt` 记录其配置。

## 为什么被取代

第一代靠"复制脚本 + 改字母后缀"来扩展实验，导致变体间大量重复、参数散落在各脚本里、结果文件名靠手工维护。`pipeline/` 用**一份 YAML 描述一个变体**（`inherit:` 深度合并复用父配置），把阶段拆成 `01_preprocess`→`02_train`→`03_evaluate`→`04_energy_dep`(+`05_ensemble`)，产物统一落在 `pipeline/output/<variant>/`。

第一代最后真正修改的时间约在 **2026-08-01**；此后所有工作都转到 `pipeline/`。

## 目录结构

```
archive/
├── scripts/      # 41 个 NN*.py + fix_training_save.py + inverseNorm.py
├── models/       # 旧的 .pth 检查点 + 训练历史 json
├── results/      # 旧的图 / JSON / CSV
├── data/         # 旧的根级 preprocessed_*.pkl
├── logs/         # 旧的训练日志（3 个；不含 pipeline 的 run_*.log）
└── config_info/  # 旧的 gef_data_loading_info*.txt + data_split_info_rare_signal.txt
```

## 重要：已冻结，不能直接运行

- 这些脚本用**当前工作目录**（仓库根）的相对路径读写 `data/`、`models/`、`results/`、`preprocessed_*.pkl`。
- 移动之后，它们依赖的 `models/`、`results/`、`preprocessed_*.pkl` 已不在原位，因此**直接运行会失败或写到错误的路径**，这是**刻意**的——它们已是历史，不再维护。
- 若确需复现旧结果，请在新位置**重新指定路径**，或直接改跑 `pipeline/` 里对应的等价变体（`g`/`h`/`i` 家族就是第一代 `f`/`g`/`h`/`i` 的配置化重写）。

## 与 pipeline 的关系

第一代与 `pipeline/` **完全解耦**：根级脚本不 import `pipeline/`，`pipeline/` 也不读取本目录下的任何文件。因此归档**不影响** pipeline 的运行。

## 历史

文件是用 `git mv` 移动的，重命名历史保留；可用 `git log --follow <文件>` 追溯归档前的提交记录。
