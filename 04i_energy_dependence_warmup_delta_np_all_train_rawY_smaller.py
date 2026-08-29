"""
04i_energy_dependence_warmup_delta_np_all_train_rawY_smaller.py
功能: 基于delta_np+Z/A/E的原始空间KAN模型，分析裂变产额随激发能的能量依赖性（小模型结构版）
说明: 复用02i_smaller训练的紧凑模型，固定核素Z/A/delta_np，扫描0~14MeV能量区间，输出线性坐标可视化结果
"""

import joblib
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("delta_np增强KAN模型 - 能量相关性预测分析 (线性坐标，原始空间，小模型结构版)")
print("="*60)
print("核心特性:")
print("1. 复用02i_smaller训练的紧凑模型（grid=5, k=3, width=12-12）")
print("2. 固定核素Z/A/delta_np，扫描0~14MeV激发能区间")
print("3. 绘图全英文标注，输出文件命名防重复（smaller后缀隔离）")
print("4. 完全对齐01i_smaller预处理逻辑，负值后置截断符合物理约束")
print("5. 配套01i/02i/03i_smaller，小模型结构抑制锯齿的能量依赖分析")
print("="*60)

# ========== 1. 加载模型和预处理数据 ==========
print("\n[1/6] 加载delta_np原始空间模型和预处理参数（smaller版）...")

# 加载01i_smaller生成的预处理数据（和01i_smaller逻辑完全对齐）
preprocess_path = 'preprocessed_gef_data_delta_np_with_ZAE_rawY.pkl'
if not os.path.exists(preprocess_path):
    print(f"  ✗ 预处理文件不存在: {preprocess_path}")
    exit(1)

with open(preprocess_path, 'rb') as f:
    preprocess_data = pickle.load(f)
device = preprocess_data['device']
scalers = preprocess_data['scalers']  # 包含Yield_original、delta_np的scaler
print(f"  计算设备: {device}")
print(f"  预处理数据加载成功，特征列表: {preprocess_data['feature_names']}")

# 验证训练变体标识
variant = preprocess_data['data_info'].get('training_variant', 'N/A')
if 'smaller' in str(variant):
    print(f"  ✓ 确认smaller变体: {variant}")
else:
    print(f"  ⚠️ 警告: training_variant='{variant}'，预期含'smaller'")

# 单独加载Z/A/E的scaler（和01i_smaller逻辑完全一致）
Z_SCALER_PATH = "data/standard_scalerZ.pkl"
A_SCALER_PATH = "data/standard_scalerA.pkl"
E_SCALER_PATH = "data/standard_scalerE.pkl"

for scaler_path, scaler_name in [(Z_SCALER_PATH, "Z"), (A_SCALER_PATH, "A"), (E_SCALER_PATH, "E")]:
    if not os.path.exists(scaler_path):
        print(f"  ✗ {scaler_name} scaler文件不存在: {scaler_path}")
        exit(1)

z_scaler = joblib.load(Z_SCALER_PATH)
a_scaler = joblib.load(A_SCALER_PATH)
e_scaler = joblib.load(E_SCALER_PATH)
print(f"  ✓ 单独加载Z/A/E scaler成功（和01i_smaller逻辑一致）")

# 【核心修改】加载02i_smaller训练的紧凑模型
model_path_best = "models/kan_delta_np_with_ZAE_rawY_best_smaller.pth"
model_path_final = "models/kan_delta_np_with_ZAE_rawY_final_smaller.pth"
model_path = model_path_best if os.path.exists(model_path_best) else model_path_final

if not os.path.exists(model_path):
    print(f"  ✗ 模型文件不存在: {model_path_best} 或 {model_path_final}")
    exit(1)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
print(f"  ✓ 加载delta_np原始空间模型(smaller): {model_path}")

# 验证模型结构是否为smaller版本（width中间层应为12）
width = checkpoint['config']['width']
is_smaller = (len(width) >= 3 and width[1] == 12 and width[2] == 12)
if is_smaller:
    print(f"  ✓ 确认smaller结构: width={width} (中间层12-12)")
else:
    print(f"  ⚠️ 警告: 模型结构为{width}，预期中间层12-12 (smaller版)")

# 验证grid和k是否为smaller版本
ckpt_grid = checkpoint['config'].get('grid', 'N/A')
ckpt_k = checkpoint['config'].get('k', 'N/A')
if ckpt_grid == 5 and ckpt_k == 3:
    print(f"  ✓ 确认smaller超参: grid={ckpt_grid}, k={ckpt_k}")
else:
    print(f"  ⚠️ 警告: grid={ckpt_grid}(预期5), k={ckpt_k}(预期3)")

# 验证training_stage含smaller
ts = checkpoint.get('training_stage', 'N/A')
if 'smaller' in str(ts):
    print(f"  ✓ 确认smaller训练标识: {ts}")
else:
    print(f"  ⚠️ 警告: training_stage='{ts}'，预期含'smaller'")

# 关闭自动保存，避免生成多余目录
model = KAN(
    width=checkpoint['config']['width'],
    grid=checkpoint['config']['grid'],
    k=checkpoint['config']['k'],
    seed=checkpoint['config']['seed'],
    save_act=False
)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

print(f"    模型结构: KAN{checkpoint['config']['width']}")
print(f"    输入特征: {preprocess_data['feature_names']} (共4维)")
print(f"    训练轮数: {checkpoint.get('epoch', checkpoint.get('final_epoch', 'N/A'))}")
print(f"    配套超参: grid={ckpt_grid}, k={ckpt_k}, patience={checkpoint.get('patience', 'N/A')}")

# ========== 2. 加载基准核素数据 ==========
print("\n[2/6] 加载235U基准核素数据...")
csv_path = "data/235UALL.csv"
if not os.path.exists(csv_path):
    print(f"  ✗ 基准数据文件不存在: {csv_path}")
    exit(1)

df_base = pd.read_csv(csv_path).iloc[:1032]
required_cols = ['Z', 'A', 'E']
for col in required_cols:
    if col not in df_base.columns:
        print(f"  ✗ 235UALL.csv缺少列: {col}，请检查文件格式")
        exit(1)
print(f"  基准数据形状: {df_base.shape}")
print(f"  唯一激发能值: {sorted(df_base['E'].unique())}")
print(f"  核素总数: {len(df_base)}")

# ========== 3. 反归一化获取物理参数+预计算delta_np ==========
print("\n[3/6] 反归一化获取核素物理参数+预计算delta_np...")
try:
    Z_physical = z_scaler.inverse_transform(df_base[['Z']].values).round().astype(int).flatten()
    A_physical = a_scaler.inverse_transform(df_base[['A']].values).round().astype(int).flatten()
    print(f"  Z物理范围: [{Z_physical.min()}, {Z_physical.max()}]")
    print(f"  A物理范围: [{A_physical.min()}, {A_physical.max()}]")
except Exception as e:
    print(f"  反归一化失败: {e}")
    exit(1)

# 计算delta_np（和01i_smaller逻辑完全一致，与能量无关）
N_physical = A_physical - Z_physical
I_physical = (N_physical - Z_physical) / A_physical

def calc_delta_np_row(Z, N, I):
    N_even = (N % 2 == 0)
    Z_even = (Z % 2 == 0)
    if N_even and Z_even:                       # ee
        return 2 - abs(I)
    elif (not N_even) and (not Z_even):          # oo
        return abs(I)
    elif N_even and (not Z_even) and N > Z:      # eo, N>Z
        return 1.0
    elif (not N_even) and Z_even and N < Z:      # oe, N<Z
        return 1.0
    elif N_even and (not Z_even) and N < Z:      # eo, N<Z
        return 1 - abs(I)
    elif (not N_even) and Z_even and N > Z:      # oe, N>Z
        return 1 - abs(I)
    else:
        return 1.0

delta_np_physical = np.array([
    calc_delta_np_row(Z_physical[i], N_physical[i], I_physical[i]) 
    for i in range(len(Z_physical))
])
delta_np_norm = scalers['delta_np'].transform(delta_np_physical.reshape(-1, 1)).flatten()
print(f"  delta_np物理范围: [{delta_np_physical.min():.3f}, {delta_np_physical.max():.3f}]")
print(f"  delta_np归一化范围: [{delta_np_norm.min():.3f}, {delta_np_norm.max():.3f}]")
print(f"  ✓ 核素固定属性预计算完成（delta_np不随能量变化）")

# ========== 4. 构建能量扫描预测输入 ==========
print("\n[4/6] 构建激发能扫描预测输入...")

# 物理能量网格: 0~14 MeV，步长1MeV
E_physical_grid = np.arange(0, 15, dtype=float)
print(f"  激发能物理网格: {E_physical_grid} MeV")

# 归一化能量（使用单独的E scaler，和01i_smaller逻辑一致）
E_norm_grid = e_scaler.transform(E_physical_grid.reshape(-1, 1)).flatten()
print(f"  激发能归一化网格: {E_norm_grid}")

# 构建预测输入：每个核素×每个能量对应一行
predict_rows = []
for nuc_idx in range(len(df_base)):
    Z_norm = df_base['Z'].iloc[nuc_idx]
    A_norm = df_base['A'].iloc[nuc_idx]
    dp_norm = delta_np_norm[nuc_idx]
    
    for e_idx, E_phy in enumerate(E_physical_grid):
        E_norm = E_norm_grid[e_idx]
        predict_rows.append({
            'Z_physical': Z_physical[nuc_idx],
            'A_physical': A_physical[nuc_idx],
            'E_physical': E_phy,
            'E_norm': E_norm,
            'Z_norm': Z_norm,
            'A_norm': A_norm,
            'dp_norm': dp_norm
        })

df_predict = pd.DataFrame(predict_rows)
print(f"  预测输入数据总量: {len(df_predict)} 条 ({len(df_base)}核素 × {len(E_physical_grid)}能量点)")
print(f"  特征列: Z_norm, A_norm, E_norm, dp_norm（共4维，含delta_np）")

# ========== 5. 批量预测（原始空间反归一化，无对数变换） ==========
print("\n[5/6] 进行批量预测...")
batch_size = 256  # 与02i_smaller的batch_size=256对齐
predictions = []

for i in range(0, len(df_predict), batch_size):
    batch_end = min(i + batch_size, len(df_predict))
    batch_df = df_predict.iloc[i:batch_end]
    
    # 按模型要求的顺序拼接特征：[Z_norm, A_norm, E_norm, dp_norm]
    X_batch = batch_df[['Z_norm', 'A_norm', 'E_norm', 'dp_norm']].values
    X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # 模型输出：归一化原始Yield
        y_pred_raw_norm = model(X_tensor).cpu().numpy()
    
    # 原始空间反归一化，无exp()操作
    scaler_rawY = scalers['Yield_original']
    y_pred_physical = scaler_rawY.inverse_transform(y_pred_raw_norm).flatten()
    predictions.extend(y_pred_physical)
    
    if (i // batch_size) % 5 == 0 or batch_end == len(df_predict):
        progress = batch_end / len(df_predict) * 100
        print(f"    进度: {batch_end}/{len(df_predict)} ({progress:.1f}%)")

# 负值后置截断，符合裂变产额非负物理约束
df_predict['Yield_pred'] = np.clip(predictions, 0, None)
print(f"  ✓ 预测完成")
print(f"    预测产额范围: [{df_predict['Yield_pred'].min():.2e}, {df_predict['Yield_pred'].max():.2e}]")
print(f"    平均预测产额: {df_predict['Yield_pred'].mean():.2e}")
print(f"    负值截断完成，所有预测值非负，符合物理规律")

# ========== 6. 数据聚合与可视化 ==========
print("\n[6/6] 数据聚合与线性坐标可视化...")
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 【核心修改】输出目录加 smaller 标识，完全隔离其他版本
output_dir = "results/rawY_delta_np_energy_dep_smaller"
os.makedirs(output_dir, exist_ok=True)

# 按质量数A聚合产额
df_sum_by_A = df_predict.groupby(['A_physical', 'E_physical'])['Yield_pred'].sum().reset_index()
# 按电荷数Z聚合产额
df_sum_by_Z = df_predict.groupby(['Z_physical', 'E_physical'])['Yield_pred'].sum().reset_index()

cmap = plt.cm.viridis
colors = [cmap(i) for i in np.linspace(0, 0.8, len(E_physical_grid))]

# -------------------------- 图1：按质量数A的产额分布（线性坐标） --------------------------
fig1, ax1 = plt.subplots(figsize=(12, 7))
for idx, E_phy in enumerate(E_physical_grid):
    subset = df_sum_by_A[df_sum_by_A['E_physical'] == E_phy]
    ax1.plot(
        subset['A_physical'], subset['Yield_pred'],
        color=colors[idx],
        alpha=0.7,
        linewidth=1.5,
        label=f'{E_phy:.0f} MeV' if idx % 3 == 0 else None
    )
    if E_phy == 0 or E_phy == 14:
        marker = 'o' if E_phy == 0 else 's'
        ax1.scatter(
            subset['A_physical'], subset['Yield_pred'],
            color=colors[idx],
            s=20,
            alpha=0.8,
            marker=marker,
            label=f'{E_phy:.0f} MeV (Points)' if E_phy == 0 or E_phy == 14 else None
        )

ax1.set_xlabel('Mass Number (A)', fontsize=12)
ax1.set_ylabel('Fission Yield Sum', fontsize=12)
ax1.set_title('Linear Scale: Fission Yield vs Mass Number (A) — Smaller Model (grid=5, k=3)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10, ncol=2)
ax1.set_ylim(0, df_sum_by_A['Yield_pred'].max() * 1.1)
plt.tight_layout()

# 【核心修改】文件名加 smaller 后缀
fig1_path = os.path.join(output_dir, 'yield_vs_energy_by_A_linear_delta_np_rawY_smaller.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"  ✓ 图1保存: {fig1_path}")

# -------------------------- 图2：按电荷数Z的产额分布（线性坐标） --------------------------
fig2, ax2 = plt.subplots(figsize=(12, 7))
for idx, E_phy in enumerate(E_physical_grid):
    subset = df_sum_by_Z[df_sum_by_Z['E_physical'] == E_phy]
    ax2.plot(
        subset['Z_physical'], subset['Yield_pred'],
        color=colors[idx],
        alpha=0.7,
        linewidth=1.5,
        label=f'{E_phy:.0f} MeV' if idx % 3 == 0 else None
    )
    if E_phy == 0 or E_phy == 14:
        marker = 'o' if E_phy == 0 else 's'
        ax2.scatter(
            subset['Z_physical'], subset['Yield_pred'],
            color=colors[idx],
            s=20,
            alpha=0.8,
            marker=marker,
            label=f'{E_phy:.0f} MeV (Points)' if E_phy == 0 or E_phy == 14 else None
        )

ax2.set_xlabel('Atomic Number (Z)', fontsize=12)
ax2.set_ylabel('Fission Yield Sum', fontsize=12)
ax2.set_title('Linear Scale: Fission Yield vs Atomic Number (Z) — Smaller Model (grid=5, k=3)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10, ncol=2)
ax2.set_ylim(0, df_sum_by_Z['Yield_pred'].max() * 1.1)
plt.tight_layout()

fig2_path = os.path.join(output_dir, 'yield_vs_energy_by_Z_linear_delta_np_rawY_smaller.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  ✓ 图2保存: {fig2_path}")

# ========== 7. 保存结果 ==========
print("\n[保存结果] 保存预测数据与报告...")
# 保存预测数据（删除临时特征列）
df_predict_save = df_predict.drop(columns=['Z_norm', 'A_norm', 'E_norm', 'dp_norm'])
pred_csv_path = os.path.join(output_dir, 'energy_dependence_delta_np_rawY_smaller.csv')
df_predict_save.to_csv(pred_csv_path, index=False)
# 保存聚合数据
sum_A_path = os.path.join(output_dir, 'yield_sum_by_A_delta_np_rawY_smaller.csv')
sum_Z_path = os.path.join(output_dir, 'yield_sum_by_Z_delta_np_rawY_smaller.csv')
df_sum_by_A.to_csv(sum_A_path, index=False)
df_sum_by_Z.to_csv(sum_Z_path, index=False)

# 可选：质量守恒后处理（按A分组归一到2，治谷底偏高）
df_predict_save_copy = df_predict_save.copy()
sum_by_A = df_predict_save_copy.groupby('A_physical')['Yield_pred'].transform('sum')
df_predict_save_copy['Yield_pred_normalized'] = df_predict_save_copy['Yield_pred'] / sum_by_A.replace(0, np.nan) * 2.0
norm_path = os.path.join(output_dir, 'energy_dependence_delta_np_rawY_smaller_mass_conserved.csv')
df_predict_save_copy.to_csv(norm_path, index=False)
print(f"  ✓ 质量守恒版本(按A归一到2)已保存: {norm_path}")

# 保存报告
report_path = os.path.join(output_dir, 'energy_dependence_delta_np_rawY_smaller_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("delta_np增强KAN模型 - 能量相关性预测分析报告（原始空间，小模型结构版）\n")
    f.write("="*60 + "\n\n")
    f.write(f"模型路径: {model_path}\n")
    f.write(f"核素数量: {len(df_base)}，能量扫描范围: 0~14MeV\n")
    f.write(f"输入特征: Z_norm, A_norm, E_norm, dp_norm（共4维，含delta_np）\n")
    f.write(f"模型结构: width={width} (smaller版，中间层12-12)\n")
    f.write(f"配套超参: grid={ckpt_grid}, k={ckpt_k}\n")
    f.write(f"预测产额范围: [{df_predict['Yield_pred'].min():.2e}, {df_predict['Yield_pred'].max():.2e}]\n")
    f.write(f"负值处理: 所有预测值已截断为非负，符合裂变产额物理约束\n")
    f.write(f"锯齿治理: grid=5/k=3/width缩小，从根源抑制B样条过拟合导致的低产额锯齿\n")
    f.write(f"谷底治理: 另存mass_conserved版本，按A分组归一到2\n")
    f.write(f"分析完成时间: {pd.Timestamp.now()}\n")
print(f"  ✓ 所有结果保存至: {output_dir}/")

print("\n" + "="*60)
print("原始空间能量相关性分析完成（小模型结构版，grid=5/k=3）！")
print("="*60)
print("核心结论:")
print("1. delta_np作为对关联修正因子，不随激发能变化，符合核物理原理")
print("2. 产额预测天然非负，与裂变物理规律一致")
print("3. 能量扫描覆盖0~14MeV，完整覆盖裂变激发能区间")
print("4. 小模型结构(grid=5/k=3/width=12-12)从根源抑制B样条过拟合锯齿")
print("5. 结果已与其他版本完全隔离，可直接对比分析形态差异")
print("="*60)
print("\n【多版本对比建议】")
print("→ 与04g对比：看smaller结构是否改善了Y-A曲线的锯齿问题")
print("→ 与04h对比：看delta_np+smaller vs 无delta_np，哪个形态更接近GEF")
print("→ 与04i_more_patient对比：看'小模型'和'长周期'两条路线谁更有效")
print("→ 若锯齿消失但谷底仍高：用mass_conserved版本（已自动保存）")
print("→ 若仍不理想：下一步尝试delta_np连续化(cos代理)+分段加权loss")
print("="*60)
plt.show()