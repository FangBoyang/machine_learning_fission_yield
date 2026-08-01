"""
03i_evaluate_warm_up_delta_np_all_train_rawY_smaller.py
功能：评估原始空间训练的KAN模型（Z/A/E+delta_np输入，小模型结构版，全训练集模式）
说明：适配原始空间预处理逻辑，目标为原始空间Yield，负值后置截断；
     配套01i_smaller/02i_smaller，评估grid=5/k=3/width缩小的紧凑模型
"""

import joblib
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型评估 - 原始空间Yield版（Z/A/E+delta_np，小模型结构版）")
print("="*60)
print("核心特性:")
print("1. 全训练集评估（无验证集，与02i_smaller训练逻辑完全对齐）")
print("2. 目标为原始空间Yield，反变换直接反归一化，无对数扭曲")
print("3. 负值后置截断，符合裂变产额非负物理约束")
print("4. 配套01i_smaller/02i_smaller，评估grid=5/k=3/width缩小的紧凑模型")
print("5. 文件/目录命名完全隔离，避免覆盖其他版本结果")
print("="*60)

# ========== 1. 加载预处理数据（原始空间版本，全训练集） ==========
print("\n[1/6] 加载原始空间预处理数据（全训练集，smaller版）...")

data_path = 'preprocessed_gef_data_delta_np_with_ZAE_rawY.pkl'
if not os.path.exists(data_path):
    print(f"  ✗ 预处理文件不存在: {data_path}")
    exit(1)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train_raw_norm = data['y_train']  # 归一化后的原始Yield
device = data['device']
scalers = data['scalers']
feature_names = data['feature_names']
y_train_original = data['raw_data']['Yield_original']  # 原始空间真实Yield
has_error = data.get('has_error_column', False)
if has_error:
    error_train = data['error_train']

print(f"  ✓ 数据加载成功")
print(f"    训练集: {X_train.shape[0]} 样本 (100%全量，无验证集)")
print(f"    输入特征: {feature_names} (共{X_train.shape[1]}维)")
print(f"    目标变量: 归一化原始Yield")
print(f"    设备: {device}")
print(f"    训练变体: {data['data_info'].get('training_variant', 'N/A')}")

# 从data_info读取配套训练配置，用于验证
paired_cfg = data['data_info'].get('paired_training_config', {})
if paired_cfg:
    print(f"    配套训练配置: grid={paired_cfg.get('grid','?')}, "
          f"k={paired_cfg.get('k','?')}, "
          f"width={paired_cfg.get('width_template','?')}, "
          f"patience={paired_cfg.get('patience','?')}")

# ========== 2. 加载Scaler ==========
print("\n[2/6] 加载Scaler...")
try:
    scaler_rawY = scalers['Yield_original']  # 原始Yield scaler（兼容MinMax/Standard）
    scaler_delta_np = scalers['delta_np']
    
    # 兼容MinMaxScaler和StandardScaler的打印逻辑
    if hasattr(scaler_rawY, 'min_'):
        print(f"  ✓ 原始Yield scaler加载成功: min={scaler_rawY.min_[0]:.6f}, scale={scaler_rawY.scale_[0]:.6f} (MinMaxScaler)")
    elif hasattr(scaler_rawY, 'mean_'):
        print(f"  ✓ 原始Yield scaler加载成功: mean={scaler_rawY.mean_[0]:.6f}, scale={scaler_rawY.scale_[0]:.6f} (StandardScaler)")
    else:
        print(f"  ✓ 原始Yield scaler加载成功（类型: {type(scaler_rawY).__name__}）")
    print(f"  ✓ delta_np scaler加载成功")
        
except Exception as e:
    print(f"  ✗ Scaler加载失败: {e}")
    exit(1)

# ========== 3. 加载训练好的smaller模型 ==========
print("\n[3/6] 加载小模型结构版训练的最优模型...")

# 【核心修改】模型路径加 _smaller 后缀，与02i_smaller训练脚本对应
model_path_best = "models/kan_delta_np_with_ZAE_rawY_best_smaller.pth"
model_path_final = "models/kan_delta_np_with_ZAE_rawY_final_smaller.pth"
model_path = model_path_best if os.path.exists(model_path_best) else model_path_final

if not os.path.exists(model_path):
    print(f"  ✗ 模型文件不存在: {model_path_best} 或 {model_path_final}")
    exit(1)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
config = checkpoint['config']

# 验证模型结构是否为smaller版本（width中间层应为12）
width = config['width']
is_smaller = (len(width) >= 3 and width[1] == 12 and width[2] == 12)
if is_smaller:
    print(f"  ✓ 确认smaller结构: width={width} (中间层12-12)")
else:
    print(f"  ⚠️ 警告: 模型结构为{width}，预期中间层12-12 (smaller版)")

# 验证grid和k是否为smaller版本
ckpt_grid = config.get('grid', 'N/A')
ckpt_k = config.get('k', 'N/A')
if ckpt_grid == 5 and ckpt_k == 3:
    print(f"  ✓ 确认smaller超参: grid={ckpt_grid}, k={ckpt_k}")
else:
    print(f"  ⚠️ 警告: grid={ckpt_grid}(预期5), k={ckpt_k}(预期3)")

# 验证training_stage含smaller
training_stage = checkpoint.get('training_stage', 'N/A')
if 'smaller' in str(training_stage):
    print(f"  ✓ 确认smaller训练标识: {training_stage}")
else:
    print(f"  ⚠️ 警告: training_stage='{training_stage}'，预期含'smaller'")

# 关闭KAN自动保存，避免生成多余目录
model = KAN(
    width=config['width'],
    grid=config['grid'],
    k=config['k'],
    seed=config['seed'],
    save_act=False
)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

print(f"  ✓ 模型加载成功: {model_path}")
print(f"    模型架构: KAN{config['width']}")

# 修复键名不匹配问题
epoch = checkpoint.get('epoch', checkpoint.get('final_epoch', 'N/A'))
print(f"    训练轮数: {epoch}")

best_loss = checkpoint.get('train_loss', checkpoint.get('best_loss', 'N/A'))
if isinstance(best_loss, (int, float)):
    print(f"    最佳训练损失: {best_loss:.3e}")
else:
    print(f"    最佳训练损失: {best_loss}")

# 打印配套超参（验证与01i_smaller/02i_smaller一致）
ckpt_patience = checkpoint.get('patience', 'N/A')
ckpt_T0 = checkpoint.get('T_0', 'N/A')
print(f"    配套超参: T_0={ckpt_T0}, patience={ckpt_patience}, grid={ckpt_grid}, k={ckpt_k}")

# ========== 4. 预测与反归一化 ==========
print("\n[4/6] 预测与反归一化...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_raw_norm_tensor = torch.tensor(y_train_raw_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_raw_norm = model(X_train_tensor).cpu().numpy().flatten()

# 反归一化到原始空间（兼容MinMax/Standard）
y_pred_original = scaler_rawY.inverse_transform(y_pred_raw_norm.reshape(-1, 1)).flatten()
y_true_original = scaler_rawY.inverse_transform(y_train_raw_norm.reshape(-1, 1)).flatten()

# 负值后置截断，符合裂变产额非负物理约束
y_pred_original = np.clip(y_pred_original, 0, None)
y_true_original = y_true_original.flatten()

print(f"  ✓ 预测完成")
print(f"    归一化原始空间预测范围: [{y_pred_raw_norm.min():.3f}, {y_pred_raw_norm.max():.3f}]")
print(f"    原始空间预测范围: [{y_pred_original.min():.2e}, {y_pred_original.max():.2e}]")
print(f"    原始空间真实范围: [{y_true_original.min():.2e}, {y_true_original.max():.2e}]")
print(f"    预测最小值: {y_pred_original.min():.2e}（截断后非负，符合要求）")

# ========== 5. 计算评估指标 ==========
print("\n[5/6] 计算评估指标...")

# 归一化原始空间指标（训练优化目标）
mse_norm = mean_squared_error(y_train_raw_norm, y_pred_raw_norm)
r2_norm = r2_score(y_train_raw_norm, y_pred_raw_norm)

# 原始空间指标（物理意义核心指标）
mse_original = mean_squared_error(y_true_original, y_pred_original)
rmse_original = np.sqrt(mse_original)
mae_original = mean_absolute_error(y_true_original, y_pred_original)
r2_original = r2_score(y_true_original, y_pred_original)

# 高产额区专项分析（取前25%作为高产额区）
high_yield_threshold = np.percentile(y_true_original, 75)
high_mask = y_true_original >= high_yield_threshold
if high_mask.sum() > 0:
    r2_high = r2_score(y_true_original[high_mask], y_pred_original[high_mask])
    mse_high = mean_squared_error(y_true_original[high_mask], y_pred_original[high_mask])
    rmse_high = np.sqrt(mse_high)
else:
    r2_high = mse_high = rmse_high = 0

# GEF误差列统计
if has_error:
    error_vals = error_train.flatten()
    error_mean = error_vals.mean()
    error_max = error_vals.max()
else:
    error_mean = error_max = 0

print(f"\n  📊 归一化原始空间指标（训练优化目标）:")
print(f"    R²: {r2_norm:.4f}")
print(f"    MSE: {mse_norm:.3e}")

print(f"\n  📊 原始空间指标（物理意义，核心关注）:")
print(f"    Total R²: {r2_original:.4f}")
print(f"    RMSE: {rmse_original:.3e}")
print(f"    MAE: {mae_original:.3e}")

print(f"\n  📊 高产额区指标（阈值>{high_yield_threshold:.2e}，共{high_mask.sum()}样本）:")
print(f"    High-yield R²: {r2_high:.4f}")
print(f"    RMSE: {rmse_high:.3e}")

if has_error:
    print(f"\n  📊 GEF误差列统计:")
    print(f"    平均误差: {error_mean:.2e}")
    print(f"    最大误差: {error_max:.2e}")

# ========== 6. 可视化与结果保存 ==========
print("\n[6/6] 生成可视化图表与评估报告...")

# 【核心修改】输出目录加 smaller 标识，完全隔离其他版本
output_dir = "results/rawY_delta_np_eval_smaller"
os.makedirs(output_dir, exist_ok=True)

# 设置英文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KAN Model Evaluation (Raw-space, Z/A/E+delta_np, Smaller Model, grid=5/k=3)', fontsize=14, fontweight='bold')

# 1. 原始空间预测vs真实（对数坐标）
ax1 = axes[0, 0]
ax1.scatter(y_true_original, y_pred_original, alpha=0.6, s=20, c='blue', edgecolors='white', linewidth=0.5)
max_val = max(y_true_original.max(), y_pred_original.max())
min_val = max(y_true_original.min(), y_pred_original.min(), 1e-15)
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal Line')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('True Yield (Original Space)', fontsize=12)
ax1.set_ylabel('Predicted Yield (Original Space)', fontsize=12)
ax1.set_title('Predicted vs True Yield (Log Scale)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'Total R² = {r2_original:.4f}\nHigh-yield R² = {r2_high:.4f}\n(grid=5, k=3, width=12-12)',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. 残差图
ax2 = axes[0, 1]
residuals = y_pred_original - y_true_original
ax2.scatter(y_pred_original, residuals, alpha=0.6, s=20, c='green', edgecolors='white', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlabel('Predicted Yield (Original Space)', fontsize=12)
ax2.set_ylabel('Residuals (Predicted - True)', fontsize=12)
ax2.set_title('Residual Plot (Original Space)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'Mean Residual: {residuals.mean():.2e}\nResidual Std: {residuals.std():.2e}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. 相对误差分布
ax3 = axes[1, 0]
rel_error = np.abs(residuals) / (np.abs(y_true_original) + 1e-12)
rel_error_clipped = np.clip(rel_error, 0, 10)
ax3.hist(rel_error_clipped, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.set_xlabel('Relative Error |Pred-True|/|True|', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Relative Error Distribution', fontsize=14)
ax3.grid(True, alpha=0.3)
median_err = np.median(rel_error)
p90_err = np.percentile(rel_error, 90)
ax3.axvline(median_err, color='r', linestyle='--', label=f'Median: {median_err:.2f}')
ax3.axvline(p90_err, color='orange', linestyle='--', label=f'90th Percentile: {p90_err:.2f}')
ax3.legend()
ax3.set_xlim(-0.5, 10.5)

# 4. 各特征维度MAE分布
ax4 = axes[1, 1]
colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
for i, (feat_name, color) in enumerate(zip(feature_names, colors)):
    feat_vals = X_train[:, i]
    bin_edges = np.percentile(feat_vals, np.linspace(0, 100, 15))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mae_bins = []
    for j in range(len(bin_edges)-1):
        mask = (feat_vals >= bin_edges[j]) & (feat_vals < bin_edges[j+1])
        if mask.sum() >= 5:
            mae_bins.append(mean_absolute_error(y_true_original[mask], y_pred_original[mask]))
        else:
            mae_bins.append(np.nan)
    valid = ~np.isnan(mae_bins)
    if valid.any():
        ax4.plot(bin_centers[valid], np.array(mae_bins)[valid], 'o-', color=color, label=feat_name, alpha=0.7, markersize=4)

ax4.set_xlabel('Feature Value (Normalized Space)', fontsize=12)
ax4.set_ylabel('MAE (Original Space)', fontsize=12)
ax4.set_title('MAE Distribution across Feature Dimensions', fontsize=14)
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# 【核心修改】图片文件名加 smaller 后缀
vis_path = os.path.join(output_dir, 'evaluation_rawY_delta_np_smaller.png')
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
print(f"  ✓ 可视化图表保存: {vis_path}")

# 保存评估报告（JSON格式）
report = {
    'model_info': {
        'name': 'KAN Raw-space Training (Z/A/E+delta_np, Smaller Model Variant)',
        'model_path': model_path,
        'architecture': config['width'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'features': feature_names,
        'training_stage': training_stage,
        'split_mode': 'full_train_no_validation',
        'variant': '03i_smaller',
        'paired_training_config': {
            'width': config['width'],
            'grid': ckpt_grid,
            'k': ckpt_k,
            'T_0': ckpt_T0,
            'patience': ckpt_patience,
            'learning_rate': config.get('learning_rate', 'N/A'),
            'batch_size': config.get('batch_size', 'N/A'),
        }
    },
    'metrics': {
        'normalized_raw_space': {'r2': float(r2_norm), 'mse': float(mse_norm)},
        'original_space': {'r2': float(r2_original), 'rmse': float(rmse_original), 'mae': float(mae_original)},
        'high_yield_region': {
            'threshold': float(high_yield_threshold),
            'sample_count': int(high_mask.sum()),
            'r2': float(r2_high),
            'rmse': float(rmse_high)
        }
    },
    'data_stats': {
        'total_samples': int(X_train.shape[0]),
        'true_yield_range': [float(y_true_original.min()), float(y_true_original.max())],
        'pred_yield_range': [float(y_pred_original.min()), float(y_pred_original.max())],
        'gefs_error_mean': float(error_mean) if has_error else None
    },
    'visualization_path': vis_path,
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

# 【核心修改】报告文件名加 smaller 后缀
report_path = os.path.join(output_dir, 'evaluation_report_rawY_delta_np_smaller.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  ✓ 评估报告保存: {report_path}")

# ========== 打印总结 ==========
print("\n" + "="*60)
print("原始空间Yield模型评估完成（小模型结构版，grid=5/k=3）！")
print("="*60)
print(f"✅ 原始空间总R²: {r2_original:.4f}")
print(f"✅ 高产额区R²: {r2_high:.4f}")
print(f"✅ 预测值已截断为非负，符合物理规律")
print(f"✅ 配套训练策略: grid={ckpt_grid}, k={ckpt_k}, width={config['width']}")
print(f"✅ 锯齿预期改善: grid从20→5，B样条自由度大幅降低")
print(f"✅ 所有评估文件已保存至: {output_dir}/（与其他版本完全隔离）")
print("="*60)

# 分级诊断（对症你之前的两个症状）
print("\n【锯齿与高产额诊断】")
if r2_high > 0.85 and r2_original > 0.80:
    print("✅ 锯齿消除 + 高产额拟合良好 → 直接进04i能量依赖分析")
elif r2_high > 0.7:
    print("△ 高产额尚可，锯齿可能仍存在 → 建议叠加分段加权loss（低产额用相对误差）")
else:
    print("✗ 高产额仍不理想 → 建议：(1)delta_np连续化 或 (2)分段加权loss")

print("\n【后续建议】")
print("1. 锯齿消失+高产额R²>0.85 → 进04i做能量依赖分析")
print("2. 锯齿消失但高产额钝 → 加分段加权loss（低产额相对误差+高产额MSE×5）")
print("3. 锯齿仍在 → delta_np改连续代理(cos奇偶)，移除硬阶跃")
print("4. 谷底偏高 → 加质量守恒后处理: Y_pred_by_A / sum * 2.0")
print("="*60)