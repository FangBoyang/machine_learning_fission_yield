"""
03f_evaluate_delta_np_all_train_logY.py
功能：评估对数空间训练的KAN模型（Z/A/E+delta_np输入，全训练集模式，无验证集）
说明：适配最新预处理逻辑，目标为对数空间Yield，输出天然非负，文件命名避免重复
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
print("KAN模型评估 - 对数空间Yield版（Z/A/E+delta_np输入，全训练集模式）")
print("="*60)
print("核心特性:")
print("1. 全训练集评估（无验证集，与训练逻辑完全对齐）")
print("2. 目标为对数空间Yield，反变换后天然非负")
print("3. 重点关注高产额区拟合效果")
print("="*60)

# ========== 1. 加载预处理数据（对数空间版本，全训练集） ==========
print("\n[1/6] 加载对数空间预处理数据（全训练集）...")

data_path = 'preprocessed_gef_data_delta_np_with_ZAE_logY.pkl'
if not os.path.exists(data_path):
    print(f"  ✗ 预处理文件不存在: {data_path}")
    exit(1)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train_log_norm = data['y_train']
device = data['device']
scalers = data['scalers']
feature_names = data['feature_names']
y_train_original = data['raw_data']['Yield_original']
has_error = data.get('has_error_column', False)
if has_error:
    error_train = data['error_train']

print(f"  ✓ 数据加载成功")
print(f"    训练集: {X_train.shape[0]} 样本 (100%全量，无验证集)")
print(f"    输入特征: {feature_names} (共{X_train.shape[1]}维)")
print(f"    目标变量: 归一化对数Yield")
print(f"    设备: {device}")

# ========== 2. 加载Scaler ==========
print("\n[2/6] 加载Scaler...")
try:
    scaler_logY = scalers['Yield_log']
    scaler_delta_np = scalers['delta_np']
    scaler_Y_original = scalers['Yield_original']
    print(f"  ✓ 对数Yield scaler加载成功: mean={scaler_logY.mean_[0]:.6f}, scale={scaler_logY.scale_[0]:.6f}")
    print(f"  ✓ delta_np scaler加载成功")
except Exception as e:
    print(f"  ✗ Scaler加载失败: {e}")
    exit(1)

# ========== 3. 加载训练好的模型 ==========
print("\n[3/6] 加载对数空间训练的最优模型...")

model_path_best = "models/kan_delta_np_with_ZAE_logY_best.pth"
model_path_final = "models/kan_delta_np_with_ZAE_logY_final.pth"
model_path = model_path_best if os.path.exists(model_path_best) else model_path_final

if not os.path.exists(model_path):
    print(f"  ✗ 模型文件不存在: {model_path_best} 或 {model_path_final}")
    exit(1)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
config = checkpoint['config']

# 【修改1：关闭KAN自动保存，避免生成多余的./model目录】
model = KAN(
    width=config['width'],
    grid=config['grid'],
    k=config['k'],
    seed=config['seed'],
    save_act=False  # 评估阶段不需要保存激活值
)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

print(f"  ✓ 模型加载成功: {model_path}")
print(f"    模型架构: KAN{config['width']}")

# 【修改2：修复键名不匹配问题，训练脚本存的是'epoch'不是'best_epoch'】
epoch = checkpoint.get('epoch', checkpoint.get('final_epoch', 'N/A'))
print(f"    训练轮数: {epoch}")

# 【修改3：修复损失键名，训练脚本存的是'train_loss'不是'best_loss'，同时兼容字符串类型】
best_loss = checkpoint.get('train_loss', checkpoint.get('best_loss', 'N/A'))
if isinstance(best_loss, (int, float)):
    print(f"    最佳训练损失: {best_loss:.3e}")
else:
    print(f"    最佳训练损失: {best_loss}")

# ========== 4. 预测与反归一化（和训练逻辑完全对齐） ==========
print("\n[4/6] 预测与反归一化...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_log_norm_tensor = torch.tensor(y_train_log_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_log_norm = model(X_train_tensor).cpu().numpy().flatten()

y_pred_log = scaler_logY.inverse_transform(y_pred_log_norm.reshape(-1, 1)).flatten()
y_true_log = scaler_logY.inverse_transform(y_train_log_norm.reshape(-1, 1)).flatten()
y_pred_original = np.exp(y_pred_log)
y_true_original = y_train_original.flatten()

print(f"  ✓ 预测完成")
print(f"    对数空间预测范围: [{y_pred_log.min():.3f}, {y_pred_log.max():.3f}]")
print(f"    原始空间预测范围: [{y_pred_original.min():.2e}, {y_pred_original.max():.2e}]")
print(f"    原始空间真实范围: [{y_true_original.min():.2e}, {y_true_original.max():.2e}]")
print(f"    预测最小值: {y_pred_original.min():.2e}（无负值，符合要求）")

# ========== 5. 计算评估指标 ==========
print("\n[5/6] 计算评估指标...")

# 对数空间指标
mse_log = mean_squared_error(y_train_log_norm, y_pred_log_norm)
r2_log = r2_score(y_train_log_norm, y_pred_log_norm)

# 原始空间指标
mse_original = mean_squared_error(y_true_original, y_pred_original)
rmse_original = np.sqrt(mse_original)
mae_original = mean_absolute_error(y_true_original, y_pred_original)
r2_original = r2_score(y_true_original, y_pred_original)

# 高产额区分析（取前25%作为高产额区）
high_yield_threshold = np.percentile(y_true_original, 75)
high_mask = y_true_original >= high_yield_threshold
if high_mask.sum() > 0:
    r2_high = r2_score(y_true_original[high_mask], y_pred_original[high_mask])
    mse_high = mean_squared_error(y_true_original[high_mask], y_pred_original[high_mask])
    rmse_high = np.sqrt(mse_high)
else:
    r2_high = mse_high = rmse_high = 0

# 误差列统计
if has_error:
    error_vals = error_train.flatten()
    error_mean = error_vals.mean()
    error_max = error_vals.max()
else:
    error_mean = error_max = 0

print(f"\n  📊 对数空间指标（训练优化目标）:")
print(f"    R²: {r2_log:.4f}")
print(f"    MSE: {mse_log:.3e}")

print(f"\n  📊 原始空间指标（物理意义，核心关注）:")
print(f"    R²: {r2_original:.4f}")
print(f"    RMSE: {rmse_original:.3e}")
print(f"    MAE: {mae_original:.3e}")

print(f"\n  📊 高产额区指标（阈值>{high_yield_threshold:.2e}，共{high_mask.sum()}样本）:")
print(f"    R²: {r2_high:.4f}")
print(f"    RMSE: {rmse_high:.3e}")

if has_error:
    print(f"\n  📊 GEF误差列统计:")
    print(f"    平均误差: {error_mean:.2e}")
    print(f"    最大误差: {error_max:.2e}")

# ========== 6. 可视化与结果保存 ==========
print("\n[6/6] 生成可视化图表与评估报告...")

output_dir = "results/logY_delta_np_eval"
os.makedirs(output_dir, exist_ok=True)

# 设置英文字体，避免中文显示问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KAN Model Evaluation (Log-space Training, Z/A/E+delta_np Input, Full Training Set)', fontsize=16, fontweight='bold')

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
ax1.set_title('Original Space: Predicted vs True Yield (Log Scale)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'Total R² = {r2_original:.4f}\nHigh-yield R² = {r2_high:.4f}',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
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
vis_path = os.path.join(output_dir, 'evaluation_logY_delta_np.png')
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
print(f"  ✓ 可视化图表保存: {vis_path}")

# 保存评估报告
report = {
    'model_info': {
        'name': 'KAN Log-space Training (Z/A/E+delta_np Input)',
        'model_path': model_path,
        'architecture': config['width'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'features': feature_names,
        'training_stage': checkpoint.get('training_stage', 'logY_fulltrain'),
        'split_mode': 'full_train_no_validation'
    },
    'metrics': {
        'log_space': {'r2': float(r2_log), 'mse': float(mse_log)},
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

report_path = os.path.join(output_dir, 'evaluation_report_logY_delta_np.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  ✓ 评估报告保存: {report_path}")

# ========== 打印总结 ==========
print("\n" + "="*60)
print("评估完成！核心结论:")
print("="*60)
print(f"✅ 原始空间总R²: {r2_original:.4f}")
print(f"✅ 高产额区R²: {r2_high:.4f}")
print(f"✅ 预测无负值，符合物理规律")
print(f"✅ 评估文件已全部保存至: {output_dir}/")
print("="*60)
print("\n【后续建议】")
print("1. 若高产额区R²>0.9，可直接用于物理分析")
print("2. 可使用KAN的符号回归功能提取拟合公式，增强可解释性")
print("3. 如需进一步提升，可尝试增加网格密度或轻微加宽网络")
print("="*60)