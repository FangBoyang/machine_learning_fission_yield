"""
03e_evaluate_warmup_model.py
功能：评估 warm-up 训练得到的 KAN 模型（GEF 数据，交互特征，归一化空间）
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Warm-up KAN Model Evaluation")
print("="*60)

# ========== 1. 加载数据 ==========
print("\n[1/6] Loading data and model...")

# 加载预处理数据
with open('preprocessed_gef_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_all, y_all = data['X_train'], data['y_train']
device = data['device']

# 重新划分验证集（与训练时一致）
val_ratio = 0.2
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=val_ratio, random_state=42, shuffle=True
)

print(f"  ✓ Data loaded")
print(f"    Validation set: {X_val.shape[0]} samples")
print(f"    Device: {device}")

# ========== 2. 加载 scaler ==========
print("\n[2/6] Loading scalers...")
try:
    scaler_Z = joblib.load('data/standard_scalerZ.pkl')
    scaler_A = joblib.load('data/standard_scalerA.pkl')
    scaler_E = joblib.load('data/standard_scalerE.pkl')
    scaler_Y = joblib.load('data/yield_scaler.pkl')
    print("  ✓ All scalers loaded")
except Exception as e:
    print(f"  ✗ Failed to load scalers: {e}")
    exit(1)

# ========== 3. 加载模型 ==========
print("\n[3/6] Loading warm-up model...")
model_path = "models/kan_warmup_simple_final.pth"
if not os.path.exists(model_path):
    model_path = "models/kan_warmup_simple_best.pth"
if not os.path.exists(model_path):
    print(f"  ✗ Model not found: {model_path}")
    exit(1)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
config = checkpoint['config']
feature_names = checkpoint.get('feature_names', ['Z', 'A', 'E', 'Z_squared', 'A_squared', 'E_squared', 'ZxA', 'ZxE', 'AxE'])
interaction_features = feature_names[3:]  # 交互特征列表

# 重建模型
model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

print(f"  ✓ Model loaded: {model_path}")
print(f"    Architecture: KAN{config['width']}")
print(f"    Input dimension: {config['width'][0]}")

# ========== 4. 特征工程（构建交互特征） ==========
print("\n[4/6] Building interaction features for validation set...")

# 提取归一化特征
Z_val_norm = X_val[:, 0:1]
A_val_norm = X_val[:, 1:2]
E_val_norm = X_val[:, 2:3]

# 构造交互特征
train_interactions = np.hstack([
    X_train[:, 0:1]**2, X_train[:, 1:2]**2, X_train[:, 2:3]**2,
    X_train[:, 0:1]*X_train[:, 1:2], X_train[:, 0:1]*X_train[:, 2:3], X_train[:, 1:2]*X_train[:, 2:3]
])
val_interactions = np.hstack([
    Z_val_norm**2, A_val_norm**2, E_val_norm**2,
    Z_val_norm*A_val_norm, Z_val_norm*E_val_norm, A_val_norm*E_val_norm
])

# 使用训练集统计量归一化交互特征
interact_mean = train_interactions.mean(axis=0)
interact_std = train_interactions.std(axis=0) + 1e-12
val_interactions_norm = (val_interactions - interact_mean) / interact_std

# 合并特征
X_val_augmented = np.hstack([X_val, val_interactions_norm])
print(f"  ✓ Feature engineering done. Shape: {X_val_augmented.shape}")

# ========== 5. 预测与反归一化 ==========
print("\n[5/6] Predicting and denormalizing...")

X_val_tensor = torch.tensor(X_val_augmented, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_norm = model(X_val_tensor).cpu().numpy().flatten()

# 反归一化到物理值
y_val_phy = scaler_Y.inverse_transform(y_val).flatten()
y_pred_phy = scaler_Y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

print(f"  ✓ Prediction completed")
print(f"    Predicted range (physical): [{y_pred_phy.min():.2e}, {y_pred_phy.max():.2e}]")
print(f"    True range (physical): [{y_val_phy.min():.2e}, {y_val_phy.max():.2e}]")

# ========== 6. 计算指标与可视化 ==========
print("\n[6/6] Computing metrics and generating plots...")

# 基本指标
mse = mean_squared_error(y_val_phy, y_pred_phy)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val_phy, y_pred_phy)
r2 = r2_score(y_val_phy, y_pred_phy)

print(f"\n  📊 Model Performance (Physical Scale):")
print(f"    MSE: {mse:.3e}")
print(f"    RMSE: {rmse:.3e}")
print(f"    MAE: {mae:.3e}")
print(f"    R²: {r2:.4f}")

# 零值分析
zero_mask = (y_val_phy == 0)
if zero_mask.any():
    zero_pred = y_pred_phy[zero_mask]
    print(f"\n  Zero-value analysis:")
    print(f"    Zero count: {zero_mask.sum()}")
    print(f"    Mean prediction at zero: {zero_pred.mean():.2e}")
    print(f"    Max prediction at zero: {zero_pred.max():.2e}")

# 高产额分析
high_thresh = np.percentile(y_val_phy, 75)
high_mask = y_val_phy >= high_thresh
if high_mask.any():
    r2_high = r2_score(y_val_phy[high_mask], y_pred_phy[high_mask])
    mse_high = mean_squared_error(y_val_phy[high_mask], y_pred_phy[high_mask])
    print(f"\n  High-yield region (>{high_thresh:.2e}):")
    print(f"    Samples: {high_mask.sum()}")
    print(f"    R²: {r2_high:.4f}")
    print(f"    MSE: {mse_high:.3e}")

# ========== 可视化 ==========
os.makedirs("results/warmup_eval", exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Warm-up KAN Model Evaluation (GEF Validation Set)', fontsize=16, fontweight='bold')

# 1. 预测 vs 真实（对数坐标）
ax1 = axes[0, 0]
ax1.scatter(y_val_phy, y_pred_phy, alpha=0.6, s=20, c='blue', edgecolors='white', linewidth=0.5)
max_val = max(y_val_phy.max(), y_pred_phy.max())
min_val = min(y_val_phy.min(), y_pred_phy.min())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('True Yield (physical)', fontsize=12)
ax1.set_ylabel('Predicted Yield (physical)', fontsize=12)
ax1.set_title('Predicted vs True (log-log)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. 残差图
ax2 = axes[0, 1]
residuals = y_pred_phy - y_val_phy
ax2.scatter(y_pred_phy, residuals, alpha=0.6, s=20, c='green', edgecolors='white', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlabel('Predicted Yield', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'Mean residual: {residuals.mean():.2e}\nStd residual: {residuals.std():.2e}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. 相对误差分布
ax3 = axes[1, 0]
rel_error = np.abs(residuals) / (np.abs(y_val_phy) + 1e-12)
rel_error_clipped = np.clip(rel_error, 0, 5)  # 限制在0-5倍
ax3.hist(rel_error_clipped, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.set_xlabel('Relative Error |Pred-True|/|True|', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Relative Error Distribution', fontsize=14)
ax3.grid(True, alpha=0.3)
median_err = np.median(rel_error)
p90_err = np.percentile(rel_error, 90)
ax3.axvline(median_err, color='r', linestyle='--', label=f'Median: {median_err:.2f}')
ax3.axvline(p90_err, color='orange', linestyle='--', label=f'90th: {p90_err:.2f}')
ax3.legend()

# 4. 按特征维度的误差分析（使用归一化特征值）
ax4 = axes[1, 1]
all_features = ['Z', 'A', 'E'] + interaction_features
colors = plt.cm.tab20(np.linspace(0, 1, len(all_features)))
for i, (feat, color) in enumerate(zip(all_features, colors)):
    if i >= X_val_augmented.shape[1]:
        continue
    # 将特征值分成若干bin，计算每个bin内的MAE
    feat_vals = X_val_augmented[:, i]
    # 使用分位数划分bin
    bin_edges = np.percentile(feat_vals, np.linspace(0, 100, 15))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mae_bins = []
    for j in range(len(bin_edges)-1):
        mask = (feat_vals >= bin_edges[j]) & (feat_vals < bin_edges[j+1])
        if mask.sum() >= 5:
            mae_bins.append(mean_absolute_error(y_val_phy[mask], y_pred_phy[mask]))
        else:
            mae_bins.append(np.nan)
    valid = ~np.isnan(mae_bins)
    if valid.any():
        ax4.plot(bin_centers[valid], np.array(mae_bins)[valid], 'o-', color=color, label=feat, alpha=0.7, markersize=4)

ax4.set_xlabel('Feature Value (normalized)', fontsize=12)
ax4.set_ylabel('MAE (physical)', fontsize=12)
ax4.set_title('MAE vs Feature Value', fontsize=14)
ax4.legend(fontsize=8, ncol=2, loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
vis_path = 'results/warmup_eval/warmup_evaluation.png'
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved: {vis_path}")

# ========== 保存评估报告 ==========
report = {
    'model_info': {
        'name': 'Warm-up KAN (GEF)',
        'architecture': config['width'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'features': all_features
    },
    'performance': {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    },
    'zero_analysis': {
        'zero_count': int(zero_mask.sum()) if zero_mask.any() else 0,
        'mean_pred_at_zero': float(zero_pred.mean()) if zero_mask.any() else 0,
        'max_pred_at_zero': float(zero_pred.max()) if zero_mask.any() else 0
    },
    'high_yield': {
        'threshold': float(high_thresh) if high_mask.any() else 0,
        'samples': int(high_mask.sum()) if high_mask.any() else 0,
        'r2': float(r2_high) if high_mask.any() else 0,
        'mse': float(mse_high) if high_mask.any() else 0
    },
    'visualization': vis_path,
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

json_path = 'results/warmup_eval/evaluation_report.json'
with open(json_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"  ✓ Report saved: {json_path}")

# 打印总结
print("\n" + "="*60)
print("EVALUATION COMPLETED")
print("="*60)
print(f"🎯 R² Score: {r2:.4f}")
print(f"📊 MSE: {mse:.3e}, MAE: {mae:.3e}")
print(f"📈 High-yield R²: {r2_high:.4f}" if high_mask.any() else "")
print(f"📁 Results saved in results/warmup_eval/")
print("="*60)