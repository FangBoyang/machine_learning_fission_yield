"""
03c_evaluate_kan_feature_engineering.py
功能：评估特征工程增强版KAN模型
注意：此脚本专门用于评估使用了额外物理特征的KAN模型
"""

import time
import joblib
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from kan import KAN

print("="*60)
print("KAN Feature-Enhanced Model Evaluation")
print("="*60)

# ========== 1. 加载数据、模型和归一化参数 ==========
print("\n[1/6] Loading data, model and scalers...")

# 加载预处理数据
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
device = data['device']
epsilon = 1e-12

print(f"  ✓ Data loaded")
print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")
print(f"    Device: {device}")

# 加载scaler用于反归一化
print("\n  Loading scalers for denormalization...")
try:
    scaler_Z = joblib.load('data/standard_scalerZ.pkl')
    scaler_A = joblib.load('data/standard_scalerA.pkl')
    scaler_E = joblib.load('data/standard_scalerE.pkl')
    scaler_Y = joblib.load('data/yield_scaler.pkl')
    print("  ✓ All scalers loaded successfully")
except Exception as e:
    print(f"  ✗ Failed to load scalers: {e}")
    exit(1)

# 加载特征增强模型
model_path = "models/kan_feature_final.pth"
if not os.path.exists(model_path):
    model_path = "models/kan_feature_best.pth"

if not os.path.exists(model_path):
    print(f"  ✗ Feature-enhanced model file not found: {model_path}")
    exit(1)

checkpoint = torch.load(model_path, map_location=device)
print(f"  ✓ Feature-enhanced model loaded: {model_path}")

# 获取模型配置和特征信息
config = checkpoint['config']
selected_features = checkpoint.get('selected_features', 
                                  ['N_over_Z', 'symmetry_energy', 'Z_magic_dist', 'any_shell', 'Z_parity'])
feature_names = checkpoint.get('feature_names', 
                              ['Z', 'A', 'E'] + selected_features)

# 重新构建模型
model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])

# 修改这里：使用正确的键名加载模型状态
# 特征增强模型保存时使用的是'model_state'，基线模型可能使用'model_state_dict'
# 添加兼容性检查
if 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
    print(f"    Loaded model state with key: 'model_state'")
elif 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Loaded model state with key: 'model_state_dict'")
else:
    # 如果两个键都不存在，尝试检查checkpoint中可用的键
    print(f"  ✗ Could not find model state in checkpoint")
    print(f"    Available keys: {list(checkpoint.keys())}")
    exit(1)

model.to(device)
model.eval()

print(f"    Model structure: KAN{config['width']}")
print(f"    Input dimension: {config['width'][0]}")
print(f"    Selected features: {', '.join(selected_features)}")

# ========== 2. 特征工程：构建增强特征 ==========
print("\n[2/6] Building enhanced features for test set...")

# 反归一化获取测试集物理值
Z_test_phy = scaler_Z.inverse_transform(X_test[:, 0:1]).flatten()
A_test_phy = scaler_A.inverse_transform(X_test[:, 1:2]).flatten()
E_test_phy = scaler_E.inverse_transform(X_test[:, 2:3]).flatten()
N_test_phy = A_test_phy - Z_test_phy

# 加载训练集的物理值用于归一化新特征
Z_train_phy = scaler_Z.inverse_transform(X_train[:, 0:1]).flatten()
A_train_phy = scaler_A.inverse_transform(X_train[:, 1:2]).flatten()
N_train_phy = A_train_phy - Z_train_phy

# 定义幻数数组
magic_numbers = np.array([2, 8, 20, 28, 50, 82, 126], dtype=np.float32)
sigma = 2.0

def compute_magic_features_vectorized(values, magic_nums=magic_numbers, sigma_val=sigma):
    """向量化计算幻数相关特征"""
    dist_matrix = np.abs(values[:, np.newaxis] - magic_nums[np.newaxis, :])
    min_dist = dist_matrix.min(axis=1)
    gaussian_kernel = np.exp(-dist_matrix**2 / (2 * sigma_val**2))
    proximity = gaussian_kernel.sum(axis=1)
    shell_closure = (min_dist < 2).astype(np.float32)
    return min_dist, proximity, shell_closure

# 计算测试集新特征
Z_test_parity = (Z_test_phy % 2).astype(np.float32)
N_test_parity = (N_test_phy % 2).astype(np.float32)
parity_test_product = Z_test_parity * N_test_parity

Z_test_magic_dist, Z_test_magic_prox, Z_test_shell = compute_magic_features_vectorized(Z_test_phy)
N_test_magic_dist, N_test_magic_prox, N_test_shell = compute_magic_features_vectorized(N_test_phy)
any_shell_test = np.logical_or(Z_test_shell, N_test_shell).astype(np.float32)

N_over_Z_test = N_test_phy / (Z_test_phy + 1e-12)
symmetry_energy_test = (N_test_phy - Z_test_phy)**2 / (4 * A_test_phy)
mass_excess_test = A_test_phy - 2 * Z_test_phy

# 构建特征字典
test_features_raw = {
    'N_over_Z': N_over_Z_test,
    'symmetry_energy': symmetry_energy_test,
    'Z_magic_dist': Z_test_magic_dist,
    'any_shell': any_shell_test,
    'Z_parity': Z_test_parity,
}

# 使用训练集的统计量归一化测试集新特征
print("  Normalizing new features using training set statistics...")

# 需要计算训练集的统计量
# 计算训练集特征
Z_train_parity = (Z_train_phy % 2).astype(np.float32)
N_train_parity = (N_train_phy % 2).astype(np.float32)
Z_train_magic_dist, Z_train_magic_prox, Z_train_shell = compute_magic_features_vectorized(Z_train_phy)
N_train_magic_dist, N_train_magic_prox, N_train_shell = compute_magic_features_vectorized(N_train_phy)
any_shell_train = np.logical_or(Z_train_shell, N_train_shell).astype(np.float32)
N_over_Z_train = N_train_phy / (Z_train_phy + 1e-12)
symmetry_energy_train = (N_train_phy - Z_train_phy)**2 / (4 * A_train_phy)

# 构建训练集特征字典用于计算统计量
train_features_raw = {
    'N_over_Z': N_over_Z_train,
    'symmetry_energy': symmetry_energy_train,
    'Z_magic_dist': Z_train_magic_dist,
    'any_shell': any_shell_train,
    'Z_parity': Z_train_parity,
}

# 归一化测试集特征
test_features_normalized = {}
for feat_name in selected_features:
    if feat_name in train_features_raw:
        feat_train = train_features_raw[feat_name]
        feat_test = test_features_raw[feat_name]
        
        # 计算训练集的均值和标准差
        feat_mean = feat_train.mean()
        feat_std = feat_train.std() + 1e-12
        
        # 归一化
        test_features_normalized[feat_name] = (feat_test - feat_mean) / feat_std
    else:
        print(f"  Warning: Feature '{feat_name}' not found in training set")

# 合并特征
print("  Combining features...")
X_test_augmented = [X_test]  # 基础特征

for feat_name in selected_features:
    if feat_name in test_features_normalized:
        X_test_augmented.append(test_features_normalized[feat_name].reshape(-1, 1))
    else:
        print(f"  Warning: Skipping missing feature '{feat_name}'")

X_test_augmented = np.hstack(X_test_augmented)
print(f"  ✓ Feature engineering completed. New shape: {X_test_augmented.shape}")

# ========== 3. 在测试集上预测 ==========
print("\n[3/6] Predicting on enhanced feature test set...")

# 转换为张量
X_test_tensor = torch.tensor(X_test_augmented, dtype=torch.float32).to(device)

# 预测（对数空间）
with torch.no_grad():
    y_pred_log = model(X_test_tensor).cpu().numpy()

# 反变换到原始尺度
y_pred_linear = 10**y_pred_log - epsilon
y_test_linear = y_test  # 注意：y_test已经是原始归一化尺度

# 展平数组
y_pred_linear = y_pred_linear.flatten()
y_test_linear = y_test_linear.flatten()

print(f"  ✓ Prediction completed")
print(f"    Prediction range: [{y_pred_linear.min():.2e}, {y_pred_linear.max():.2e}]")
print(f"    True range: [{y_test_linear.min():.2e}, {y_test_linear.max():.2e}]")

# ========== 4. 计算评估指标 ==========
print("\n[4/6] Calculating evaluation metrics...")

# 基本指标
mse = mean_squared_error(y_test_linear, y_pred_linear)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_linear, y_pred_linear)
r2 = r2_score(y_test_linear, y_pred_linear)

print(f"  📊 Model Performance (Original Scale):")
print(f"    Mean Squared Error (MSE): {mse:.3e}")
print(f"    Root Mean Squared Error (RMSE): {rmse:.3e}")
print(f"    Mean Absolute Error (MAE): {mae:.3e}")
print(f"    Coefficient of Determination (R²): {r2:.4f}")

# 与基线对比
print(f"\n  🔄 Comparison with Baseline (Improved KAN):")
print(f"    Baseline MSE: ~1.66e-02")
print(f"    Baseline R²: ~0.9218")
improvement_mse = (1.66e-02 - mse) / 1.66e-02 * 100
improvement_r2 = (r2 - 0.9218) / 0.9218 * 100
print(f"    MSE Improvement: {improvement_mse:+.1f}%")
print(f"    R² Change: {improvement_r2:+.1f}%")

# 零值预测分析
print(f"\n  🔍 Zero-value Analysis:")
zero_mask = (y_test_linear == 0)
if zero_mask.any():
    zero_count = zero_mask.sum()
    zero_predictions = y_pred_linear[zero_mask]
    
    mean_pred_zero = zero_predictions.mean()
    max_pred_zero = zero_predictions.max()
    
    print(f"    Zero values in test set: {zero_count}")
    print(f"    Average prediction in zero regions: {mean_pred_zero:.2e}")
    print(f"    Max prediction in zero regions: {max_pred_zero:.2e}")
    
    # 检查是否有明显的假阳性
    false_positive_threshold = 1e-10
    false_positives = (zero_predictions > false_positive_threshold).sum()
    false_positive_rate = false_positives / zero_count * 100
    print(f"    False positives(>{false_positive_threshold:.0e}): {false_positives} ({false_positive_rate:.1f}%)")

# 高产额区域分析
print(f"\n  📈 High-Yield Region Analysis:")
high_yield_threshold = np.percentile(y_test_linear, 70)  # 前30%为高产额
high_yield_mask = y_test_linear >= high_yield_threshold
if high_yield_mask.any():
    y_test_high = y_test_linear[high_yield_mask]
    y_pred_high = y_pred_linear[high_yield_mask]
    
    mse_high = mean_squared_error(y_test_high, y_pred_high)
    mae_high = mean_absolute_error(y_test_high, y_pred_high)
    r2_high = r2_score(y_test_high, y_pred_high)
    
    print(f"    High-yield threshold: {high_yield_threshold:.3e}")
    print(f"    High-yield samples: {high_yield_mask.sum()}")
    print(f"    High-yield MSE: {mse_high:.3e}")
    print(f"    High-yield MAE: {mae_high:.3e}")
    print(f"    High-yield R²: {r2_high:.4f}")

# 非零值预测分析
non_zero_mask = ~zero_mask
if non_zero_mask.any():
    y_test_nonzero = y_test_linear[non_zero_mask]
    y_pred_nonzero = y_pred_linear[non_zero_mask]
    
    mse_nonzero = mean_squared_error(y_test_nonzero, y_pred_nonzero)
    r2_nonzero = r2_score(y_test_nonzero, y_pred_nonzero)
    
    print(f"\n  📊 Non-zero Region Performance:")
    print(f"    Non-zero values: {non_zero_mask.sum()}")
    print(f"    Non-zero MSE: {mse_nonzero:.3e}")
    print(f"    Non-zero R²: {r2_nonzero:.4f}")

# ========== 5. 生成可视化图表 ==========
print("\n[5/6] Generating visualization charts...")

# 确保结果目录存在
os.makedirs("results/feature_enhanced", exist_ok=True)

# 设置matplotlib使用英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature-Enhanced KAN Model Evaluation', fontsize=16, fontweight='bold')

# 1. 预测值 vs 真实值散点图
ax1 = axes[0, 0]
scatter = ax1.scatter(y_test_linear, y_pred_linear, alpha=0.6, s=20, c='blue', edgecolors='white', linewidth=0.5)

# 添加对角线
max_val = max(y_test_linear.max(), y_pred_linear.max())
min_val = min(y_test_linear.min(), y_pred_linear.min())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal Prediction')

# 区分零值点
if zero_mask.any():
    ax1.scatter(y_test_linear[zero_mask], y_pred_linear[zero_mask], 
                alpha=0.8, s=30, c='red', edgecolors='white', linewidth=0.5, label='True Zero Values')

# 区分高产额点
if high_yield_mask.any():
    ax1.scatter(y_test_linear[high_yield_mask], y_pred_linear[high_yield_mask], 
                alpha=0.8, s=40, c='gold', edgecolors='black', linewidth=0.5, marker='^', label='High-Yield Samples')

ax1.set_xlabel('True Yield Values', fontsize=12)
ax1.set_ylabel('Predicted Yield Values', fontsize=12)
ax1.set_title('Predicted vs True Values (Log Scale)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# 添加R²文本
ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. 残差图
ax2 = axes[0, 1]
residuals = y_pred_linear - y_test_linear
ax2.scatter(y_pred_linear, residuals, alpha=0.6, s=20, c='green', edgecolors='white', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax2.set_xlabel('Predicted Yield Values', fontsize=12)
ax2.set_ylabel('Residuals (Predicted - True)', fontsize=12)
ax2.set_title('Residual Analysis', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# 添加残差统计
residual_mean = residuals.mean()
residual_std = residuals.std()
ax2.text(0.05, 0.95, f'Residual Mean: {residual_mean:.2e}\nResidual Std: {residual_std:.2e}', 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. 误差分布直方图
ax3 = axes[1, 0]
relative_error = np.abs(residuals) / (np.abs(y_test_linear) + 1e-12)
# 避免极端值影响可视化
relative_error_clipped = np.clip(relative_error, 0, 10)  # 裁剪到0-10倍

ax3.hist(relative_error_clipped, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.set_xlabel('Relative Error |Predicted-True|/|True|', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Relative Error Distribution', fontsize=14)
ax3.grid(True, alpha=0.3)

# 添加误差统计
median_error = np.median(relative_error)
p90_error = np.percentile(relative_error, 90)
ax3.axvline(median_error, color='r', linestyle='--', label=f'Median: {median_error:.2f}')
ax3.axvline(p90_error, color='orange', linestyle='--', label=f'90th Percentile: {p90_error:.2f}')
ax3.legend()

# 4. 按特征维度分析
ax4 = axes[1, 1]
features_all = ['Z', 'A', 'E'] + selected_features
colors = plt.cm.tab20(np.linspace(0, 1, len(features_all)))

for i, (feat, color) in enumerate(zip(features_all, colors)):
    if i < X_test_augmented.shape[1]:  # 确保不超过特征维度
        # 计算每个特征值的平均绝对误差
        unique_vals = np.unique(X_test_augmented[:, i])
        # 只取部分值避免过于密集
        if len(unique_vals) > 20:
            unique_vals = np.linspace(unique_vals.min(), unique_vals.max(), 20)
        
        mae_by_feat = []
        
        for val in unique_vals:
            # 找到接近该值的样本
            mask = (np.abs(X_test_augmented[:, i] - val) < 0.05)
            if mask.any() and mask.sum() > 5:  # 至少有5个样本
                mae_val = mean_absolute_error(y_test_linear[mask], y_pred_linear[mask])
                mae_by_feat.append(mae_val)
            else:
                mae_by_feat.append(np.nan)
        
        # 绘制，跳过NaN值
        valid_mask = ~np.isnan(mae_by_feat)
        if valid_mask.any():
            ax4.plot(unique_vals[valid_mask], np.array(mae_by_feat)[valid_mask], 
                    'o-', color=color, label=feat, alpha=0.7, markersize=4)

ax4.set_xlabel('Feature Values (Normalized)', fontsize=12)
ax4.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
ax4.set_title('Prediction Error by Feature Dimension', fontsize=14)
ax4.legend(fontsize=8, ncol=2, loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'results/feature_enhanced/kan_feature_evaluation_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved: {output_path}")

# ========== 6. 保存详细评估结果 ==========
print("\n[6/6] Saving evaluation results...")

# 创建评估报告
evaluation_report = {
    'model_info': {
        'name': 'Feature-Enhanced KAN Model',
        'architecture': config['width'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'selected_features': selected_features,
        'total_features': len(features_all)
    },
    'performance_metrics': {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'improvement_vs_baseline_mse': float(improvement_mse),
        'improvement_vs_baseline_r2': float(improvement_r2)
    },
    'zero_value_analysis': {
        'zero_count_test': int(zero_mask.sum()) if zero_mask.any() else 0,
        'mean_prediction_zero': float(mean_pred_zero) if zero_mask.any() else 0,
        'false_positive_rate': float(false_positive_rate) if zero_mask.any() else 0
    },
    'high_yield_performance': {
        'threshold': float(high_yield_threshold) if high_yield_mask.any() else 0,
        'sample_count': int(high_yield_mask.sum()) if high_yield_mask.any() else 0,
        'mse': float(mse_high) if high_yield_mask.any() else 0,
        'mae': float(mae_high) if high_yield_mask.any() else 0,
        'r2': float(r2_high) if high_yield_mask.any() else 0
    },
    'non_zero_performance': {
        'mse': float(mse_nonzero) if non_zero_mask.any() else 0,
        'r2': float(r2_nonzero) if non_zero_mask.any() else 0
    },
    'data_info': {
        'test_set_size': int(X_test.shape[0]),
        'enhanced_feature_count': len(selected_features),
        'total_feature_count': X_test_augmented.shape[1],
        'log_transform': 'log10(y + 1e-12)',
        'epsilon': float(epsilon)
    },
    'visualization_files': [
        output_path
    ],
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

# 保存为JSON
import json
json_path = 'results/feature_enhanced/feature_evaluation_report.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
print(f"  ✓ JSON report saved: {json_path}")

# 保存为文本报告
txt_path = 'results/feature_enhanced/feature_evaluation_report.txt'
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("FEATURE-ENHANCED KAN MODEL EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. MODEL INFORMATION\n")
    f.write(f"   Model Architecture: KAN{config['width']}\n")
    f.write(f"   Parameters: {evaluation_report['model_info']['parameters']:,}\n")
    f.write(f"   Selected Features: {', '.join(selected_features)}\n")
    f.write(f"   Total Features: {evaluation_report['model_info']['total_features']}\n\n")
    
    f.write("2. PERFORMANCE METRICS\n")
    f.write(f"   Mean Squared Error (MSE): {mse:.3e}\n")
    f.write(f"   Root Mean Squared Error (RMSE): {rmse:.3e}\n")
    f.write(f"   Mean Absolute Error (MAE): {mae:.3e}\n")
    f.write(f"   Coefficient of Determination (R²): {r2:.4f}\n")
    f.write(f"   MSE Improvement vs Baseline: {improvement_mse:+.1f}%\n")
    f.write(f"   R² Change vs Baseline: {improvement_r2:+.1f}%\n\n")
    
    f.write("3. ZERO-VALUE ANALYSIS\n")
    if zero_mask.any():
        f.write(f"   Zero values in test set: {zero_count}\n")
        f.write(f"   Average prediction in zero regions: {mean_pred_zero:.2e}\n")
        f.write(f"   False positive rate: {false_positive_rate:.1f}%\n\n")
    
    f.write("4. HIGH-YIELD REGION PERFORMANCE\n")
    if high_yield_mask.any():
        f.write(f"   High-yield threshold: {high_yield_threshold:.3e}\n")
        f.write(f"   High-yield samples: {high_yield_mask.sum()}\n")
        f.write(f"   High-yield MSE: {mse_high:.3e}\n")
        f.write(f"   High-yield MAE: {mae_high:.3e}\n")
        f.write(f"   High-yield R²: {r2_high:.4f}\n\n")
    
    f.write("5. NON-ZERO REGION PERFORMANCE\n")
    if non_zero_mask.any():
        f.write(f"   Non-zero MSE: {mse_nonzero:.3e}\n")
        f.write(f"   Non-zero R²: {r2_nonzero:.4f}\n\n")
    
    f.write("6. DATA INFORMATION\n")
    f.write(f"   Test set size: {X_test.shape[0]}\n")
    f.write(f"   Base features: Z, A, E\n")
    f.write(f"   Enhanced features: {', '.join(selected_features)}\n")
    f.write(f"   Data transformation: {evaluation_report['data_info']['log_transform']}\n\n")
    
    f.write("7. GENERATED FILES\n")
    for file in evaluation_report['visualization_files']:
        f.write(f"   - {file}\n")

print(f"  ✓ Text report saved: {txt_path}")

# ========== 7. 展示关键结果 ==========
print("\n" + "="*60)
print("FEATURE-ENHANCED MODEL EVALUATION COMPLETED")
print("="*60)
print(f"🎯 CORE PERFORMANCE:")
print(f"   R² Score: {r2:.4f}")
if r2 > 0.9218:
    print(f"   ✓ Improved from baseline (0.9218)")
else:
    print(f"   ⚠️ Lower than baseline (0.9218)")
print(f"   RMSE: {rmse:.3e}")
print(f"   MSE Improvement: {improvement_mse:+.1f}%")

print(f"\n📊 HIGH-YIELD REGION (Key Focus):")
if high_yield_mask.any():
    print(f"   High-yield R²: {r2_high:.4f}")
    if r2_high > 0.8220:  # 之前训练脚本报告的高产额R²
        print(f"   ✓ Improved from previous (0.8220)")
    else:
        print(f"   ⚠️ Same as or lower than previous (0.8220)")

print(f"\n🔍 FEATURE ANALYSIS:")
print(f"   Original features: 3 (Z, A, E)")
print(f"   Enhanced features: {len(selected_features)}")
print(f"   Total features: {len(features_all)}")

print(f"\n📈 PERFORMANCE SUMMARY:")
if improvement_mse > 0:
    print(f"   ✅ MSE improved by {improvement_mse:.1f}%")
else:
    print(f"   ⚠️ MSE decreased by {-improvement_mse:.1f}%")

if improvement_r2 > 0:
    print(f"   ✅ R² improved by {improvement_r2:.1f}%")
else:
    print(f"   ⚠️ R² decreased by {-improvement_r2:.1f}%")

print(f"\n📁 GENERATED FILES:")
print(f"   1. Visualization: {output_path}")
print(f"   2. JSON report: {json_path}")
print(f"   3. Text report: {txt_path}")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
if improvement_r2 < 0:
    print("1. ⚠️ Feature enhancement did NOT improve overall R²")
    print("2. Consider feature selection or different feature combinations")
    print("3. Focus on high-yield region improvement strategies")
else:
    print("1. ✅ Feature enhancement improved model performance")
    print("2. Consider exploring additional physics-informed features")
    print("3. Further optimize for high-yield region")

print("="*60)