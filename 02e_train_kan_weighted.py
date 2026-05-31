"""
02e_train_kan_weighted.py
高效特征工程版KAN训练 - 高产额区加权
"""

import joblib
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from kan import KAN
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型训练 - 特征工程增强版（高效CPU优化）")
print("="*60)

# ========== 1. 加载并转换数据 ==========
print("\n[1/6] 加载数据并提取物理特征...")

# 加载预处理数据
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
device = data['device']

# 加载scaler用于反归一化
print("  加载归一化参数...")
try:
    scaler_Z = joblib.load('data/standard_scalerZ.pkl')
    scaler_A = joblib.load('data/standard_scalerA.pkl')
    scaler_E = joblib.load('data/standard_scalerE.pkl')
    scaler_Y = joblib.load('data/yield_scaler.pkl')
    print("  ✓ 所有scaler加载成功")
except Exception as e:
    print(f"  ✗ 加载scaler失败: {e}")
    exit(1)

# 反归一化获取物理值（批量向量化计算）
print("\n  反归一化获取物理值...")
# 训练集
Z_train_phy = scaler_Z.inverse_transform(X_train[:, 0:1]).flatten()
A_train_phy = scaler_A.inverse_transform(X_train[:, 1:2]).flatten()
E_train_phy = scaler_E.inverse_transform(X_train[:, 2:3]).flatten()

# 验证集
Z_val_phy = scaler_Z.inverse_transform(X_val[:, 0:1]).flatten()
A_val_phy = scaler_A.inverse_transform(X_val[:, 1:2]).flatten()
E_val_phy = scaler_E.inverse_transform(X_val[:, 2:3]).flatten()

# 计算中子数
N_train_phy = A_train_phy - Z_train_phy
N_val_phy = A_val_phy - Z_val_phy

print(f"  物理值统计:")
print(f"    - Z范围: [{Z_train_phy.min():.1f}, {Z_train_phy.max():.1f}]")
print(f"    - A范围: [{A_train_phy.min():.1f}, {A_train_phy.max():.1f}]")
print(f"    - N范围: [{N_train_phy.min():.1f}, {N_train_phy.max():.1f}]")

# ========== 2. 高效特征工程 ==========
print("\n[2/6] 构建增强物理特征（向量化计算）...")

# 定义幻数数组
magic_numbers = np.array([2, 8, 20, 28, 50, 82, 126], dtype=np.float32)
sigma = 2.0

def compute_magic_features_vectorized(values, magic_nums=magic_numbers, sigma_val=sigma):
    """向量化计算幻数相关特征"""
    # 广播计算距离矩阵
    dist_matrix = np.abs(values[:, np.newaxis] - magic_nums[np.newaxis, :])
    
    # 计算到最近幻数的距离
    min_dist = dist_matrix.min(axis=1)
    
    # 计算幻数接近度（高斯核）
    gaussian_kernel = np.exp(-dist_matrix**2 / (2 * sigma_val**2))
    proximity = gaussian_kernel.sum(axis=1)
    
    # 壳闭合标志 - 返回浮点数
    shell_closure = (min_dist < 2).astype(np.float32)
    
    return min_dist, proximity, shell_closure

# 计算训练集特征
print("  计算训练集特征...")
start_time = time.time()

# 奇偶性特征
Z_train_parity = (Z_train_phy % 2).astype(np.float32)
N_train_parity = (N_train_phy % 2).astype(np.float32)
parity_train_product = Z_train_parity * N_train_parity

# 幻数特征
Z_train_magic_dist, Z_train_magic_prox, Z_train_shell = compute_magic_features_vectorized(Z_train_phy)
N_train_magic_dist, N_train_magic_prox, N_train_shell = compute_magic_features_vectorized(N_train_phy)
any_shell_train = np.logical_or(Z_train_shell, N_train_shell).astype(np.float32)

# 其他物理特征
N_over_Z_train = N_train_phy / (Z_train_phy + 1e-12)
symmetry_energy_train = (N_train_phy - Z_train_phy)**2 / (4 * A_train_phy)
mass_excess_train = A_train_phy - 2 * Z_train_phy

# 计算验证集特征
print("  计算验证集特征...")
Z_val_parity = (Z_val_phy % 2).astype(np.float32)
N_val_parity = (N_val_phy % 2).astype(np.float32)
parity_val_product = Z_val_parity * N_val_parity

Z_val_magic_dist, Z_val_magic_prox, Z_val_shell = compute_magic_features_vectorized(Z_val_phy)
N_val_magic_dist, N_val_magic_prox, N_val_shell = compute_magic_features_vectorized(N_val_phy)
any_shell_val = np.logical_or(Z_val_shell, N_val_shell).astype(np.float32)

N_over_Z_val = N_val_phy / (Z_val_phy + 1e-12)
symmetry_energy_val = (N_val_phy - Z_val_phy)**2 / (4 * A_val_phy)
mass_excess_val = A_val_phy - 2 * Z_val_phy

# 选择最重要的特征（基于之前的统计分析）
selected_features = [
    'N_over_Z',           # 最重要的正相关特征
    'symmetry_energy',    # 最重要的负相关特征
    'Z_magic_dist',       # 幻数距离特征
    'any_shell',          # 壳闭合标志
    'Z_parity',           # 质子奇偶性
]

# 构建特征字典
train_features = {
    'N_over_Z': N_over_Z_train,
    'symmetry_energy': symmetry_energy_train,
    'Z_magic_dist': Z_train_magic_dist,
    'any_shell': any_shell_train,
    'Z_parity': Z_train_parity,
}

val_features = {
    'N_over_Z': N_over_Z_val,
    'symmetry_energy': symmetry_energy_val,
    'Z_magic_dist': Z_val_magic_dist,
    'any_shell': any_shell_val,
    'Z_parity': Z_val_parity,
}

# 归一化新特征（使用训练集的统计量）
print("  归一化新特征...")
for feat_name in selected_features:
    feat_train = train_features[feat_name]
    feat_val = val_features[feat_name]
    
    # 计算训练集的均值和标准差
    feat_mean = feat_train.mean()
    feat_std = feat_train.std() + 1e-12
    
    # 归一化
    train_features[feat_name] = (feat_train - feat_mean) / feat_std
    val_features[feat_name] = (feat_val - feat_mean) / feat_std

# 合并特征
print("  合并特征...")
X_train_augmented = [X_train]  # 基础特征
X_val_augmented = [X_val]    # 基础特征

for feat_name in selected_features:
    X_train_augmented.append(train_features[feat_name].reshape(-1, 1))
    X_val_augmented.append(val_features[feat_name].reshape(-1, 1))

X_train_augmented = np.hstack(X_train_augmented)
X_val_augmented = np.hstack(X_val_augmented)

feature_time = time.time() - start_time
print(f"\n  ✓ 特征工程完成，用时: {feature_time:.2f}秒")
print(f"  特征维度: 从{X_train.shape[1]}增加到{X_train_augmented.shape[1]}")
print(f"  使用特征: 原始(Z,A,E) + {len(selected_features)}个新特征")
print(f"  新特征列表: {', '.join(selected_features)}")

# ========== 3. 对数变换处理零值 ==========
print("\n[3/6] 应用对数变换...")

# 原始Yield统计
print("  原始Yield统计:")
print(f"    - 零值数量: {np.sum(y_train == 0)}")
print(f"    - 最小值(非零): {y_train[y_train > 0].min():.2e}")
print(f"    - 最大值: {y_train.max():.2e}")

# 对数变换
epsilon = 1e-12
y_train_log = np.log10(y_train + epsilon)
y_val_log = np.log10(y_val + epsilon)

print("\n  对数变换后:")
print(f"    - 范围: [{y_train_log.min():.3f}, {y_train_log.max():.3f}]")

# 转换为张量
X_train_t = torch.tensor(X_train_augmented, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_augmented, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_log, dtype=torch.float32).to(device)

# ========== 4. 构建增强模型 ==========
print("\n[4/6] 构建增强KAN模型...")

# 增强配置 - 加深加宽网络
input_dim = X_train_augmented.shape[1]
config = {
    'width': [input_dim, 16, 12, 8, 4, 1],  # 加深加宽: 从4层增加到5层，每层宽度增加
    'grid': 7,                          # 适当增加网格分辨率
    'k': 3,
    'seed': 42,
    'epochs': 300,                      # 增加训练轮数，让早停机制发挥作用
    'batch_size': 256,                  # 由于网络加深，适当减小批量大小
    'learning_rate': 0.015,             # 适当降低学习率
    'weight_decay': 1e-3,               # 调整正则化强度
    'patience': 50,                     # 早停耐心值
    'min_delta': 1e-6,                  # 最小改进阈值
}

print("  模型配置:")
print(f"    - 输入维度: {input_dim}")
print(f"    - 网络结构: {config['width']} (加深加宽)")
print(f"    - 网格大小: {config['grid']}")
print(f"    - 训练轮数: {config['epochs']} (配合早停)")
print(f"    - 批量大小: {config['batch_size']}")
print(f"    - 早停耐心: {config['patience']} epochs")

# 构建模型
model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])
model.to(device)

# 参数量
params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型构建完成，参数量: {params:,}")

# ========== 4.5. 计算加权函数统计量 ==========
print("\n[4.5/6] 计算加权函数统计量...")

# 计算训练集的对数产额统计量
y_train_log_calc = np.log10(y_train + epsilon)  # 重新计算，避免与之前的变量混淆

# 加权函数超参数
W_min = 1.0    # 最小权重
W_max = 3.0    # 最大权重
k = -0.5        # 标准差倍数（控制权重开始增长的阈值）

# 计算加权函数所需的统计量
μ = np.mean(y_train_log_calc)  # 均值
σ = np.std(y_train_log_calc)   # 标准差
M = np.max(y_train_log_calc)   # 最大值
threshold = μ + k * σ  # 计算阈值

print(f"  对数产额统计量:")
print(f"    - 均值 (μ): {μ:.6f}")
print(f"    - 标准差 (σ): {σ:.6f}")
print(f"    - 最大值 (M): {M:.6f}")
print(f"    - 阈值 (μ + {k}σ): {threshold:.6f}")

print(f"  加权函数超参数:")
print(f"    - W_min: {W_min}")
print(f"    - W_max: {W_max}")
print(f"    - k: {k}")

# 定义加权函数
def linear_weight_with_sigma(y_log, μ=μ, σ=σ, M=M, W_min=W_min, W_max=W_max, k=k):
    """
    使用标准差定义权重增长区间的线性加权函数
    
    参数:
        y_log: 样本的对数产额
        μ: 全体对数产额的均值
        σ: 全体对数产额的标准差
        M: 全体对数产额的最大值
        W_min: 最小权重（默认1.0）
        W_max: 最大权重
        k: 标准差倍数，定义权重开始增长的阈值
    """
    # 定义权重开始增长的阈值
    threshold = μ + k * σ
    
    if y_log <= threshold:
        weight = W_min
    else:
        # 在[threshold, M]区间内线性增长
        weight = np.clip(
            W_min + (W_max - W_min) * (y_log - threshold) / (M - threshold),
            W_min,
            W_max
        )
    
    return weight

# 测试加权函数
test_y_logs = np.linspace(y_train_log_calc.min(), y_train_log_calc.max(), 5)
print(f"\n  加权函数测试 (阈值={threshold:.3f}):")
for y_log in test_y_logs:
    weight = linear_weight_with_sigma(y_log)
    print(f"    y_log={y_log:.3f} -> weight={weight:.3f}")

# 计算训练集的平均权重
train_weights = np.array([linear_weight_with_sigma(y) for y in y_train_log_calc.flatten()])
avg_weight = np.mean(train_weights)
print(f"  训练集平均权重: {avg_weight:.3f}")

# 保存加权函数配置
weight_config = {
    'W_min': float(W_min),
    'W_max': float(W_max),
    'k': float(k),
    'μ': float(μ),
    'σ': float(σ),
    'M': float(M),
    'threshold': float(threshold),
    'avg_weight': float(avg_weight)
}

# ========== 5. 训练模型（CPU优化） ==========
# 修改损失函数定义
print("\n[5/6] 开始训练模型（带加权损失）...")

# 数据加载器 - 使用多个worker预取数据
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

# 设置num_workers=0以避免在Windows上的多进程问题
train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# 加权MSE损失函数
def weighted_mse_loss(y_pred_log, y_true_log, y_true_linear, μ=μ, σ=σ, M=M, W_min=W_min, W_max=W_max, k=k):
    """
    加权MSE损失函数
    """
    # 计算每个样本的对数产额
    y_log = np.log10(y_true_linear + epsilon)
    
    # 计算每个样本的权重
    weights = np.array([linear_weight_with_sigma(y, μ, σ, M, W_min, W_max, k) for y in y_log.flatten()])
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(y_pred_log.device)
    
    # 计算加权MSE
    losses = weights_tensor * (y_pred_log - y_true_log) ** 2
    return losses.mean()

# 在训练循环中使用加权损失
criterion = weighted_mse_loss  # 使用加权损失函数替代标准MSE

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay']
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.7,
    patience=10,
    min_lr=1e-6
)

# 训练记录
history = {'train_loss': [], 'val_loss': [], 'lr': []}
best_loss = float('inf')
best_epoch = 0
patience_counter = 0
start_time = time.time()

print("\n  开始训练（带早停）...")
for epoch in range(config['epochs']):
    epoch_start = time.time()
    
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # 获取批量数据的线性值用于计算权重
        batch_y_linear = 10**batch_y - epsilon
        
        # 使用加权损失
        loss = criterion(outputs, batch_y, batch_y_linear)
        loss.backward()
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
    
    avg_train = train_loss / train_batches
    history['train_loss'].append(avg_train)
    
    # 验证阶段（同样使用加权损失）
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            # 获取批量数据的线性值用于计算权重
            batch_y_linear = 10**batch_y - epsilon
            loss = criterion(outputs, batch_y, batch_y_linear)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val = val_loss / val_batches
    history['val_loss'].append(avg_val)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # 学习率调整
    scheduler.step(avg_val)
    
    # 检查是否是最佳模型
    if avg_val < best_loss - config['min_delta']:
        best_loss = avg_val
        best_epoch = epoch + 1
        patience_counter = 0
        
        # 保存最佳模型（包含加权配置）
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'config': config,
            'input_dim': input_dim,
            'feature_names': ['Z', 'A', 'E'] + selected_features,
            'selected_features': selected_features,
            'weight_config': weight_config  # 保存加权配置
        }, 'models/kan_feature_best.pth')
        
        print(f"    Best val_loss: {best_loss:.3e} (Epoch {best_epoch})")
    else:
        patience_counter += 1
        # 早停检查
        if patience_counter >= config['patience']:
            print(f"\n  ⏹️  早停触发: 连续{config['patience']}个epoch验证损失未改善")
            print(f"     最佳验证损失: {best_loss:.3e} (Epoch {best_epoch})")
            break
    
    # 打印进度
    epoch_time = time.time() - epoch_start
    if (epoch + 1) % 10 == 0 or epoch < 5 or epoch + 1 == config['epochs']:
        status = "⏹️ 早停" if patience_counter >= config['patience'] else f"Patience{patience_counter}/{config['patience']}"
        print(f"    Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {avg_train:.3e} | Val: {avg_val:.3e} | "
              f"LR: {history['lr'][-1]:.3e} | {status} | Time: {epoch_time:.1f}s")

# 训练时间
train_time = time.time() - start_time
print(f"\n  ✓ 训练完成，总用时: {train_time:.1f}秒")
print(f"    实际训练轮数: {epoch+1}")
print(f"    最佳验证损失: {best_loss:.3e} (Epoch {best_epoch})")
print(f"    平均每轮: {train_time/(epoch+1):.1f}秒")

# ========== 6. 保存结果 ==========
print("\n[6/6] 保存增强模型和结果...")

# 加载最佳模型用于最终状态保存
checkpoint = torch.load('models/kan_feature_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])

# 保存最终模型
final_state = {
    'model_state': model.state_dict(),
    'config': config,
    'history': history,
    'best_loss': best_loss,
    'best_epoch': best_epoch,
    'early_stopped': patience_counter >= config['patience'],
    'final_epoch': epoch + 1,
    'patience_counter': patience_counter,
    'train_time': train_time,
    'data_transform': 'log10(y + 1e-12)',
    'epsilon': epsilon,
    'input_dim': input_dim,
    'feature_names': ['Z', 'A', 'E'] + selected_features,
    'selected_features': selected_features,
    'weight_config': weight_config,  # 包含加权配置
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
torch.save(final_state, 'models/kan_feature_final.pth')

# 保存训练历史
import json
history_data = {
    'train_loss': [float(x) for x in history['train_loss']],
    'val_loss': [float(x) for x in history['val_loss']],
    'learning_rate': [float(x) for x in history['lr']],
    'best_epoch': int(best_epoch),
    'best_loss': float(best_loss),
    'early_stopped': patience_counter >= config['patience'],
    'final_epoch': int(epoch + 1),
    'patience_counter': int(patience_counter),
    'config': config,
    'feature_info': {
        'total_features': int(input_dim),
        'original_features': 3,
        'additional_features': int(input_dim - 3),
        'selected_features': selected_features,
        'feature_names': ['Z', 'A', 'E'] + selected_features
    },
    'log_transform_info': {
        'epsilon': epsilon,
        'y_train_log_range': [float(y_train_log_calc.min()), float(y_train_log_calc.max())],
        'original_zero_count': int(np.sum(y_train == 0))
    },
    'performance_stats': {
        'total_train_time': float(train_time),
        'avg_epoch_time': float(train_time / (epoch + 1)),
        'batch_size': int(config['batch_size']),
        'early_stop_patience': int(config['patience'])
    },
    'weight_config': weight_config  # 在训练历史中也保存加权配置
}
with open('models/feature_training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"  ✓ 模型保存: models/kan_feature_final.pth")
print(f"  ✓ 训练历史: models/feature_training_history.json")

# 快速验证
print("\n[快速验证] 验证特征增强效果...")

# 显示训练总结
if patience_counter >= config['patience']:
    print(f"  ⏹️  训练因早停而终止 (耐心值: {config['patience']})")
    print(f"     实际训练轮数: {epoch+1}/{config['epochs']}")
else:
    print(f"  ✓ 训练完成所有{config['epochs']}轮")

print(f"     最佳验证损失在第{best_epoch}轮达到: {best_loss:.3e}")

# 加载最佳模型
checkpoint = torch.load('models/kan_feature_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 在验证集上预测
with torch.no_grad():
    y_pred_log = model(X_val_t).cpu().numpy()

# 反变换到原始尺度
y_pred_linear = 10**y_pred_log - epsilon
y_val_linear = 10**y_val_t.cpu().numpy() - epsilon

# 计算原始尺度的误差
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse_original = mean_squared_error(y_val_linear, y_pred_linear)
mae_original = mean_absolute_error(y_val_linear, y_pred_linear)
r2_original = r2_score(y_val_linear, y_pred_linear)

# 高产额区域分析
high_yield_threshold = np.percentile(y_val_linear, 70)  # 前30%为高产额
high_yield_mask = y_val_linear >= high_yield_threshold
if np.sum(high_yield_mask) > 0:
    y_val_high = y_val_linear[high_yield_mask]
    y_pred_high = y_pred_linear[high_yield_mask]
    r2_high = r2_score(y_val_high, y_pred_high)
    mse_high = mean_squared_error(y_val_high, y_pred_high)
    mae_high = mean_absolute_error(y_val_high, y_pred_high)
else:
    r2_high = 0
    mse_high = 0
    mae_high = 0

print("\n  特征增强模型性能:")
print(f"    - 整体R²: {r2_original:.4f}")
print(f"    - 整体MSE: {mse_original:.3e}")
print(f"    - 整体MAE: {mae_original:.3e}")
print(f"    - 高产额R²: {r2_high:.4f}")
print(f"    - 高产额MSE: {mse_high:.3e}")
print(f"    - 高产额MAE: {mae_high:.3e}")

# 与基线对比
baseline_mse = 1.66e-02
improvement = (baseline_mse - mse_original) / baseline_mse * 100
print(f"\n  与基线对比:")
print(f"    - 基线MSE: {baseline_mse:.3e}")
print(f"    - 改进MSE: {mse_original:.3e}")
print(f"    - 相对改进: {improvement:+.1f}%")

print("\n" + "="*60)
print("特征增强训练完成！")
print("="*60)
print("关键改进点:")
print(f"1. ✅ 网络结构加深加宽: KAN{config['width']}")
print(f"2. ✅ 引入早停机制: 耐心值={config['patience']} epochs")
print(f"3. ✅ 实际训练轮数: {epoch+1}/{config['epochs']}")
if patience_counter >= config['patience']:
    print(f"4. ✅ 早停触发: 连续{config['patience']}轮验证损失未改善")
print(f"5. ✅ 加权损失函数: W_max={W_max}, k={k}")
print(f"6. ✅ 加权平均权重: {avg_weight:.3f}")
print("\n加权函数配置:")
for key, value in weight_config.items():
    if key != 'avg_weight':
        print(f"  - {key}: {value}")
print("="*60)