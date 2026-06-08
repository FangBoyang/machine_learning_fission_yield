"""
02e_train_kan_warm_up_log.py
功能: 在GEF理论数据的对数上进行warm-up训练
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
print("KAN模型warm-up训练 - 对数空间，标准MSE损失")
print("="*60)

# ========== 1. 加载GEF预处理数据 ==========
print("\n[1/7] 加载GEF预处理数据（warm-up阶段）...")

try:
    # 加载GEF数据
    with open('preprocessed_gef_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # 提取数据
    X_train_gef, y_train_gef = data['X_train'], data['y_train']
    device = data['device']
    
    print(f"  ✓ 数据加载成功")
    print(f"    训练集: {X_train_gef.shape[0]} 样本")
    print(f"    设备: {device}")
    
    # 由于GEF数据没有验证集，我们需要从训练集中划分验证集
    from sklearn.model_selection import train_test_split
    val_ratio = 0.2  # 20%作为验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_gef, y_train_gef, test_size=val_ratio, random_state=42, shuffle=True
    )
    
    print(f"  ✓ 划分验证集: {X_val.shape[0]} 样本 (20%)")
    print(f"  ✓ 最终训练集: {X_train.shape[0]} 样本")
    
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    print("  请先运行 01e_data_loading_rare_signal_warm_up.py 生成GEF预处理数据")
    exit(1)

# ========== 2. 加载scaler用于反归一化 ==========
print("\n[2/7] 加载归一化参数...")

scaler_files = {
    'Z': 'data/standard_scalerZ.pkl',
    'A': 'data/standard_scalerA.pkl',
    'E': 'data/standard_scalerE.pkl',
    'Yield': 'data/yield_scaler.pkl'
}

scalers = {}
try:
    for name, filepath in scaler_files.items():
        # 使用joblib加载scaler文件
        scalers[name] = joblib.load(filepath)
    print("  ✓ 所有scaler加载成功")
except Exception as e:
    print(f"  ✗ 加载scaler失败: {e}")
    exit(1)

# ========== 3. 反归一化获取物理值 ==========
print("\n[3/7] 反归一化获取物理值...")

# 训练集物理值
Z_train_phy = scalers['Z'].inverse_transform(X_train[:, 0:1]).flatten()
A_train_phy = scalers['A'].inverse_transform(X_train[:, 1:2]).flatten()
E_train_phy = scalers['E'].inverse_transform(X_train[:, 2:3]).flatten()
y_train_phy = scalers['Yield'].inverse_transform(y_train).flatten()

# 验证集物理值
Z_val_phy = scalers['Z'].inverse_transform(X_val[:, 0:1]).flatten()
A_val_phy = scalers['A'].inverse_transform(X_val[:, 1:2]).flatten()
E_val_phy = scalers['E'].inverse_transform(X_val[:, 2:3]).flatten()
y_val_phy = scalers['Yield'].inverse_transform(y_val).flatten()

print(f"  物理值统计:")
print(f"    - Z范围: [{Z_train_phy.min():.1f}, {Z_train_phy.max():.1f}]")
print(f"    - A范围: [{A_train_phy.min():.1f}, {A_train_phy.max():.1f}]")
print(f"    - Yield范围: [{y_train_phy.min():.2e}, {y_train_phy.max():.2e}]")

# ========== 4. 应用对数变换 ==========
print("\n[4/7] 应用对数变换（在对数空间训练）...")

epsilon = 1e-12

# 对训练集和验证集的Yield应用对数变换
y_train_log = np.log10(y_train_phy + epsilon)
y_val_log = np.log10(y_val_phy + epsilon)

# 记录对数变换后的统计信息
print(f"  对数变换后统计:")
print(f"    - 训练集Yield_log范围: [{y_train_log.min():.3f}, {y_train_log.max():.3f}]")
print(f"    - 验证集Yield_log范围: [{y_val_log.min():.3f}, {y_val_log.max():.3f}]")
print(f"    - 训练集零值数量: {np.sum(y_train_phy == 0)}")
print(f"    - 验证集零值数量: {np.sum(y_val_phy == 0)}")

# ========== 5. 特征工程 ==========
print("\n[5/7] 构建增强物理特征（向量化计算）...")

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

# 计算训练集特征
print("  计算训练集特征...")
start_time = time.time()

# 计算中子数
N_train_phy = A_train_phy - Z_train_phy
N_val_phy = A_val_phy - Z_val_phy

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

# 选择最重要的特征
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

# ========== 6. 准备训练数据 ==========
print("\n[6/7] 准备训练数据（对数空间）...")

# 转换为张量
X_train_t = torch.tensor(X_train_augmented, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_augmented, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_log, dtype=torch.float32).to(device)

print(f"  ✓ 训练数据准备完成")
print(f"    - 训练集形状: {X_train_t.shape}")
print(f"    - 验证集形状: {X_val_t.shape}")

# ========== 7. 构建KAN模型 ==========
print("\n[7/7] 构建KAN模型（对数空间训练）...")

# 模型配置 - 使用较深的网络结构
input_dim = X_train_augmented.shape[1]  # 应该是8（3个原始特征+5个增强特征）
config = {
    'width': [input_dim, 20, 16, 12, 8, 4, 1],  # 6层网络，比22 * 22更深
    'grid': 7,                         # 网格点数
    'k': 3,
    'seed': 42,
    'epochs': 200,                     # warm-up阶段epochs
    'batch_size': 256,                 # 批量大小
    'learning_rate': 0.01,             # 对数空间可用较大学习率
    'weight_decay': 1e-3,
    'patience': 30,                    # 早停耐心值
    'min_delta': 1e-6,
}

print("  模型配置:")
print(f"    - 输入维度: {input_dim}")
print(f"    - 网络结构: {config['width']} (6层)")
print(f"    - 训练空间: 对数空间")
print(f"    - 损失函数: 标准MSE (不利用误差列)")
print(f"    - 训练轮数: {config['epochs']}")
print(f"    - 批量大小: {config['batch_size']}")
print(f"    - 早停耐心: {config['patience']} epochs")
print(f"    - 训练阶段: Warm-up (GEF理论数据)")

# 构建模型
model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])
model.to(device)

# 参数量
params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型构建完成，参数量: {params:,}")

# 使用标准MSE损失函数
criterion = nn.MSELoss()

# ========== 8. 训练模型（warm-up阶段） ==========
print("\n[开始训练] 开始warm-up训练（GEF理论数据）...")

# 数据加载器（不使用误差列）
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

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

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay']
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# 训练记录
history = {
    'train_loss': [], 
    'val_loss': [], 
    'lr': []
}
best_loss = float('inf')
best_epoch = 0
patience_counter = 0
start_time = time.time()

print("\n  开始warm-up训练（带早停）...")
for epoch in range(config['epochs']):
    epoch_start = time.time()
    
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
    
    avg_train = train_loss / train_batches
    history['train_loss'].append(avg_train)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
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
    
        # 保存最佳模型
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
           'training_stage': 'warmup_log_mse_gef'
        }, 'models/kan_warmup_log_mse_best.pth')
    
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
        status = "⏹️ 早停" if patience_counter >= config['patience'] else f"Patience: {patience_counter}/{config['patience']}"
        print(f"    Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {avg_train:.3e} | Val: {avg_val:.3e} | "
              f"LR: {history['lr'][-1]:.3e} | {status} | Time: {epoch_time:.1f}s")

# 训练时间
train_time = time.time() - start_time
print(f"\n  ✓ warm-up训练完成，总用时: {train_time:.1f}秒")
print(f"    实际训练轮数: {epoch+1}")
print(f"    最佳验证损失: {best_loss:.3e} (Epoch {best_epoch})")
print(f"    平均每轮: {train_time/(epoch+1):.1f}秒")

# ========== 9. 保存最终结果 ==========
print("\n[保存结果] 保存warm-up模型和结果...")

# 加载最佳模型
try:
    checkpoint = torch.load('models/kan_warmup_log_mse_best.pth', 
                           map_location=device, 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print("  ✓ 加载最佳模型成功")
except Exception as e:
    print(f"  ✗ 加载模型失败: {e}")

# 保存最终模型
print("\n  保存最终模型...")
try:
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
        'training_stage': 'warmup_log_mse_gef',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(final_state, 'models/kan_warmup_log_mse_final.pth')
    print("  ✓ 最终模型保存成功")
    
except Exception as e:
    print(f"  ✗ 保存最终模型失败: {e}")

# 保存训练历史
print("\n  保存训练历史到JSON...")
try:
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
        'data_info': {
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'training_space': 'logarithmic',
            'loss_function': 'standard_mse',
            'training_stage': 'warmup',
            'data_source': 'GEF_theoretical'
        },
        'performance_stats': {
            'total_train_time': float(train_time),
            'avg_epoch_time': float(train_time / (epoch + 1)),
            'batch_size': int(config['batch_size']),
            'early_stop_patience': int(config['patience'])
        }
    }
    
    with open('models/warmup_log_mse_training_history.json', 'w') as f:
        import json
        json.dump(history_data, f, indent=2)
    
    print(f"  ✓ 训练历史保存: models/warmup_log_mse_training_history.json")
    
except Exception as e:
    print(f"  ✗ 保存训练历史失败: {e}")

# ========== 10. 快速验证 ==========
print("\n[快速验证] 验证模型性能...")

# 显示训练总结
if patience_counter >= config['patience']:
    print(f"  ⏹️  训练因早停而终止 (耐心值: {config['patience']})")
    print(f"     实际训练轮数: {epoch+1}/{config['epochs']}")
else:
    print(f"  ✓ 训练完成所有{config['epochs']}轮")

print(f"     最佳验证损失在第{best_epoch}轮达到: {best_loss:.3e}")

# 加载最佳模型
model.eval()

# 在验证集上预测
with torch.no_grad():
    y_pred_log = model(X_val_t).cpu().numpy()

# 将预测结果转换回原始空间
y_pred_linear = 10**y_pred_log - epsilon

# 计算原始尺度的误差
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse_original = mean_squared_error(y_val_phy, y_pred_linear)
mae_original = mean_absolute_error(y_val_phy, y_pred_linear)
r2_original = r2_score(y_val_phy, y_pred_linear)

# 高产额区域分析
high_yield_threshold = np.percentile(y_val_phy, 70)  # 前30%为高产额
high_yield_mask = y_val_phy >= high_yield_threshold
if np.sum(high_yield_mask) > 0:
    y_val_high = y_val_phy[high_yield_mask]
    y_pred_high = y_pred_linear[high_yield_mask]
    r2_high = r2_score(y_val_high, y_pred_high)
    mse_high = mean_squared_error(y_val_high, y_pred_high)
    mae_high = mean_absolute_error(y_val_high, y_pred_high)
else:
    r2_high = 0
    mse_high = 0
    mae_high = 0

print("\n  模型性能 (对数空间训练，原始空间评估):")
print(f"    - 整体R²: {r2_original:.4f}")
print(f"    - 整体MSE: {mse_original:.3e}")
print(f"    - 整体MAE: {mae_original:.3e}")
print(f"    - 高产额R²: {r2_high:.4f}")
print(f"    - 高产额MSE: {mse_high:.3e}")
print(f"    - 高产额MAE: {mae_high:.3e}")

# 预测范围检查
print(f"\n  预测范围检查:")
print(f"    - 真实值范围: [{y_val_phy.min():.2e}, {y_val_phy.max():.2e}]")
print(f"    - 预测值范围: [{y_pred_linear.min():.2e}, {y_pred_linear.max():.2e}]")
print(f"    - 负预测数量: {np.sum(y_pred_linear < 0)} ({np.sum(y_pred_linear < 0)/len(y_pred_linear)*100:.1f}%)")

# 对数空间性能
mse_log = mean_squared_error(y_val_log, y_pred_log)
r2_log = r2_score(y_val_log, y_pred_log)
print(f"\n  对数空间性能:")
print(f"    - 对数空间R²: {r2_log:.4f}")
print(f"    - 对数空间MSE: {mse_log:.3e}")

print("\n" + "="*60)
print("KAN模型warm-up训练完成！")
print("="*60)
print("关键特性:")
print(f"1. ✅ 训练空间: 对数空间 (log10(y + 1e-12))")
print(f"2. ✅ 损失函数: 标准MSE (不使用误差列)")
print(f"3. ✅ 网络结构: KAN{config['width']} (6层)")
print(f"4. ✅ 特征工程: 原始(Z,A,E) + {len(selected_features)}个增强特征")
print(f"5. ✅ 实际训练轮数: {epoch+1}/{config['epochs']}")
print(f"6. ✅ 已保存模型参数，可用于后续235UALL.csv训练")
print("\n后续步骤:")
print(f"  1. 加载模型: models/kan_warmup_log_mse_final.pth")
print(f"  2. 在235UALL.csv数据上继续训练 (使用相同对数空间设置)")
print("="*60)