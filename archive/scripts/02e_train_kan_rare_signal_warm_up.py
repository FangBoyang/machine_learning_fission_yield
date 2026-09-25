"""
02e_train_kan_warm_up.py
功能: 在GEF理论数据的原始归一化空间上进行warm-up训练（直接使用归一化值）
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型warm-up训练 - 原始归一化空间，标准MSE损失")
print("="*60)

# ========== 1. 加载GEF预处理数据 ==========
print("\n[1/7] 加载GEF预处理数据（warm-up阶段）...")

try:
    with open('preprocessed_gef_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train_gef, y_train_gef = data['X_train'], data['y_train']
    device = data['device']
    
    print(f"  ✓ 数据加载成功")
    print(f"    训练集: {X_train_gef.shape[0]} 样本")
    print(f"    设备: {device}")
    
    # 划分验证集
    val_ratio = 0.2
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_gef, y_train_gef, test_size=val_ratio, random_state=42, shuffle=True
    )
    
    print(f"  ✓ 划分验证集: {X_val.shape[0]} 样本 (20%)")
    print(f"  ✓ 最终训练集: {X_train.shape[0]} 样本")
    
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    exit(1)

# ========== 2. 数据统计 ==========
print("\n[2/7] 数据统计（归一化空间）...")
print(f"    X范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"    y范围: [{y_train.min():.3f}, {y_train.max():.3f}]")
print(f"    y均值: {y_train.mean():.3f}, 标准差: {y_train.std():.3f}")

# ========== 3. 特征工程（仅添加物理特征，不反归一化） ==========
print("\n[3/7] 构建增强物理特征（基于归一化值）...")

# 注意：由于数据已经归一化，我们直接使用归一化后的Z,A,E来计算物理特征
# 但物理特征如N_over_Z需要原始物理值才能有意义。这里我们采用另一种思路：
# 直接使用原始归一化特征（Z,A,E）加上从归一化值推导的近似特征。
# 实际上，归一化后的Z,A,E已经包含了线性信息，我们只需添加非线性组合。

# 方案：使用归一化后的Z,A,E直接构造二次交互特征
print("  基于归一化特征构造交互项...")
start_time = time.time()

# 提取归一化后的特征
Z_norm = X_train[:, 0:1]
A_norm = X_train[:, 1:2]
E_norm = X_train[:, 2:3]

Z_val_norm = X_val[:, 0:1]
A_val_norm = X_val[:, 1:2]
E_val_norm = X_val[:, 2:3]

# 构造交互特征（在归一化空间）
interaction_features = [
    'Z_squared', 'A_squared', 'E_squared',  # 平方项
    'ZxA', 'ZxE', 'AxE'                     # 交叉项
]

train_interactions = np.hstack([
    Z_norm**2, A_norm**2, E_norm**2,
    Z_norm * A_norm, Z_norm * E_norm, A_norm * E_norm
])

val_interactions = np.hstack([
    Z_val_norm**2, A_val_norm**2, E_val_norm**2,
    Z_val_norm * A_val_norm, Z_val_norm * E_val_norm, A_val_norm * E_val_norm
])

# 归一化交互特征（使用训练集统计）
interact_mean = train_interactions.mean(axis=0)
interact_std = train_interactions.std(axis=0) + 1e-12
train_interactions = (train_interactions - interact_mean) / interact_std
val_interactions = (val_interactions - interact_mean) / interact_std

# 合并特征
X_train_augmented = np.hstack([X_train, train_interactions])
X_val_augmented = np.hstack([X_val, val_interactions])

feature_time = time.time() - start_time
print(f"\n  ✓ 特征工程完成，用时: {feature_time:.2f}秒")
print(f"  特征维度: 从{X_train.shape[1]}增加到{X_train_augmented.shape[1]}")
print(f"  新增特征: {', '.join(interaction_features)}")

# ========== 4. 准备训练数据（直接使用归一化y，不取对数） ==========
print("\n[4/7] 准备训练数据（原始归一化空间）...")

X_train_t = torch.tensor(X_train_augmented, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_augmented, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

print(f"  ✓ 训练数据准备完成")
print(f"    - 训练集形状: {X_train_t.shape}")
print(f"    - 验证集形状: {X_val_t.shape}")
print(f"    - 目标范围: [{y_train.min():.3f}, {y_train.max():.3f}]")

# ========== 5. 构建KAN模型 ==========
print("\n[5/7] 构建KAN模型（原始归一化空间训练）...")

input_dim = X_train_augmented.shape[1]  # 3原始 + 6交互 = 9
config = {
    'width': [input_dim, 20, 16, 12, 8, 4, 1],  # 6层网络
    'grid': 7,
    'k': 3,
    'seed': 42,
    'epochs': 500,                     # 增加epochs
    'batch_size': 2048,                # 增大batch_size加速
    'learning_rate': 0.1,              # 使用之前证明有效的学习率
    'weight_decay': 1e-3,
    'patience': 50,                    # 增大耐心值
    'min_delta': 1e-6,
}

print("  模型配置:")
print(f"    - 输入维度: {input_dim}")
print(f"    - 网络结构: {config['width']} (6层)")
print(f"    - 训练空间: 原始归一化空间")
print(f"    - 损失函数: 标准MSE")
print(f"    - 训练轮数: {config['epochs']}")
print(f"    - 批量大小: {config['batch_size']}")
print(f"    - 早停耐心: {config['patience']} epochs")

model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])
model.to(device)

params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型构建完成，参数量: {params:,}")

criterion = nn.MSELoss()

# ========== 6. 训练模型 ==========
print("\n[6/7] 开始warm-up训练（GEF理论数据）...")

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

history = {'train_loss': [], 'val_loss': [], 'lr': []}
best_loss = float('inf')
best_epoch = 0
patience_counter = 0
start_time = time.time()

print("\n  开始warm-up训练（带早停）...")
for epoch in range(config['epochs']):
    epoch_start = time.time()
    
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
    
    avg_train = train_loss / train_batches
    history['train_loss'].append(avg_train)
    
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
    
    scheduler.step(avg_val)
    
    if avg_val < best_loss - config['min_delta']:
        best_loss = avg_val
        best_epoch = epoch + 1
        patience_counter = 0
        
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'config': config,
            'input_dim': input_dim,
            'feature_names': ['Z', 'A', 'E'] + interaction_features,
            'training_stage': 'warmup_raw_norm_gef'
        }, 'models/kan_warmup_raw_norm_best.pth')
        
        print(f"    Best val_loss: {best_loss:.3e} (Epoch {best_epoch})")
    else:
        patience_counter += 1
        if patience_counter >= config['patience']:
            print(f"\n  ⏹️  早停触发: 连续{config['patience']}个epoch验证损失未改善")
            print(f"     最佳验证损失: {best_loss:.3e} (Epoch {best_epoch})")
            break
    
    epoch_time = time.time() - epoch_start
    if (epoch + 1) % 10 == 0 or epoch < 5 or epoch + 1 == config['epochs']:
        status = "⏹️ 早停" if patience_counter >= config['patience'] else f"Patience: {patience_counter}/{config['patience']}"
        print(f"    Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {avg_train:.3e} | Val: {avg_val:.3e} | "
              f"LR: {history['lr'][-1]:.3e} | {status} | Time: {epoch_time:.1f}s")

train_time = time.time() - start_time
print(f"\n  ✓ warm-up训练完成，总用时: {train_time:.1f}秒")
print(f"    实际训练轮数: {epoch+1}")
print(f"    最佳验证损失: {best_loss:.3e} (Epoch {best_epoch})")
print(f"    平均每轮: {train_time/(epoch+1):.1f}秒")

# ========== 7. 保存最终结果 ==========
print("\n[7/7] 保存warm-up模型和结果...")

try:
    checkpoint = torch.load('models/kan_warmup_raw_norm_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print("  ✓ 加载最佳模型成功")
except Exception as e:
    print(f"  ✗ 加载模型失败: {e}")

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
        'input_dim': input_dim,
        'feature_names': ['Z', 'A', 'E'] + interaction_features,
        'training_stage': 'warmup_raw_norm_gef',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(final_state, 'models/kan_warmup_raw_norm_final.pth')
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
            'feature_names': ['Z', 'A', 'E'] + interaction_features
        },
        'data_info': {
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'training_space': 'raw_normalized',
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
    with open('models/warmup_raw_norm_training_history.json', 'w') as f:
        import json
        json.dump(history_data, f, indent=2)
    print(f"  ✓ 训练历史保存: models/warmup_raw_norm_training_history.json")
except Exception as e:
    print(f"  ✗ 保存训练历史失败: {e}")

# ========== 快速验证 ==========
print("\n[快速验证] 验证模型性能...")

if patience_counter >= config['patience']:
    print(f"  ⏹️  训练因早停而终止 (耐心值: {config['patience']})")
    print(f"     实际训练轮数: {epoch+1}/{config['epochs']}")
else:
    print(f"  ✓ 训练完成所有{config['epochs']}轮")

print(f"     最佳验证损失在第{best_epoch}轮达到: {best_loss:.3e}")

model.eval()
with torch.no_grad():
    y_pred = model(X_val_t).cpu().numpy()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\n  模型性能 (归一化空间):")
print(f"    - R²: {r2:.4f}")
print(f"    - MSE: {mse:.3e}")
print(f"    - MAE: {mae:.3e}")
print(f"    - 预测范围: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

print("\n" + "="*60)
print("KAN模型warm-up训练完成！")
print("="*60)
print("关键特性:")
print(f"1. ✅ 训练空间: 原始归一化空间（无对数变换）")
print(f"2. ✅ 损失函数: 标准MSE")
print(f"3. ✅ 网络结构: KAN{config['width']} (6层)")
print(f"4. ✅ 特征工程: 原始(Z,A,E) + 6个交互特征")
print(f"5. ✅ 学习率: 0.1, batch_size: 2048")
print(f"6. ✅ 实际训练轮数: {epoch+1}/{config['epochs']}")
print(f"7. ✅ 已保存模型参数，可用于后续235UALL.csv训练")
print("="*60)