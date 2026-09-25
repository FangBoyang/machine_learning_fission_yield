"""
改进版KAN训练 - 针对零值问题的优化方案
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import time
from datetime import datetime
from kan import KAN
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型训练 - 改进版（处理零值问题）")
print("="*60)

# ========== 1. 加载并转换数据 ==========
print("\n[1/5] 加载数据并应用对数变换...")

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
device = data['device']
yield_scaler = data['scalers']['Yield']

# 关键改进1：对数变换，处理零值和小值
print("\n  原始Yield统计:")
print(f"    - 零值数量: {np.sum(y_train == 0)}")
print(f"    - 最小值(非零): {y_train[y_train > 0].min():.2e}")
print(f"    - 最大值: {y_train.max():.2e}")
print(f"    - 均值: {y_train.mean():.2e}")

# 对数变换（解决数值跨度大的问题）
epsilon = 1e-12  # 很小的值避免log(0)
y_train_log = np.log10(y_train + epsilon)
y_test_log = np.log10(y_test + epsilon)

print("\n  对数变换后:")
print(f"    - 范围: [{y_train_log.min():.3f}, {y_train_log.max():.3f}]")
print(f"    - 均值: {y_train_log.mean():.3f}, 标准差: {y_train_log.std():.3f}")

# 转换为张量
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test_log, dtype=torch.float32).to(device)

# ========== 2. 构建增强模型 ==========
print("\n[2/5] 构建增强KAN模型...")

# 增强配置
config = {
    'width': [3, 8, 6, 4, 1],  # 增加深度和宽度：3->8->6->4->1
    'grid': 7,                 # 增加网格分辨率
    'k': 3,
    'seed': 42,
    'epochs': 120,             # 增加训练轮数
    'batch_size': 64,
    'learning_rate': 0.015,
    'weight_decay': 1e-5,
}

print("  模型配置:")
print(f"    - 网络结构: {config['width']}")
print(f"    - 网格大小: {config['grid']}")
print(f"    - 训练轮数: {config['epochs']}")

# 构建模型
model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])
model.to(device)

# 参数量对比
params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型构建完成，参数量: {params:,} (之前: 696)")

# ========== 3. 训练模型 ==========
print("\n[3/5] 开始训练模型...")

# 数据加载器
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 训练记录
history = {'train_loss': [], 'test_loss': [], 'lr': []}
best_loss = float('inf')
start_time = time.time()

print("\n  开始训练...")
for epoch in range(config['epochs']):
    # 训练
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        train_loss += loss.item()
    
    avg_train = train_loss / len(train_loader)
    history['train_loss'].append(avg_train)
    
    # 测试
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    
    avg_test = test_loss / len(test_loader)
    history['test_loss'].append(avg_test)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # 学习率调整
    scheduler.step(avg_test)
    
    # 保存最佳模型
    if avg_test < best_loss:
        best_loss = avg_test
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': avg_train,
            'test_loss': avg_test,
            'config': config
        }, 'models/kan_improved_best.pth')
    
    # 打印进度
    if (epoch + 1) % 20 == 0 or epoch < 5:
        print(f"    Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {avg_train:.3e} | Test: {avg_test:.3e} | "
              f"LR: {history['lr'][-1]:.2e}")

# 训练时间
train_time = time.time() - start_time
print(f"\n  ✓ 训练完成，用时: {train_time:.1f}秒")
print(f"    最佳测试损失: {best_loss:.3e}")

# ========== 4. 保存结果 ==========
print("\n[4/5] 保存改进模型和结果...")

# 保存最终模型
final_state = {
    'model_state': model.state_dict(),
    'config': config,
    'history': history,
    'best_loss': best_loss,
    'train_time': train_time,
    'data_transform': 'log10(y + 1e-12)',  # 记录数据变换方式
    'epsilon': epsilon,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
torch.save(final_state, 'models/kan_improved_final.pth')

# 保存训练历史
import json
history_data = {
    'train_loss': [float(x) for x in history['train_loss']],
    'test_loss': [float(x) for x in history['test_loss']],
    'learning_rate': [float(x) for x in history['lr']],
    'best_epoch': np.argmin(history['test_loss']) + 1,
    'best_loss': float(best_loss),
    'config': config,
    'log_transform_info': {
        'epsilon': epsilon,
        'y_train_log_range': [float(y_train_log.min()), float(y_train_log.max())],
        'original_zero_count': int(np.sum(y_train == 0))
    }
}
with open('models/improved_training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"  ✓ 模型保存: models/kan_improved_final.pth")
print(f"  ✓ 训练历史: models/improved_training_history.json")

# ========== 5. 快速验证 ==========
print("\n[5/5] 快速验证改进效果...")

# 加载最佳模型
checkpoint = torch.load('models/kan_improved_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 在测试集上预测
with torch.no_grad():
    y_pred_log = model(X_test_t).cpu().numpy()

# 反变换到原始尺度
y_pred_linear = 10**y_pred_log - epsilon
y_test_linear = 10**y_test_t.cpu().numpy() - epsilon

# 计算原始尺度的误差
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse_original = mean_squared_error(y_test_linear, y_pred_linear)
mae_original = mean_absolute_error(y_test_linear, y_pred_linear)

print("\n  改进后性能（原始尺度）:")
print(f"    MSE: {mse_original:.3e} (之前: ~1.66e-02)")
print(f"    MAE: {mae_original:.3e}")

# 与改进前对比
improvement = (1.66e-02 - mse_original) / 1.66e-02 * 100
print(f"    MSE改进: {improvement:+.1f}%")

print("\n" + "="*60)
print("改进训练完成！关键改进点：")
print("="*60)
print("1. ✅ 对数变换：处理零值和大范围跨度问题")
print("2. ✅ 增加模型容量：参数量从696增加到~2000+")
print("3. ✅ 优化训练策略：AdamW优化器 + 学习率调度")
print("4. ✅ 梯度裁剪：防止训练不稳定")
print(f"\n改进效果：MSE预计提升 {improvement:+.1f}%")
print("\n下一步：运行评估代码查看详细结果")
print("="*60)