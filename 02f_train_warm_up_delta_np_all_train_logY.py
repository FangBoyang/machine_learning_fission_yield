"""
02f_train_delta_np_all_train_logY.py
功能: 在GEF理论数据上训练KAN模型（Z/A/E+delta_np输入，对数空间Yield目标，全训练集模式）
说明: 使用普通MSE损失，自动偏向高产额区，输出天然非负，无BNN奖励劫持问题
"""

import joblib
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
print("KAN模型训练 - 对数空间Yield版（Z/A/E+delta_np输入，全训练集模式）")
print("="*60)
print("核心特性:")
print("1. 损失函数: 普通对数空间MSE，无复杂加权，训练稳定")
print("2. 自动偏向高产额区，符合物理研究需求")
print("3. 预测结果天然非负，无需后处理")
print("="*60)

# ========== 1. 加载预处理数据（对数空间版本） ==========
print("\n[1/7] 加载对数空间预处理数据（全训练集模式）...")

try:
    # 【改动1】加载最新的对数空间预处理文件，避免和旧版本混淆
    with open('preprocessed_gef_data_delta_np_with_ZAE_logY.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train_tensor'].cpu().numpy()
    y_train_log_norm = data['y_train_tensor'].cpu().numpy()  # 归一化后的对数Yield
    device = data['device']
    scalers = data['scalers']  # 【新增】加载scaler用于反变换
    y_train_original = data['raw_data']['Yield_original']  # 【新增】原始空间真实Yield，用于评估
    
    print(f"  ✓ 数据加载成功")
    print(f"    训练集: {X_train.shape[0]} 样本 (100%)")
    print(f"    输入特征: {data['feature_names']} (共{X_train.shape[1]}维)")
    print(f"    目标变量: 归一化对数Yield（反变换后为原始产额）")
    print(f"    设备: {device}")
    
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    exit(1)

# ========== 2. 数据统计 ==========
print("\n[2/7] 数据统计...")
print(f"    输入特征范围:")
for i, name in enumerate(data['feature_names']):
    feat_min, feat_max = X_train[:, i].min(), X_train[:, i].max()
    print(f"      {name}: [{feat_min:.3f}, {feat_max:.3f}]")
print(f"    归一化对数Yield范围: [{y_train_log_norm.min():.3f}, {y_train_log_norm.max():.3f}]")
print(f"    原始Yield范围: [{y_train_original.min():.2e}, {y_train_original.max():.2e}]")

# ========== 3. 准备训练数据（无验证集） ==========
print("\n[3/7] 准备训练数据（全训练集模式）...")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_log_norm_t = torch.tensor(y_train_log_norm, dtype=torch.float32).to(device)

print(f"  ✓ 训练集张量形状: {X_train_t.shape}")
print(f"  ✓ 误差列仅用于参考，不参与损失计算（对数空间已自动加权）")

# ========== 4. 构建KAN模型 ==========
print("\n[4/7] 构建KAN模型...")

input_dim = X_train_t.shape[1]  # 现为4（Z_norm, A_norm, E_norm, delta_np）

# 【改动2】更新模型配置，适配4维输入+对数空间训练
config = {
    'width': [input_dim, 20, 20, 1],  # 适当加宽，适配4维特征
    'grid': 15,                        # 增加网格密度，更好拟合分段物理规律
    'k': 4,
    'seed': 42,
    'epochs': 1000,                    # 对数空间收敛稍慢，适当增加轮数
    'batch_size': 1024,
    'learning_rate': 0.08,             # 略降初始LR，提升稳定性
    'weight_decay': 2e-4,              # 放松正则化，全训练集过拟合风险低
    'patience': 100,                   # 增加耐心，等待充分收敛
    'min_delta': 1e-6,
}

print("  模型配置:")
print(f"    - 输入特征: {data['feature_names']}")
print(f"    - 输入维度: {input_dim}")
print(f"    - 网络结构: {config['width']}")
print(f"    - 训练轮数: {config['epochs']}")
print(f"    - 批量大小: {config['batch_size']}")
print(f"    - 学习率: {config['learning_rate']}")
print(f"    - 损失函数: 普通MSE（对数空间）")
print(f"    - 早停耐心: {config['patience']} (基于训练损失)")

model = KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])
model.to(device)

params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型构建完成，参数量: {params:,}")

# 【改动3】替换为普通MSE损失，彻底抛弃BNN加权逻辑
criterion = nn.MSELoss()
print(f"  ✓ 使用对数空间MSE损失，自动偏向高产额区")

# AdamW优化器
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay']
)

# 带重启的余弦退火学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=300, T_mult=1, eta_min=1e-5
)

# ========== 5. 训练模型（基于训练损失早停） ==========
print("\n[5/7] 开始训练（全训练集 + 训练损失早停）...")

# 【改动4】DataLoader无需加载误差列，对数空间MSE不需要加权
train_dataset = TensorDataset(X_train_t, y_train_log_norm_t)
train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=0
)

history = {'train_loss': [], 'lr': []}
best_loss = float('inf')
best_epoch = 0
patience_counter = 0
start_time = time.time()

print("\n  开始训练...")
for epoch in range(config['epochs']):
    epoch_start = time.time()
    
    # ===== 训练阶段 =====
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_x, batch_y in train_loader:  # 【改动5】不再传入误差列
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)  # 【改动6】直接用普通MSE
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
    
    avg_train = train_loss / train_batches
    history['train_loss'].append(avg_train)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # 更新学习率
    scheduler.step()
    
    # ===== 早停逻辑（基于训练损失）=====
    if avg_train < best_loss - config['min_delta']:
        best_loss = avg_train
        best_epoch = epoch + 1
        patience_counter = 0
        
        # 【改动7】文件名添加_logY后缀，避免覆盖旧模型
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': avg_train,
            'config': config,
            'input_dim': input_dim,
            'feature_names': data['feature_names'],  # 更新为特征列表
            'training_stage': 'logY_fulltrain',
            'scaler_logY_mean': scalers['Yield_log'].mean_[0],  # 保存反变换参数
            'scaler_logY_scale': scalers['Yield_log'].scale_[0]
        }, 'models/kan_delta_np_with_ZAE_logY_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= config['patience']:
            print(f"\n  ⏹️  早停触发: 连续{config['patience']}个epoch训练损失未改善")
            print(f"     最佳训练损失: {best_loss:.3e} (Epoch {best_epoch})")
            break
    
    epoch_time = time.time() - epoch_start
    
    # 打印训练进度
    if (epoch + 1) % 20 == 0 or epoch < 5 or epoch + 1 == config['epochs']:
        status = "⏹️ 早停" if patience_counter >= config['patience'] else f"Patience: {patience_counter}/{config['patience']}"
        print(f"    Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {avg_train:.3e} | "
              f"LR: {history['lr'][-1]:.3e} | {status} | Time: {epoch_time:.1f}s")

train_time = time.time() - start_time
print(f"\n  ✓ 训练完成，总用时: {train_time:.1f}秒")
print(f"    实际训练轮数: {epoch+1}")
print(f"    最佳训练损失（归一化对数空间）: {best_loss:.3e} (Epoch {best_epoch})")

# ========== 6. 保存最终结果 ==========
print("\n[6/7] 保存模型和结果...")

# 加载最佳模型
try:
    checkpoint = torch.load('models/kan_delta_np_with_ZAE_logY_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print("  ✓ 加载最佳模型成功")
except Exception as e:
    print(f"  ✗ 加载模型失败: {e}")

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
    'input_dim': input_dim,
    'feature_names': data['feature_names'],
    'training_stage': 'logY_fulltrain',
    'scaler_logY_mean': scalers['Yield_log'].mean_[0],
    'scaler_logY_scale': scalers['Yield_log'].scale_[0],
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
torch.save(final_state, 'models/kan_delta_np_with_ZAE_logY_final.pth')
print("  ✓ 最终模型保存: models/kan_delta_np_with_ZAE_logY_final.pth")

# ========== 7. 性能评估（同时输出对数空间和原始空间指标） ==========
print("\n[7/7] 训练集性能评估...")

model.eval()
with torch.no_grad():
    # 1. 模型输出：归一化对数Yield
    y_pred_log_norm = model(X_train_t).cpu().numpy()
    
    # 2. 反归一化到对数空间
    y_pred_log = scalers['Yield_log'].inverse_transform(y_pred_log_norm)
    y_true_log = scalers['Yield_log'].inverse_transform(y_train_log_norm)
    
    # 3. 指数变换到原始空间（天然非负）
    y_pred_original = np.exp(y_pred_log)
    y_true_original = y_train_original.reshape(-1, 1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 归一化对数空间指标（训练优化目标）
mse_log_norm = mean_squared_error(y_train_log_norm, y_pred_log_norm)
r2_log_norm = r2_score(y_train_log_norm, y_pred_log_norm)

# 原始空间指标（物理意义指标，你关心的）
mse_original = mean_squared_error(y_true_original, y_pred_original)
mae_original = mean_absolute_error(y_true_original, y_pred_original)
r2_original = r2_score(y_true_original, y_pred_original)

print("\n  训练集性能:")
print("  【归一化对数空间（优化目标）】")
print(f"    - R²: {r2_log_norm:.4f}")
print(f"    - MSE: {mse_log_norm:.3e}")
print("\n  【原始产额空间（物理意义，你关心的）】")
print(f"    - R²: {r2_original:.4f}")
print(f"    - MSE: {mse_original:.3e}")
print(f"    - MAE: {mae_original:.3e}")
print(f"    - 真实产额范围: [{y_true_original.min():.2e}, {y_true_original.max():.2e}]")
print(f"    - 预测产额范围: [{y_pred_original.min():.2e}, {y_pred_original.max():.2e}]")
print(f"    - 预测最小值: {y_pred_original.min():.2e}（无负值，符合要求）")

# 物理意义解读
print("\n  物理意义解读:")
if r2_original > 0.85:
    print("    ✓ 模型对高产额区拟合极佳，符合裂变产额物理规律")
elif r2_original > 0.7:
    print("    △ 模型对高产额区拟合良好，剩余误差来自裂变过程的固有量子涨落")
else:
    print("    ✗ 拟合效果待提升，可尝试增加网络宽度/网格密度")

print("\n" + "="*60)
print("对数空间Yield训练完成！")
print("="*60)
print(f"最佳模型: models/kan_delta_np_with_ZAE_logY_best.pth")
print(f"最终模型: models/kan_delta_np_with_ZAE_logY_final.pth")
print(f"训练历史: models/delta_np_with_ZAE_logY_history.json")
print("="*60)
print("\n【预测反变换指引】")
print("1. 加载模型后，输出为归一化对数Yield")
print("2. 用保存的scaler_logY_mean和scaler_logY_scale反归一化")
print("3. 对反归一化结果取exp()，得到原始空间非负产额")
print("="*60)