"""
KAN模型训练 - 模型训练模块
功能: 构建、训练和保存KAN模型
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型训练 - 模型训练模块")
print("="*60)

# ========== 1. 加载预处理数据 ==========
print("\n[步骤1/5] 加载预处理数据...")

try:
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # 提取数据
    X_train_tensor = data['X_train_tensor']
    y_train_tensor = data['y_train_tensor']
    X_test_tensor = data['X_test_tensor']
    y_test_tensor = data['y_test_tensor']
    device = data['device']
    
    print(f"  ✓ 成功加载预处理数据")
    print(f"    训练集: {X_train_tensor.shape[0]} 样本")
    print(f"    测试集: {X_test_tensor.shape[0]} 样本")
    print(f"    特征维度: {X_train_tensor.shape[1]}")
    print(f"    计算设备: {device}")
    
except Exception as e:
    print(f"  ✗ 加载预处理数据失败: {e}")
    print("  请先运行 01_data_loading.py")
    exit(1)

# 检查零值问题
print(f"\n  Yield零值统计:")
y_train_np = data['y_train']
zero_count = np.sum(y_train_np == 0)
total_count = len(y_train_np)
zero_ratio = zero_count / total_count * 100
print(f"    训练集中有 {zero_count}/{total_count} 个零值 ({zero_ratio:.1f}%)")

if zero_count > 0:
    print("  ⚠️  注意: 零值可能影响模型训练")
    print("    可选解决方案:")
    print("    1. 添加微小噪声: y = y + 1e-10")
    print("    2. 使用对数变换: y = log(y + epsilon)")
    print("    3. 使用加权损失函数")
    print("    当前: 保持原样继续训练")

# ========== 2. 导入KAN库 ==========
print("\n[步骤2/5] 导入KAN库...")

try:
    from kan import KAN
    print("  ✓ 成功导入KAN库")
except ImportError as e:
    print(f"  ✗ 导入KAN库失败: {e}")
    print("  请确保已安装KAN: pip install git+https://github.com/KindXiaoming/pykan.git")
    exit(1)

# ========== 3. 构建KAN模型 ==========
print("\n[步骤3/5] 构建KAN模型...")

# 模型配置参数
config = {
    'input_dim': 3,  # 输入特征维度: Z, A, E
    'hidden_dims': [5, 3],  # 隐藏层维度
    'output_dim': 1,  # 输出维度: Yield
    'grid_size': 5,  # 网格大小
    'spline_order': 3,  # 样条阶数
    'seed': 42,  # 随机种子
    'num_epochs': 100,  # 训练轮数
    'batch_size': 32,  # 批大小
    'learning_rate': 0.01,  # 学习率
    'weight_decay': 1e-4,  # 权重衰减
}

print(f"  模型配置:")
print(f"    - 输入维度: {config['input_dim']} (Z, A, E)")
print(f"    - 隐藏层结构: {config['hidden_dims']}")
print(f"    - 输出维度: {config['output_dim']}")
print(f"    - 网格大小: {config['grid_size']}")
print(f"    - 样条阶数: {config['spline_order']}")
print(f"    - 训练轮数: {config['num_epochs']}")
print(f"    - 批大小: {config['batch_size']}")
print(f"    - 学习率: {config['learning_rate']}")

# 构建模型宽度列表
width = [config['input_dim']] + config['hidden_dims'] + [config['output_dim']]
print(f"  KAN网络宽度: {width}")

# 创建KAN模型
torch.manual_seed(config['seed'])
model = KAN(
    width=width,
    grid=config['grid_size'],
    k=config['spline_order'],
    seed=config['seed']
)
model.to(device)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  ✓ 模型构建完成")
print(f"    总参数量: {total_params:,}")
print(f"    可训练参数量: {trainable_params:,}")

# ========== 4. 训练模型 ==========
print("\n[步骤4/5] 开始训练模型...")

# 创建数据加载器
from torch.utils.data import DataLoader, TensorDataset

# 创建数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

# 训练记录
train_losses = []
test_losses = []
learning_rates = []
best_loss = float('inf')
best_epoch = 0

# 训练循环
start_time = time.time()
print(f"\n  开始训练 (共{config['num_epochs']}轮)...")

for epoch in range(config['num_epochs']):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        num_batches += 1
    
    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # 学习率调度
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    scheduler.step()
    
    # 评估阶段（每10轮或最后5轮）
    if (epoch + 1) % 10 == 0 or epoch >= config['num_epochs'] - 5:
        model.eval()
        epoch_test_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_test_loss += loss.item()
                num_test_batches += 1
        
        avg_test_loss = epoch_test_loss / num_test_batches
        test_losses.append(avg_test_loss)
        
        # 保存最佳模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch + 1
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'config': config
            }, 'models/kan_best_model.pth')
        
        # 打印训练进度
        print(f"    轮次 [{epoch+1:3d}/{config['num_epochs']}] | "
              f"训练损失: {avg_train_loss:.3e} | "
              f"测试损失: {avg_test_loss:.3e} | "
              f"学习率: {current_lr:.2e}")
    
    # 每20轮保存一次检查点
    if (epoch + 1) % 20 == 0:
        checkpoint_path = f"models/kan_checkpoint_epoch{epoch+1:03d}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss if 'avg_test_loss' in locals() else None,
            'config': config
        }, checkpoint_path)

# 计算训练时间
end_time = time.time()
training_time = end_time - start_time
print(f"\n  ✓ 训练完成!")
print(f"    总训练时间: {training_time:.1f} 秒")
print(f"    最佳轮次: {best_epoch}, 最佳测试损失: {best_loss:.3e}")

# ========== 5. 保存最终模型和训练记录 ==========
print("\n[步骤5/5] 保存最终结果...")

# 保存最终模型
final_model_path = "models/kan_final_model.pth"
torch.save({
    'epoch': config['num_epochs'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'learning_rates': learning_rates,
    'best_epoch': best_epoch,
    'best_loss': best_loss,
    'config': config,
    'training_time': training_time
}, final_model_path)
print(f"  ✓ 最终模型已保存: {final_model_path}")

# 保存训练记录
training_history = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'learning_rates': learning_rates,
    'best_epoch': best_epoch,
    'best_loss': best_loss,
    'training_time': training_time,
    'config': config,
    'zero_yield_count': zero_count,
    'zero_yield_ratio': zero_ratio,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(training_history, f)
print(f"  ✓ 训练记录已保存: models/training_history.pkl")

# 保存训练配置
with open('models/training_config.txt', 'w') as f:
    f.write("KAN模型训练配置\n")
    f.write("=" * 40 + "\n")
    f.write(f"训练时间: {training_history['timestamp']}\n")
    f.write(f"训练时长: {training_time:.1f}秒\n")
    f.write(f"最佳轮次: {best_epoch}\n")
    f.write(f"最佳测试损失: {best_loss:.3e}\n")
    f.write(f"零值比例: {zero_ratio:.1f}% ({zero_count}/{total_count})\n\n")
    
    f.write("模型配置:\n")
    for key, value in config.items():
        f.write(f"  {key}: {value}\n")
    
    f.write(f"\n数据信息:\n")
    f.write(f"  训练集大小: {X_train_tensor.shape[0]}\n")
    f.write(f"  测试集大小: {X_test_tensor.shape[0]}\n")
    f.write(f"  特征维度: {X_train_tensor.shape[1]}\n")
    f.write(f"  计算设备: {device}\n")
    
    f.write(f"\n损失记录:\n")
    f.write(f"  最终训练损失: {train_losses[-1]:.3e}\n")
    f.write(f"  最终测试损失: {test_losses[-1] if test_losses else 'N/A'}\n")
    f.write(f"  总训练轮次: {len(train_losses)}\n")

print(f"  ✓ 训练配置已保存: models/training_config.txt")

# ========== 6. 生成训练摘要 ==========
print("\n" + "="*60)
print("训练完成！摘要信息:")
print("="*60)
print(f"1. 模型架构: KAN{width}")
print(f"2. 训练轮次: {config['num_epochs']}")
print(f"3. 最佳轮次: {best_epoch} (损失: {best_loss:.3e})")
print(f"4. 最终训练损失: {train_losses[-1]:.3e}")
if test_losses:
    print(f"5. 最终测试损失: {test_losses[-1]:.3e}")
print(f"6. 训练时间: {training_time:.1f}秒")
print(f"7. 参数量: {total_params:,}")
print(f"8. 零值比例: {zero_ratio:.1f}%")
print(f"\n保存的文件:")
print(f"  - 最佳模型: models/kan_best_model.pth")
print(f"  - 最终模型: models/kan_final_model.pth")
print(f"  - 训练记录: models/training_history.pkl")
print(f"  - 训练配置: models/training_config.txt")

# 损失曲线预览
if len(train_losses) > 1:
    print(f"\n损失变化:")
    print(f"  初始损失: {train_losses[0]:.3e}")
    print(f"  最终损失: {train_losses[-1]:.3e}")
    improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"  改进幅度: {improvement:.1f}%")

print("\n" + "="*60)
print("下一步操作:")
print("1. 运行 'python src/03_evaluate_kan.py' 评估模型")
print("2. 检查 models/ 文件夹中的模型文件")
print("3. 检查 results/ 文件夹中的评估结果")
print("="*60)