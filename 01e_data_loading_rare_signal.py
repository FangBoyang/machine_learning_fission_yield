"""
01f_data_loading_rare_signal.py
KAN模型训练 - 数据加载模块（稀有信号保护版）
功能: 从data/文件夹加载所有数据文件，进行预处理和验证
     仅在前3096个数据上划分验证集和测试集，剩余数据全部加入训练集
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
import torch
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型训练 - 数据加载模块（稀有信号保护版）")
print("="*60)

# ========== 1. 设置路径 ==========
print("\n[步骤1/6] 设置文件路径...")

# 数据文件路径
DATA_DIR = "data/"
CSV_FILE = os.path.join(DATA_DIR, "235UALL.csv")
SCALER_FILES = {
    'Z': os.path.join(DATA_DIR, "standard_scalerZ.pkl"),
    'A': os.path.join(DATA_DIR, "standard_scalerA.pkl"),
    'E': os.path.join(DATA_DIR, "standard_scalerE.pkl"),
    'Yield': os.path.join(DATA_DIR, "yield_scaler.pkl")
}

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ========== 2. 文件存在性检查 ==========
print("\n[步骤2/6] 检查必需文件...")

missing_files = []

# 检查CSV文件
if os.path.exists(CSV_FILE):
    print(f"  ✓ 找到CSV文件: {CSV_FILE}")
else:
    print(f"  ✗ 缺失CSV文件: {CSV_FILE}")
    missing_files.append(CSV_FILE)

# 检查scaler文件
for name, filepath in SCALER_FILES.items():
    if os.path.exists(filepath):
        print(f"  ✓ 找到scaler文件: {filepath}")
    else:
        print(f"  ✗ 缺失scaler文件: {filepath}")
        missing_files.append(filepath)

if missing_files:
    print(f"\n错误: 以下文件缺失:")
    for f in missing_files:
        print(f"  - {f}")
    print("\n请确保所有文件在data/文件夹中")
    exit(1)

# ========== 3. 加载CSV数据 ==========
print("\n[步骤3/6] 加载CSV数据...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"  ✓ 成功加载CSV文件")
    print(f"    数据形状: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"    列名: {list(df.columns)}")
    
    # 显示数据示例
    print("\n    前5行数据:")
    print(df.head().to_string())
    
except Exception as e:
    print(f"  ✗ 加载CSV文件失败: {e}")
    exit(1)

# 数据质量检查
print("\n    数据质量检查:")
print(f"    - 总缺失值: {df.isnull().sum().sum()}")
if df.isnull().sum().sum() > 0:
    print(f"    - 各列缺失值:")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"      {col}: {missing_count} 个缺失值")

# ========== 4. 使用joblib加载Scaler文件 ==========
print("\n[步骤4/6] 使用joblib加载归一化参数文件...")
scalers = {}

try:
    for name, filepath in SCALER_FILES.items():
        # 使用joblib加载scaler文件
        scalers[name] = joblib.load(filepath)
        print(f"  ✓ 已加载: {name} scaler")
    
    # 显示scaler信息
    print("\n    Scaler信息:")
    for name, scaler in scalers.items():
        if hasattr(scaler, 'mean_'):
            if hasattr(scaler.mean_, '__len__'):
                mean_val = scaler.mean_[0] if len(scaler.mean_) > 0 else scaler.mean_
                scale_val = scaler.scale_[0] if len(scaler.scale_) > 0 else scaler.scale_
            else:
                mean_val = scaler.mean_
                scale_val = scaler.scale_
            print(f"      {name}: mean={mean_val:.6f}, scale={scale_val:.6f}")
        else:
            print(f"      {name}: 无mean_属性")
            
except Exception as e:
    print(f"  ✗ 加载scaler文件失败: {e}")
    print(f"    错误详情: {type(e).__name__}: {e}")
    exit(1)

# ========== 5. 验证数据归一化状态 ==========
print("\n[步骤5/6] 验证数据归一化状态...")

# 检查特征范围
features = ['Z', 'A', 'E']
print("    特征值范围检查:")
for feat in features:
    col_min = df[feat].min()
    col_max = df[feat].max()
    col_mean = df[feat].mean()
    col_std = df[feat].std()
    
    # 判断是否归一化
    is_normalized = (-1.5 <= col_min <= 1.5 and -1.5 <= col_max <= 1.5)
    
    status = "✓ 已归一化" if is_normalized else "? 可能未完全归一化"
    print(f"      {feat}:")
    print(f"        范围: [{col_min:.6f}, {col_max:.6f}]")
    print(f"        均值: {col_mean:.6f}, 标准差: {col_std:.6f}")
    print(f"        状态: {status}")

# 检查目标变量
yield_min, yield_max = df['Yield'].min(), df['Yield'].max()
yield_mean, yield_std = df['Yield'].mean(), df['Yield'].std()
print(f"\n    目标变量Yield:")
print(f"        范围: [{yield_min:.2e}, {yield_max:.2e}]")
print(f"        均值: {yield_mean:.2e}, 标准差: {yield_std:.2e}")

# 检查Error列
if 'Error' in df.columns:
    error_min, error_max = df['Error'].min(), df['Error'].max()
    print(f"\n    误差列Error:")
    print(f"        范围: [{error_min:.2e}, {error_max:.2e}]")

# ========== 6. 数据分割和转换（稀有信号保护） ==========
print("\n[步骤6/6] 数据分割和转换（稀有信号保护）...")

# 将数据分为两部分：前3096行（基础数据）和剩余行（稀有信号数据）
split_index = 3096
print(f"    数据分割点: 前{split_index}行作为基础数据，剩余{len(df)-split_index}行作为稀有信号数据")

# 分离特征和目标
X = df[['Z', 'A', 'E']].values
y = df['Yield'].values.reshape(-1, 1)

# 获取基础数据（前3096行）
X_base = X[:split_index]
y_base = y[:split_index]

# 获取稀有信号数据（剩余行）
X_rare = X[split_index:]
y_rare = y[split_index:]

print(f"    ✓ 基础数据: {X_base.shape[0]} 个样本")
print(f"    ✓ 稀有信号数据: {X_rare.shape[0]} 个样本")

# 检查是否有Error列，如果有也进行同样的分割
if 'Error' in df.columns:
    error = df['Error'].values.reshape(-1, 1)
    error_base = error[:split_index]
    error_rare = error[split_index:]
    print(f"    ✓ 已分离误差列Error: 基础数据{len(error_base)}个，稀有数据{len(error_rare)}个")

# 在基础数据中划分验证集和测试集
test_size = 600
val_size = 600

# 确保基础数据足够划分
if test_size + val_size >= len(X_base):
    print(f"  ✗ 错误: 基础数据样本数({len(X_base)})不足，无法分配{test_size}测试+{val_size}验证样本")
    print(f"    请减少test_size或val_size，或增加split_index")
    exit(1)

# 第一次分割：从基础数据中分出测试集
X_base_temp, X_test, y_base_temp, y_test = train_test_split(
    X_base, y_base, test_size=test_size, random_state=42, shuffle=True
)

# 第二次分割：从剩余基础数据中分出验证集
remaining_samples = len(X_base_temp)
val_ratio = val_size / remaining_samples

X_base_train, X_val, y_base_train, y_val = train_test_split(
    X_base_temp, y_base_temp, test_size=val_ratio, random_state=42, shuffle=True
)

# 合并基础训练集和稀有信号数据作为最终训练集
X_train = np.vstack([X_base_train, X_rare])
y_train = np.vstack([y_base_train, y_rare])

print(f"    ✓ 基础训练集: {X_base_train.shape[0]} 个样本")
print(f"    ✓ 验证集: {X_val.shape[0]} 个样本")
print(f"    ✓ 测试集: {X_test.shape[0]} 个样本")
print(f"    ✓ 稀有信号数据: {X_rare.shape[0]} 个样本")
print(f"    ✓ 最终训练集: {X_train.shape[0]} 个样本 (基础训练集 + 稀有信号数据)")
print(f"    ✓ 特征维度: {X_train.shape[1]}")
print(f"    ✓ 分割比例: 训练({X_train.shape[0]/len(X)*100:.1f}%), "
      f"验证({X_val.shape[0]/len(X)*100:.1f}%), "
      f"测试({X_test.shape[0]/len(X)*100:.1f}%)")

# 如果有Error列，也需要进行相同的划分
if 'Error' in df.columns:
    # 基础数据Error的划分（与y_base保持相同的索引）
    from sklearn.model_selection import train_test_split
    error_base_temp, error_test = train_test_split(
        error_base, test_size=test_size, random_state=42, shuffle=True
    )
    error_base_train, error_val = train_test_split(
        error_base_temp, test_size=val_ratio, random_state=42, shuffle=True
    )
    
    # 合并训练集的Error
    error_train = np.vstack([error_base_train, error_rare])
    
    print(f"    ✓ 已相应划分误差列Error:")
    print(f"       训练集Error: {len(error_train)} 个")
    print(f"       验证集Error: {len(error_val)} 个")
    print(f"       测试集Error: {len(error_test)} 个")

# 转换为PyTorch张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    ✓ 使用设备: {device}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 如果有Error列，也转换为张量
if 'Error' in df.columns:
    error_train_tensor = torch.tensor(error_train, dtype=torch.float32).to(device)
    error_val_tensor = torch.tensor(error_val, dtype=torch.float32).to(device)
    error_test_tensor = torch.tensor(error_test, dtype=torch.float32).to(device)
else:
    error_train_tensor = None
    error_val_tensor = None
    error_test_tensor = None

# ========== 7. 保存预处理结果 ==========
print("\n[保存结果] 保存预处理数据...")

# 创建数据字典
data_dict = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'X_train_tensor': X_train_tensor,
    'y_train_tensor': y_train_tensor,
    'X_val_tensor': X_val_tensor,
    'y_val_tensor': y_val_tensor,
    'X_test_tensor': X_test_tensor,
    'y_test_tensor': y_test_tensor,
    'device': device,
    'scalers': scalers,
    'feature_names': ['Z', 'A', 'E'],
    'target_name': 'Yield',
    'original_shape': df.shape,
    'data_info': {
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'base_train_size': X_base_train.shape[0],
        'rare_signal_size': X_rare.shape[0],
        'split_index': split_index,
        'num_features': X_train.shape[1],
        'device': str(device)
    }
}

# 如果有Error列，也添加到数据字典
if 'Error' in df.columns:
    data_dict['error_train'] = error_train
    data_dict['error_val'] = error_val
    data_dict['error_test'] = error_test
    data_dict['error_train_tensor'] = error_train_tensor
    data_dict['error_val_tensor'] = error_val_tensor
    data_dict['error_test_tensor'] = error_test_tensor
    data_dict['has_error_column'] = True
else:
    data_dict['has_error_column'] = False

# 保存到文件
output_file = "preprocessed_data_rare_signal.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"    ✓ 数据已保存到: {output_file}")

# ========== 8. 生成数据统计报告 ==========
print("\n" + "="*60)
print("数据加载完成！摘要信息（稀有信号保护版）:")
print("="*60)
print(f"1. 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"2. 特征列: {', '.join(features)}")
print(f"3. 目标列: Yield")
if 'Error' in df.columns:
    print(f"4. 误差列: Error (共{df.shape[0]}个值)")
print(f"5. 数据分割策略:")
print(f"   - 基础数据（前{split_index}行）: 划分训练/验证/测试")
print(f"   - 稀有信号数据（{len(X_rare)}行）: 全部加入训练集")
print(f"6. 最终划分结果:")
print(f"   - 训练集: {X_train.shape[0]} 样本 (基础训练集{X_base_train.shape[0]} + 稀有信号{X_rare.shape[0]})")
print(f"   - 验证集: {X_val.shape[0]} 样本")
print(f"   - 测试集: {X_test.shape[0]} 样本")
print(f"7. 计算设备: {device}")
print(f"8. 已加载scaler文件: {len(scalers)}个")
print(f"9. 输出文件: {output_file}")

# 显示各集合特征统计
print("\n特征统计 (训练集):")
for i, feat in enumerate(features):
    col_data = X_train[:, i]
    print(f"  {feat}:")
    print(f"    范围: [{col_data.min():.6f}, {col_data.max():.6f}]")
    print(f"    均值: {col_data.mean():.6f}, 标准差: {col_data.std():.6f}")

print(f"\n目标变量统计:")
print(f"  训练集 Yield:")
print(f"    范围: [{y_train.min():.2e}, {y_train.max():.2e}]")
print(f"    均值: {y_train.mean():.2e}, 标准差: {y_train.std():.2e}")

print(f"\n  验证集 Yield:")
print(f"    范围: [{y_val.min():.2e}, {y_val.max():.2e}]")
print(f"    均值: {y_val.mean():.2e}, 标准差: {y_val.std():.2e}")

print(f"\n  测试集 Yield:")
print(f"    范围: [{y_test.min():.2e}, {y_test.max():.2e}]")
print(f"    均值: {y_test.mean():.2e}, 标准差: {y_test.std():.2e}")

# 数据分布检查
print("\n数据分布检查:")
train_zero_count = np.sum(y_train == 0)
val_zero_count = np.sum(y_val == 0)
test_zero_count = np.sum(y_test == 0)

if train_zero_count > 0:
    print(f"  ⚠️  警告: 训练集中有 {train_zero_count} 个Yield值为0 ({train_zero_count/len(y_train)*100:.1f}%)")

if val_zero_count > 0:
    print(f"  ⚠️  警告: 验证集中有 {val_zero_count} 个Yield值为0 ({val_zero_count/len(y_val)*100:.1f}%)")

if test_zero_count > 0:
    print(f"  ⚠️  警告: 测试集中有 {test_zero_count} 个Yield值为0 ({test_zero_count/len(y_test)*100:.1f}%)")

# 检查数据尺度差异
print(f"\n特征尺度差异 (训练集):")
for i, feat in enumerate(features):
    feat_std = X_train[:, i].std()
    print(f"  {feat}: 标准差 = {feat_std:.6f}")

# 稀有信号数据统计
print(f"\n稀有信号数据统计:")
print(f"  - 样本数量: {X_rare.shape[0]}")
print(f"  - Yield范围: [{y_rare.min():.2e}, {y_rare.max():.2e}]")
print(f"  - Yield均值: {y_rare.mean():.2e}")

# 保存分割信息到文本文件
split_info = f"""数据分割信息（稀有信号保护版）
========================================
生成时间: {pd.Timestamp.now()}
原始数据文件: {CSV_FILE}
原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列

分割策略:
1. 基础数据: 前{split_index}行
2. 稀有信号数据: 剩余{len(X_rare)}行（全部加入训练集）

最终划分结果:
- 训练集: {X_train.shape[0]} 样本
  - 基础训练集: {X_base_train.shape[0]} 样本
  - 稀有信号数据: {X_rare.shape[0]} 样本
- 验证集: {X_val.shape[0]} 样本
- 测试集: {X_test.shape[0]} 样本

数据统计:
训练集Yield: 范围[{y_train.min():.2e}, {y_train.max():.2e}], 均值{y_train.mean():.2e}
验证集Yield: 范围[{y_val.min():.2e}, {y_val.max():.2e}], 均值{y_val.mean():.2e}
测试集Yield: 范围[{y_test.min():.2e}, {y_test.max():.2e}], 均值{y_test.mean():.2e}

设备: {device}
输出文件: {output_file}
"""

with open("data_split_info_rare_signal.txt", "w") as f:
    f.write(split_info)

print(f"\n  ✓ 分割信息已保存到: data_split_info_rare_signal.txt")
print("\n" + "="*60)
print("稀有信号保护数据加载完成！")
print("="*60)