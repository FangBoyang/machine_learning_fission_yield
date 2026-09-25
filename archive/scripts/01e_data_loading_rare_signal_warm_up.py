"""
01e_data_loading_rare_signal_warm_up.py
KAN模型训练 - 数据加载模块
功能: 从data/GEF.csv加载所有数据文件，进行预处理
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
print("KAN模型训练 - 数据加载模块（GEF数据加载版）")
print("="*60)

# ========== 1. 设置路径 ==========
print("\n[步骤1/6] 设置文件路径...")

# 数据文件路径
DATA_DIR = "data/"
CSV_FILE = os.path.join(DATA_DIR, "GEF.csv")  # 修改为GEF.csv
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
    # 修改这里：GEF.csv没有列名，使用默认列名
    df = pd.read_csv(CSV_FILE, header=None)  # 修改：指定没有表头
    
    # 根据数据形状设置列名
    if df.shape[1] == 5:  # 如果数据有5列
        # 使用正确的列名
        df.columns = ['Z', 'A', 'E', 'Yield', 'Error']
    else:
        # 如果不是5列，则只取前5列
        print(f"  ⚠️  警告: 数据有{df.shape[1]}列，但预期5列，只取前5列")
        df = df.iloc[:, :5]
        df.columns = ['Z', 'A', 'E', 'Yield', 'Error']
    
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

# ========== 6. 准备GEF训练数据 ==========
print("\n[步骤6/6] 准备GEF训练数据（全部作为训练集）...")

# 分离特征和目标
X = df[['Z', 'A', 'E']].values
y = df['Yield'].values.reshape(-1, 1)

print(f"    ✓ 全部数据作为训练集: {X.shape[0]} 个样本")
print(f"    ✓ 特征维度: {X.shape[1]}")

# 检查是否有Error列
if 'Error' in df.columns:
    error = df['Error'].values.reshape(-1, 1)
    print(f"    ✓ 已加载误差列Error: {len(error)}个")
    has_error = True
else:
    error = None
    has_error = False
    print(f"    ⚠️ 数据中没有Error列")

# 转换为PyTorch张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    ✓ 使用设备: {device}")

# 全部数据作为训练集
X_train = X
y_train = y
error_train = error

# 对于GEF数据，验证集和测试集设为空
X_val = np.array([]).reshape(0, X.shape[1])
y_val = np.array([]).reshape(0, 1)
X_test = np.array([]).reshape(0, X.shape[1])
y_test = np.array([]).reshape(0, 1)

if has_error:
    error_val = np.array([]).reshape(0, 1)
    error_test = np.array([]).reshape(0, 1)
else:
    error_val = None
    error_test = None

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device) if len(X_val) > 0 else torch.tensor([], dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device) if len(y_val) > 0 else torch.tensor([], dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device) if len(X_test) > 0 else torch.tensor([], dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device) if len(y_test) > 0 else torch.tensor([], dtype=torch.float32).to(device)

# 如果有Error列，也转换为张量
if has_error:
    error_train_tensor = torch.tensor(error_train, dtype=torch.float32).to(device)
    error_val_tensor = torch.tensor(error_val, dtype=torch.float32).to(device) if error_val is not None else None
    error_test_tensor = torch.tensor(error_test, dtype=torch.float32).to(device) if error_test is not None else None
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
        'num_features': X_train.shape[1],
        'device': str(device),
        'data_source': 'GEF_theoretical'
    }
}

# 如果有Error列，也添加到数据字典
if has_error:
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
output_file = "preprocessed_gef_data.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"    ✓ 数据已保存到: {output_file}")

# ========== 8. 生成数据统计报告 ==========
print("\n" + "="*60)
print("GEF数据加载完成！摘要信息:")
print("="*60)
print(f"1. 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"2. 数据文件: GEF.csv (理论计算数据)")
print(f"3. 特征列: {', '.join(features)}")
print(f"4. 目标列: Yield")
if has_error:
    print(f"5. 误差列: Error (共{df.shape[0]}个值)")
print(f"6. 数据使用策略:")
print(f"   - 全部数据作为训练集: {X_train.shape[0]} 样本")
print(f"   - 验证集: 0 样本 (用于warm-up阶段)")
print(f"   - 测试集: 0 样本 (用于warm-up阶段)")
print(f"7. 计算设备: {device}")
print(f"8. 已加载scaler文件: {len(scalers)}个")
print(f"9. 输出文件: {output_file}")

# 显示训练集特征统计
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

# 数据分布检查
print("\n数据分布检查:")
train_zero_count = np.sum(y_train == 0)
if train_zero_count > 0:
    print(f"  ⚠️  警告: 训练集中有 {train_zero_count} 个Yield值为0 ({train_zero_count/len(y_train)*100:.1f}%)")

# 检查数据尺度差异
print(f"\n特征尺度差异 (训练集):")
for i, feat in enumerate(features):
    feat_std = X_train[:, i].std()
    print(f"  {feat}: 标准差 = {feat_std:.6f}")

# 保存分割信息到文本文件
split_info = f"""GEF数据加载信息
========================================
生成时间: {pd.Timestamp.now()}
原始数据文件: {CSV_FILE}
原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列

数据使用策略:
- 全部数据作为训练集: {X_train.shape[0]} 样本
- 验证集: 0 样本
- 测试集: 0 样本

数据统计:
训练集Yield: 范围[{y_train.min():.2e}, {y_train.max():.2e}], 均值{y_train.mean():.2e}

设备: {device}
输出文件: {output_file}
"""

with open("gef_data_loading_info.txt", "w") as f:
    f.write(split_info)

print(f"\n  ✓ 数据信息已保存到: gef_data_loading_info.txt")
print("\n" + "="*60)
print("GEF数据加载完成！")
print("="*60)