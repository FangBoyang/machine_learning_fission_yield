"""
01f_data_loading_warm_up_delta_np_all_train_logY.py
KAN模型训练 - 数据加载模块（归一化Z/A/E + delta_np版 - 对数空间MSE适配 - 全训练集模式）
功能: 从data/GEF.csv加载数据，使用归一化后的Z/A/E + delta_np作为输入特征，
     目标变量转为对数空间（防止负值、偏向高产额区），全部作为训练集
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import torch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAN模型训练 - 数据加载模块（对数空间Yield版 - 全训练集模式）")
print("="*60)
print("核心特性:")
print("1. 目标变量Yield转为对数空间，天然避免输出负值")
print("2. 对数变换自动提升高产额区拟合权重，匹配物理需求")
print("3. 保留原有Z/A/E/delta_np物理特征，无新增物理量")
print("="*60)

# ========== 1. 设置路径 ==========
print("\n[步骤1/6] 设置文件路径...")
DATA_DIR = "data/"
CSV_FILE = os.path.join(DATA_DIR, "GEF.csv")
YIELD_SCALER_FILE = os.path.join(DATA_DIR, "yield_scaler.pkl")  # 原始Yield scaler（仅参考）
DELTA_NP_SCALER_FILE = os.path.join(DATA_DIR, "delta_np_scaler.pkl")
Z_SCALER_FILE = os.path.join(DATA_DIR, "standard_scalerZ.pkl")
A_SCALER_FILE = os.path.join(DATA_DIR, "standard_scalerA.pkl")
LOG_YIELD_SCALER_FILE = os.path.join(DATA_DIR, "log_yield_scaler.pkl")  # 新增：对数Yield scaler

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ========== 2. 文件存在性检查 ==========
print("\n[步骤2/6] 检查必需文件...")
missing_files = []
if os.path.exists(CSV_FILE):
    print(f"  ✓ 找到CSV文件: {CSV_FILE}")
else:
    missing_files.append(CSV_FILE)
    print(f"  ✗ 缺失CSV文件: {CSV_FILE}")

# 加载原始Yield scaler（仅作参考，训练不使用）
if os.path.exists(YIELD_SCALER_FILE):
    print(f"  ✓ 找到原始Yield scaler（仅参考）: {YIELD_SCALER_FILE}")
else:
    missing_files.append(YIELD_SCALER_FILE)
    print(f"  ✗ 缺失原始Yield scaler: {YIELD_SCALER_FILE}")

# 物理计算必需的scaler
for name, path in [("Z scaler", Z_SCALER_FILE), ("A scaler", A_SCALER_FILE)]:
    if os.path.exists(path):
        print(f"  ✓ 找到{name}: {path}")
    else:
        missing_files.append(path)
        print(f"  ✗ 缺失{name}: {path}")

if missing_files:
    print(f"\n错误: 以下文件缺失:")
    for f in missing_files:
        print(f"  - {f}")
    print("\n请确保所有文件在data/文件夹中")
    exit(1)

# ========== 3. 加载CSV数据并计算delta_np特征 ==========
print("\n[步骤3/6] 加载CSV数据并计算delta_np特征...")
try:
    df = pd.read_csv(CSV_FILE, header=None)
    if df.shape[1] == 5:
        df.columns = ['Z', 'A', 'E', 'Yield', 'Error']
    else:
        print(f"  ⚠️  警告: 数据有{df.shape[1]}列，但预期5列，只取前5列")
        df = df.iloc[:, :5]
        df.columns = ['Z', 'A', 'E', 'Yield', 'Error']
    
    print(f"  ✓ 成功加载CSV文件，形状: {df.shape[0]}行 × {df.shape[1]}列")
    
    # ---------- 新增：Yield转对数空间 ----------
    EPS = 1e-12  # 防止Yield=0时log出错，物理上极低产额等效为1e-12
    df['Yield_log'] = np.log(df['Yield'].values + EPS)
    print(f"  ✓ Yield已转为对数空间，原始Yield范围: [{df['Yield'].min():.2e}, {df['Yield'].max():.2e}]")
    print(f"    对数Yield范围: [{df['Yield_log'].min():.6f}, {df['Yield_log'].max():.6f}]")
    
    # ---------- 原有物理计算逻辑（完全不变） ----------
    z_scaler = joblib.load(Z_SCALER_FILE)
    a_scaler = joblib.load(A_SCALER_FILE)
    df['Z_original'] = z_scaler.inverse_transform(df['Z'].values.reshape(-1, 1)).round().astype(int)
    df['A_original'] = a_scaler.inverse_transform(df['A'].values.reshape(-1, 1)).round().astype(int)
    df['N'] = df['A_original'] - df['Z_original']
    df['I'] = (df['N'] - df['Z_original']) / df['A_original']
    
    def calculate_delta_np(row):
        N, Z, I = row['N'], row['Z_original'], row['I']
        N_even, Z_even = (N % 2 == 0), (Z % 2 == 0)
        if N_even and Z_even:                       # ee
            return 2 - abs(I)
        elif (not N_even) and (not Z_even):          # oo
            return abs(I)
        elif N_even and (not Z_even) and N > Z:      # eo, N>Z
            return 1.0
        elif (not N_even) and Z_even and N < Z:      # oe, N<Z
            return 1.0
        elif N_even and (not Z_even) and N < Z:      # eo, N<Z
            return 1 - abs(I)
        elif (not N_even) and Z_even and N > Z:      # oe, N>Z
            return 1 - abs(I)
        else:
            return 1.0
    
    df['delta_np'] = df.apply(calculate_delta_np, axis=1)
    print(f"  ✓ delta_np计算完成，范围: [{df['delta_np'].min():.6f}, {df['delta_np'].max():.6f}]")
    print("\n    前5行核心数据:")
    print(df[['Z_original', 'A_original', 'N', 'E', 'I', 'delta_np', 'Yield', 'Yield_log', 'Error']].head().to_string())

except Exception as e:
    print(f"  ✗ 加载数据或计算delta_np失败: {e}")
    exit(1)

# ========== 4. 加载/生成Scaler文件 ==========
print("\n[步骤4/6] 加载/生成归一化参数...")
scalers = {}
# 加载原始Yield scaler（仅参考，不参与训练）
scalers['Yield_original'] = joblib.load(YIELD_SCALER_FILE)
# 生成delta_np scaler（派生特征，每次重新计算）
scalers['delta_np'] = StandardScaler()
scalers['delta_np'].fit(df['delta_np'].values.reshape(-1, 1))
joblib.dump(scalers['delta_np'], DELTA_NP_SCALER_FILE)
print(f"  ✓ delta_np scaler已更新: {DELTA_NP_SCALER_FILE}")
# ---------- 新增：生成对数Yield scaler ----------
scalers['Yield_log'] = StandardScaler()
scalers['Yield_log'].fit(df['Yield_log'].values.reshape(-1, 1))
joblib.dump(scalers['Yield_log'], LOG_YIELD_SCALER_FILE)
print(f"  ✓ 对数Yield scaler已生成: {LOG_YIELD_SCALER_FILE}")
print(f"    对数Yield均值: {scalers['Yield_log'].mean_[0]:.6f}, 标准差: {scalers['Yield_log'].scale_[0]:.6f}")

# ========== 5. 验证数据归一化状态 ==========
print("\n[步骤5/6] 验证数据归一化状态...")
# 输入特征（归一化后）
z_norm = df['Z'].values
a_norm = df['A'].values
e_norm = df['E'].values
delta_np_norm = scalers['delta_np'].transform(df['delta_np'].values.reshape(-1, 1)).flatten()
print("    输入特征（归一化空间）:")
print(f"    Z: [{z_norm.min():.6f}, {z_norm.max():.6f}] | 均值: {z_norm.mean():.6f}")
print(f"    A: [{a_norm.min():.6f}, {a_norm.max():.6f}] | 均值: {a_norm.mean():.6f}")
print(f"    E: [{e_norm.min():.6f}, {e_norm.max():.6f}] | 均值: {e_norm.mean():.6f}")
print(f"    delta_np: [{delta_np_norm.min():.6f}, {delta_np_norm.max():.6f}] | 均值: {delta_np_norm.mean():.6f}")

# 目标变量（对数空间归一化后）
yield_log_raw = df['Yield_log'].values
yield_log_norm = scalers['Yield_log'].transform(yield_log_raw.reshape(-1, 1)).flatten()
print("\n    目标变量（对数空间归一化后）:")
print(f"    范围: [{yield_log_norm.min():.6f}, {yield_log_norm.max():.6f}]")
print(f"    均值: {yield_log_norm.mean():.6f}, 标准差: {yield_log_norm.std():.6f}")

if 'Error' in df.columns:
    error_vals = df['Error'].values
    print(f"\n    误差列Error统计:")
    print(f"    范围: [{error_vals.min():.2e}, {error_vals.max():.2e}] | 均值: {error_vals.mean():.2e}")

# ========== 6. 准备训练数据（全训练集模式） ==========
print("\n[步骤6/6] 准备训练数据（100%训练集模式）...")
# 输入特征：归一化Z/A/E + 归一化delta_np
z_feat = df['Z'].values.reshape(-1, 1)
a_feat = df['A'].values.reshape(-1, 1)
e_feat = df['E'].values.reshape(-1, 1)
delta_np_feat = scalers['delta_np'].transform(df['delta_np'].values.reshape(-1, 1))
X = np.concatenate([z_feat, a_feat, e_feat, delta_np_feat], axis=1)
# 目标变量：归一化对数Yield
y = scalers['Yield_log'].transform(df['Yield_log'].values.reshape(-1, 1))

print(f"    ✓ 训练集样本数: {X.shape[0]} (100%全量)")
print(f"    ✓ 输入特征维度: {X.shape[1]} (Z_norm, A_norm, E_norm, delta_np)")
print(f"    ✓ 目标变量: 归一化对数Yield")

# 误差列处理
if 'Error' in df.columns:
    error_train = df['Error'].values.reshape(-1, 1)
    print(f"    ✓ 加载误差列Error: {len(error_train)}个")
    has_error = True
else:
    error_train = None
    has_error = False
    print(f"    ⚠️ 未找到Error列")

# 转换为PyTorch张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    ✓ 计算设备: {device}")

X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device)
error_train_tensor = torch.tensor(error_train, dtype=torch.float32).to(device) if has_error else None

# ========== 7. 保存预处理结果 ==========
print("\n[保存结果] 保存预处理数据...")
data_dict = {
    'X_train': X,
    'y_train': y,
    'X_train_tensor': X_train_tensor,
    'y_train_tensor': y_train_tensor,
    'device': device,
    'scalers': scalers,
    'feature_names': ['Z_norm', 'A_norm', 'E_norm', 'delta_np'],
    'target_name': 'Yield_log',  # 明确目标是对数Yield
    'raw_data': {
        'Z': df['Z_original'].values,
        'A': df['A_original'].values,
        'N': df['N'].values,
        'E': df['E'].values,
        'I': df['I'].values,
        'delta_np': df['delta_np'].values,
        'Yield_original': df['Yield'].values,
        'Yield_log': df['Yield_log'].values,
        'Error': df['Error'].values if has_error else None
    },
    'data_info': {
        'train_size': X.shape[0],
        'val_size': 0,
        'test_size': 0,
        'num_features': X.shape[1],
        'device': str(device),
        'data_source': 'GEF_theoretical',
        'split_mode': 'full_train_no_validation',
        'feature_type': 'ZAE_norm_plus_delta_np',
        'loss_type': 'log_space_MSE',  # 明确损失函数类型
        'physics_notes': '对数空间训练避免负值，自动偏向高产额区，符合裂变产额对数正态分布特性'
    }
}

if has_error:
    data_dict['error_train'] = error_train
    data_dict['error_train_tensor'] = error_train_tensor
    data_dict['has_error_column'] = True
else:
    data_dict['has_error_column'] = False

# 输出文件名（避免覆盖旧版本）
output_file = "preprocessed_gef_data_delta_np_with_ZAE_logY.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)
print(f"    ✓ 数据已保存: {output_file}")

# ========== 8. 生成统计报告 ==========
print("\n" + "="*60)
print("数据加载完成！核心信息摘要:")
print("="*60)
print(f"1. 数据规模: {df.shape[0]}样本 | 4维输入特征 | 对数空间目标")
print(f"2. 输入特征: Z_norm(质子数), A_norm(质量数), E_norm(激发能), delta_np(对关联修正)")
print(f"3. 目标变量: log(Yield + 1e-12)，反变换后无负值")
print(f"4. 训练策略: 100%全量训练，无验证/测试集，早停基于训练损失")
print(f"5. 输出文件: {output_file}")
print(f"6. 后续训练建议: 使用普通MSE损失，预测后先反归一化再取指数得到原始Yield")

# 保存详细报告
report = f"""GEF数据加载信息（对数空间Yield版 - 全训练集模式）
========================================
生成时间: {pd.Timestamp.now()}
数据文件: {CSV_FILE} | 样本数: {df.shape[0]}
物理特征: Z/A/E(归一化) + delta_np(Moller-Nix对关联修正)
目标变量: log(Yield + 1e-12)（对数空间，避免负值，偏向高产额）

核心统计:
输入特征（归一化后）:
  Z: 范围[{z_norm.min():.6f}, {z_norm.max():.6f}] | 均值{z_norm.mean():.6f}
  A: 范围[{a_norm.min():.6f}, {a_norm.max():.6f}] | 均值{a_norm.mean():.6f}
  E: 范围[{e_norm.min():.6f}, {e_norm.max():.6f}] | 均值{e_norm.mean():.6f}
  delta_np: 范围[{delta_np_norm.min():.6f}, {delta_np_norm.max():.6f}] | 均值{delta_np_norm.mean():.6f}

目标变量:
  原始Yield范围: [{df['Yield'].min():.2e}, {df['Yield'].max():.2e}]
  对数Yield范围: [{df['Yield_log'].min():.6f}, {df['Yield_log'].max():.6f}]
  归一化对数Yield范围: [{yield_log_norm.min():.6f}, {yield_log_norm.max():.6f}] | 均值{yield_log_norm.mean():.6f}

训练配置:
  损失函数: 对数空间MSE（无需加权，自动偏向高产额区）
  设备: {device}
  输出文件: {output_file}
"""
report_file = "gef_data_loading_info_delta_np_with_ZAE_logY.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  ✓ 详细报告已保存: {report_file}")
print("="*60)

print("\n【重要提示】训练后预测结果反变换流程:")
print("1. 模型输出 → 归一化对数Yield")
print("2. 用log_yield_scaler反归一化 → 对数Yield")
print("3. 取指数exp() → 原始空间Yield（天然非负）")
print("="*60)