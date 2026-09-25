"""
01i_data_loading_warm_up_delta_np_all_train_rawY_smaller.py
KAN模型训练 - 数据加载模块（归一化Z/A/E + delta_np版 - 原始空间MSE适配 - 全训练集模式 - 小模型结构版）
功能: 从data/GEF.csv加载数据，使用归一化后的Z/A/E + delta_np作为输入特征，
     目标变量保留原始空间Yield（原始空间MSE天然偏向高产额区，负值后置截断），全部作为训练集。
     本脚本为smaller变体配套预处理：配合grid=5/k=3/width缩小的小模型结构，
     用更少的B样条自由度抑制低产额区锯齿过拟合。
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
print("KAN模型训练 - 数据加载模块（原始空间Yield版 - 小模型结构版）")
print("="*60)
print("核心特性:")
print("1. 目标变量保留原始空间Yield，原始空间MSE天然放大高产额误差，优先拟合核心物理规律")
print("2. 负值后置截断处理，符合裂变产额非负的物理约束，不干扰训练优化方向")
print("3. 保留Z/A/E/delta_np物理特征，与logY版本特征完全对齐")
print("4. 本脚本为smaller变体配套：配合grid=5/k=3/width缩小的紧凑模型结构")
print("5. 小模型B样条自由度更低，从根源抑制低产额区锯齿过拟合")
print("="*60)

# ========== 1. 设置路径 ==========
print("\n[步骤1/6] 设置文件路径...")
DATA_DIR = "data/"
CSV_FILE = os.path.join(DATA_DIR, "GEF.csv")
RAW_YIELD_SCALER_FILE = os.path.join(DATA_DIR, "yield_scaler.pkl")  # 原始Yield scaler（核心，用于归一化）
DELTA_NP_SCALER_FILE = os.path.join(DATA_DIR, "delta_np_scaler.pkl")
Z_SCALER_FILE = os.path.join(DATA_DIR, "standard_scalerZ.pkl")
A_SCALER_FILE = os.path.join(DATA_DIR, "standard_scalerA.pkl")

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

# 加载原始Yield scaler（核心依赖，用于目标变量归一化）
if os.path.exists(RAW_YIELD_SCALER_FILE):
    print(f"  ✓ 找到原始Yield scaler: {RAW_YIELD_SCALER_FILE}")
else:
    missing_files.append(RAW_YIELD_SCALER_FILE)
    print(f"  ✗ 缺失原始Yield scaler: {RAW_YIELD_SCALER_FILE}")

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
    print(f"    原始Yield范围: [{df['Yield'].min():.2e}, {df['Yield'].max():.2e}]")
    
    # ---------- 原有物理计算逻辑（完全不变，和logY版本对齐） ----------
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
    print(df[['Z_original', 'A_original', 'N', 'E', 'I', 'delta_np', 'Yield', 'Error']].head().to_string())

except Exception as e:
    print(f"  ✗ 加载数据或计算delta_np失败: {e}")
    exit(1)

# ========== 4. 加载/生成Scaler文件 ==========
print("\n[步骤4/6] 加载/生成归一化参数...")
scalers = {}

# 加载原始Yield scaler（用于目标变量归一化，与历史版本保持一致）
scalers['Yield_original'] = joblib.load(RAW_YIELD_SCALER_FILE)
scaler_y = scalers['Yield_original']
if hasattr(scaler_y, 'mean_'):
    # StandardScaler
    print(f"  ✓ 原始Yield scaler加载成功: mean={scaler_y.mean_[0]:.6f}, scale={scaler_y.scale_[0]:.6f}")
elif hasattr(scaler_y, 'min_'):
    # MinMaxScaler
    print(f"  ✓ 原始Yield scaler加载成功: min={scaler_y.min_[0]:.6f}, scale={scaler_y.scale_[0]:.6f}")
else:
    print(f"  ✓ 原始Yield scaler加载成功（类型: {type(scaler_y).__name__}）")

# 生成delta_np scaler（派生特征，每次重新计算，覆盖旧版本保证一致性）
scalers['delta_np'] = StandardScaler()
scalers['delta_np'].fit(df['delta_np'].values.reshape(-1, 1))
joblib.dump(scalers['delta_np'], DELTA_NP_SCALER_FILE)
print(f"  ✓ delta_np scaler已更新: {DELTA_NP_SCALER_FILE}")

# ========== 5. 验证数据归一化状态 ==========
print("\n[步骤5/6] 验证数据归一化状态...")
# 输入特征（归一化后，与logY版本完全一致）
z_norm = df['Z'].values
a_norm = df['A'].values
e_norm = df['E'].values
delta_np_norm = scalers['delta_np'].transform(df['delta_np'].values.reshape(-1, 1)).flatten()
print("    输入特征（归一化空间）:")
print(f"    Z: [{z_norm.min():.6f}, {z_norm.max():.6f}] | 均值: {z_norm.mean():.6f}")
print(f"    A: [{a_norm.min():.6f}, {a_norm.max():.6f}] | 均值: {a_norm.mean():.6f}")
print(f"    E: [{e_norm.min():.6f}, {e_norm.max():.6f}] | 均值: {e_norm.mean():.6f}")
print(f"    delta_np: [{delta_np_norm.min():.6f}, {delta_np_norm.max():.6f}] | 均值: {delta_np_norm.mean():.6f}")

# 目标变量（原始空间归一化后）
yield_raw = df['Yield'].values
yield_norm = scalers['Yield_original'].transform(yield_raw.reshape(-1, 1)).flatten()
print("\n    目标变量（原始空间归一化后）:")
print(f"    范围: [{yield_norm.min():.6f}, {yield_norm.max():.6f}]")
print(f"    均值: {yield_norm.mean():.6f}, 标准差: {yield_norm.std():.6f}")

if 'Error' in df.columns:
    error_vals = df['Error'].values
    print(f"\n    误差列Error统计:")
    print(f"    范围: [{error_vals.min():.2e}, {error_vals.max():.2e}] | 均值: {error_vals.mean():.2e}")

# ========== 6. 准备训练数据（全训练集模式） ==========
print("\n[步骤6/6] 准备训练数据（100%训练集模式）...")
# 输入特征：归一化Z/A/E + 归一化delta_np（与logY版本完全一致）
z_feat = df['Z'].values.reshape(-1, 1)
a_feat = df['A'].values.reshape(-1, 1)
e_feat = df['E'].values.reshape(-1, 1)
delta_np_feat = scalers['delta_np'].transform(df['delta_np'].values.reshape(-1, 1))
X = np.concatenate([z_feat, a_feat, e_feat, delta_np_feat], axis=1)

# 目标变量：归一化原始Yield
y = scalers['Yield_original'].transform(df['Yield'].values.reshape(-1, 1))

print(f"    ✓ 训练集样本数: {X.shape[0]} (100%全量)")
print(f"    ✓ 输入特征维度: {X.shape[1]} (Z_norm, A_norm, E_norm, delta_np)")
print(f"    ✓ 目标变量: 归一化原始Yield")

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
    'target_name': 'Yield_original',
    'raw_data': {
        'Z': df['Z_original'].values,
        'A': df['A_original'].values,
        'N': df['N'].values,
        'E': df['E'].values,
        'I': df['I'].values,
        'delta_np': df['delta_np'].values,
        'Yield_original': df['Yield'].values,
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
        'loss_type': 'raw_space_MSE',
        'training_variant': '01i_smaller',  # 标识本脚本为小模型结构版
        # 记录配套的smaller训练配置，供02i训练脚本校验
        'paired_training_config': {
            'width_template': '[input_dim, 12, 12, 1]',  # 24→12，砍半宽度
            'grid': 5,          # 20→5，消除锯齿最关键
            'k': 3,             # 4→3，二次样条更平滑
            'learning_rate': 0.01,
            'batch_size': 256,
            'epochs': 2000,
            'patience': 300,
            'min_delta': 5e-6,
            'T_0': 300,         # 余弦周期与patience匹配
        },
        'physics_notes': '原始空间MSE天然放大高产额误差，优先拟合核心物理规律；负值后置截断，符合裂变产额非负约束；小模型结构(grid=5,k=3,width缩小)从根源抑制B样条过拟合导致的低产额锯齿'
    }
}

if has_error:
    data_dict['error_train'] = error_train
    data_dict['error_train_tensor'] = error_train_tensor
    data_dict['has_error_column'] = True
else:
    data_dict['has_error_column'] = False

# 输出文件名（与delta_np+rawY版本一致，01i/02i共用同一份预处理数据）
output_file = "preprocessed_gef_data_delta_np_with_ZAE_rawY.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)
print(f"    ✓ 数据已保存: {output_file}")

# ========== 8. 生成统计报告 ==========
print("\n" + "="*60)
print("数据加载完成！核心信息摘要:")
print("="*60)
print(f"1. 数据规模: {df.shape[0]}样本 | 4维输入特征 | 原始空间目标")
print(f"2. 输入特征: Z_norm(质子数), A_norm(质量数), E_norm(激发能), delta_np(对关联修正)")
print(f"3. 目标变量: 归一化原始Yield，预测后负值截断为0，符合物理规律")
print(f"4. 训练策略: 100%全量训练，无验证/测试集，早停基于训练损失")
print(f"5. 输出文件: {output_file}")
print(f"6. 配套训练脚本: 02i（smaller变体: width=[·,12,12,1], grid=5, k=3）")
print(f"7. 后续训练建议: 使用普通MSE损失，预测后用yield_original_scaler反归一化，负值截断得到最终Yield")

# 保存详细报告（带smaller后缀，与其他变体隔离）
report = f"""GEF数据加载信息（原始空间Yield版 - 全训练集模式 - 小模型结构版）
========================================
生成时间: {pd.Timestamp.now()}
数据文件: {CSV_FILE} | 样本数: {df.shape[0]}
物理特征: Z/A/E(归一化) + delta_np(Moller-Nix对关联修正)
目标变量: 原始空间Yield（未做对数变换，适配原始空间MSE损失）
训练变体: 01i_smaller（配套02i: 紧凑模型结构抑制锯齿）

核心统计:
输入特征（归一化后）:
  Z: 范围[{z_norm.min():.6f}, {z_norm.max():.6f}] | 均值{z_norm.mean():.6f}
  A: 范围[{a_norm.min():.6f}, {a_norm.max():.6f}] | 均值{a_norm.mean():.6f}
  E: 范围[{e_norm.min():.6f}, {e_norm.max():.6f}] | 均值{e_norm.mean():.6f}
  delta_np: 范围[{delta_np_norm.min():.6f}, {delta_np_norm.max():.6f}] | 均值{delta_np_norm.mean():.6f}

目标变量:
  原始Yield范围: [{df['Yield'].min():.2e}, {df['Yield'].max():.2e}]
  归一化原始Yield范围: [{yield_norm.min():.6f}, {yield_norm.max():.6f}] | 均值{yield_norm.mean():.6f}

配套训练配置（smaller变体）:
  模型结构: width=[input_dim, 12, 12, 1]（宽度从24砍半）
  grid: 5（从20大幅降低，剥夺B样条过拟合自由度）
  k: 3（三次→二次样条，弯曲能力降低）
  learning_rate: 0.01（配合小模型防震荡）
  batch_size: 256（小batch梯度噪声有助泛化）
  epochs上限: 2000
  patience: 300（对应T_0=300周期）
  min_delta: 5e-6
  损失函数: 原始空间MSE（平方损失天然放大高产额误差）

训练配置:
  设备: {device}
  输出文件: {output_file}
"""
report_file = "gef_data_loading_info_delta_np_with_ZAE_rawY_smaller.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  ✓ 详细报告已保存: {report_file}")
print("="*60)

print("\n【重要提示】训练后预测结果反变换流程:")
print("1. 模型输出 → 归一化原始Yield")
print("2. 用yield_original_scaler反归一化 → 原始空间Yield")
print("3. 负值截断: np.clip(y_pred, 0, None) → 符合物理规律的最终产额")
print("4. 配套训练脚本 02i 使用 smaller 配置(grid=5,k=3,width缩小)，请勿混用其他变体")
print("="*60)