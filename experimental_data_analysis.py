"""
experimental_data_analysis.py
功能: 对data/235UALL.csv中的实验数据进行分析
      按实际物理能量分组，绘制裂变产额分布图
"""

import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("实验数据裂变产额分布分析")
print("="*60)

# ========== 1. 加载数据 ==========
print("\n[1/5] 加载实验数据...")

csv_path = "data/235UALL.csv"
try:
    df_exp = pd.read_csv(csv_path)
    print(f"  ✓ 成功加载数据: {csv_path}")
    print(f"    数据形状: {df_exp.shape}")
    print(f"    列名: {list(df_exp.columns)}")
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    exit(1)

# 检查必要的列
required_columns = ['Z', 'A', 'E', 'Yield']
if not all(col in df_exp.columns for col in required_columns):
    print(f"  ✗ 数据缺少必要列。现有列: {list(df_exp.columns)}")
    exit(1)

# 显示数据基本信息
print("\n  数据基本信息:")
print(f"    - 样本数: {len(df_exp)}")
print(f"    - Z范围(归一化): [{df_exp['Z'].min():.3f}, {df_exp['Z'].max():.3f}]")
print(f"    - A范围(归一化): [{df_exp['A'].min():.3f}, {df_exp['A'].max():.3f}]")
print(f"    - E范围(归一化): [{df_exp['E'].min():.3f}, {df_exp['E'].max():.3f}]")
print(f"    - Yield范围: [{df_exp['Yield'].min():.3e}, {df_exp['Yield'].max():.3e}]")
print(f"    - 零值Yield数量: {(df_exp['Yield'] == 0).sum()} ({100*(df_exp['Yield']==0).sum()/len(df_exp):.1f}%)")

# 检查是否有Error列
has_error = 'Error' in df_exp.columns
if has_error:
    print(f"    - 有Error列，范围: [{df_exp['Error'].min():.3e}, {df_exp['Error'].max():.3e}]")

# ========== 2. 反归一化获取物理值 ==========
print("\n[2/5] 反归一化获取物理值...")

# 加载scaler用于反归一化
scaler_files = {
    'Z': 'data/standard_scalerZ.pkl',
    'A': 'data/standard_scalerA.pkl', 
    'E': 'data/standard_scalerE.pkl',
    'Yield': 'data/yield_scaler.pkl'
}

scalers = {}
try:
    for name, filepath in scaler_files.items():
        scalers[name] = joblib.load(filepath)
    print("  ✓ 所有scaler加载成功")
except Exception as e:
    print(f"  ✗ 加载scaler失败: {e}")
    exit(1)

# 反归一化获取物理值
Z_physical = scalers['Z'].inverse_transform(df_exp[['Z']].values).flatten()
A_physical = scalers['A'].inverse_transform(df_exp[['A']].values).flatten()
E_physical = scalers['E'].inverse_transform(df_exp[['E']].values).flatten()
Yield_physical = scalers['Yield'].inverse_transform(df_exp[['Yield']].values).flatten()

# 添加到DataFrame
df_exp['Z_physical'] = Z_physical
df_exp['A_physical'] = A_physical
df_exp['E_physical'] = E_physical
df_exp['Yield_physical'] = Yield_physical

if has_error and 'Error' in df_exp.columns:
    Error_physical = scalers['Yield'].inverse_transform(df_exp[['Error']].values).flatten()
    df_exp['Error_physical'] = Error_physical

print(f"\n  物理值统计:")
print(f"    - Z物理范围: [{Z_physical.min():.1f}, {Z_physical.max():.1f}]")
print(f"    - A物理范围: [{A_physical.min():.1f}, {A_physical.max():.1f}]")
print(f"    - E物理范围: [{E_physical.min():.3f}, {E_physical.max():.3f}] MeV")
print(f"    - Yield物理范围: [{Yield_physical.min():.3e}, {Yield_physical.max():.3e}]")

# 分析能量值的分布
unique_E_physical = np.unique(np.round(E_physical, 3))  # 四舍五入到3位小数
print(f"\n  独特的物理能量值 ({len(unique_E_physical)} 个):")
for i, E in enumerate(unique_E_physical):
    count = np.sum(np.abs(E_physical - E) < 0.001)  # 容差0.001
    print(f"    {i+1:2d}. {E:.3f} MeV: {count} 样本")

# 对物理能量进行分组，将非常接近的值视为同一能量
E_bins = {}
tolerance = 0.1  # 0.1 MeV的容差
current_bin = []
current_bin_center = None

for E in sorted(unique_E_physical):
    if current_bin_center is None or abs(E - current_bin_center) > tolerance:
        if current_bin:
            E_bins[f"{np.mean(current_bin):.3f} MeV"] = current_bin.copy()
        current_bin = [E]
        current_bin_center = E
    else:
        current_bin.append(E)
        current_bin_center = np.mean(current_bin)

if current_bin:
    E_bins[f"{np.mean(current_bin):.3f} MeV"] = current_bin.copy()

print(f"\n  分组后的能量值 ({len(E_bins)} 组):")
for i, (label, energies) in enumerate(E_bins.items()):
    print(f"    {i+1:2d}. {label} (包含 {len(energies)} 个独特值)")

# 为每个样本分配分组标签
df_exp['E_group'] = 'Other'
for label, energies in E_bins.items():
    mask = np.zeros_like(E_physical, dtype=bool)
    for E in energies:
        mask |= (np.abs(E_physical - E) < 0.001)
    df_exp.loc[mask, 'E_group'] = label

# ========== 3. 数据聚合 ==========
print("\n[3/5] 数据聚合...")

# 按A求和（按能量分组）
print("  按A求和...")
df_sum_by_A = df_exp.groupby(['A_physical', 'E_group'])['Yield_physical'].sum().reset_index()

# 按Z求和（按能量分组）
print("  按Z求和...")
df_sum_by_Z = df_exp.groupby(['Z_physical', 'E_group'])['Yield_physical'].sum().reset_index()

# 转换为透视表，便于绘图
try:
    df_sum_by_A_pivot = df_sum_by_A.pivot(index='A_physical', columns='E_group', values='Yield_physical')
    print(f"  按A求和数据形状: {df_sum_by_A_pivot.shape}")
except Exception as e:
    print(f"  创建A透视表错误: {e}")
    df_sum_by_A_pivot = None

try:
    df_sum_by_Z_pivot = df_sum_by_Z.pivot(index='Z_physical', columns='E_group', values='Yield_physical')
    print(f"  按Z求和数据形状: {df_sum_by_Z_pivot.shape}")
except Exception as e:
    print(f"  创建Z透视表错误: {e}")
    df_sum_by_Z_pivot = None

# 统计每个能量组的样本数
print("\n  每个能量组的样本统计:")
group_stats = df_exp.groupby('E_group').agg({
    'Yield_physical': ['count', 'mean', 'std', 'min', 'max']
}).round(6)
print(group_stats)

# ========== 4. 可视化 ==========
print("\n[4/5] 创建可视化图表...")

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("results/experimental", exist_ok=True)

# 颜色映射 - 为每个能量组分配颜色
E_groups = sorted(df_exp['E_group'].unique())
n_groups = len(E_groups)

# 使用不同的颜色映射
if n_groups <= 8:
    colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
else:
    colors = plt.cm.tab20(np.linspace(0, 1, n_groups))

# 创建颜色字典
color_dict = {group: colors[i] for i, group in enumerate(E_groups)}

# 图1: 按质量数A求和
fig1, ax1 = plt.subplots(figsize=(12, 7))

if df_sum_by_A_pivot is not None and not df_sum_by_A_pivot.empty:
    for col in df_sum_by_A_pivot.columns:
        A_values = df_sum_by_A_pivot.index
        yield_sum = df_sum_by_A_pivot[col].values
        
        # 绘制曲线
        ax1.plot(A_values, yield_sum, 
                color=color_dict[col], 
                alpha=0.8, 
                linewidth=2,
                label=f'Experimental: {col}')
        
        # 标记数据点
        ax1.scatter(A_values, yield_sum, 
                   color=color_dict[col], 
                   s=20, 
                   alpha=0.6)
    
    ax1.set_xlabel('Mass Number (A)', fontsize=12)
    ax1.set_ylabel('Yield Sum (per A)', fontsize=12)
    ax1.set_title('Experimental Fission Yield Distribution by Mass Number (A) at Different Energies', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    
    # 自动设置y轴范围
    max_yield = df_sum_by_A_pivot.max().max()
    ax1.set_ylim(0, max_yield * 1.2)
    
    plt.tight_layout()
    fig1.savefig('results/experimental/experimental_yield_by_A.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 图1保存: results/experimental/experimental_yield_by_A.png")
else:
    # 备选方案: 直接从分组数据绘图
    for group in E_groups:
        subset = df_sum_by_A[df_sum_by_A['E_group'] == group]
        ax1.plot(subset['A_physical'], subset['Yield_physical'], 
                color=color_dict[group], 
                alpha=0.8, 
                linewidth=2,
                label=f'Experimental: {group}')
    
    ax1.set_xlabel('Mass Number (A)', fontsize=12)
    ax1.set_ylabel('Yield Sum (per A)', fontsize=12)
    ax1.set_title('Experimental Fission Yield Distribution by Mass Number (A) at Different Energies', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    plt.tight_layout()
    fig1.savefig('results/experimental/experimental_yield_by_A.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 图1保存: results/experimental/experimental_yield_by_A.png (使用分组数据)")

# 图2: 按电荷数Z求和
fig2, ax2 = plt.subplots(figsize=(12, 7))

if df_sum_by_Z_pivot is not None and not df_sum_by_Z_pivot.empty:
    for col in df_sum_by_Z_pivot.columns:
        Z_values = df_sum_by_Z_pivot.index
        yield_sum = df_sum_by_Z_pivot[col].values
        
        # 绘制曲线
        ax2.plot(Z_values, yield_sum, 
                color=color_dict[col], 
                alpha=0.8, 
                linewidth=2,
                label=f'Experimental: {col}')
        
        # 标记数据点
        ax2.scatter(Z_values, yield_sum, 
                   color=color_dict[col], 
                   s=20, 
                   alpha=0.6)
    
    ax2.set_xlabel('Atomic Number (Z)', fontsize=12)
    ax2.set_ylabel('Yield Sum (per Z)', fontsize=12)
    ax2.set_title('Experimental Fission Yield Distribution by Atomic Number (Z) at Different Energies', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10, ncol=2)
    
    # 自动设置y轴范围
    max_yield = df_sum_by_Z_pivot.max().max()
    ax2.set_ylim(0, max_yield * 1.2)
    
    plt.tight_layout()
    fig2.savefig('results/experimental/experimental_yield_by_Z.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 图2保存: results/experimental/experimental_yield_by_Z.png")
else:
    # 备选方案: 直接从分组数据绘图
    for group in E_groups:
        subset = df_sum_by_Z[df_sum_by_Z['E_group'] == group]
        ax2.plot(subset['Z_physical'], subset['Yield_physical'], 
                color=color_dict[group], 
                alpha=0.8, 
                linewidth=2,
                label=f'Experimental: {group}')
    
    ax2.set_xlabel('Atomic Number (Z)', fontsize=12)
    ax2.set_ylabel('Yield Sum (per Z)', fontsize=12)
    ax2.set_title('Experimental Fission Yield Distribution by Atomic Number (Z) at Different Energies', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10, ncol=2)
    plt.tight_layout()
    fig2.savefig('results/experimental/experimental_yield_by_Z.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 图2保存: results/experimental/experimental_yield_by_Z.png (使用分组数据)")

# 图3: 能量分布图
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# 左子图: 物理能量值的直方图
ax3a.hist(E_physical, bins=20, alpha=0.7, color='blue', edgecolor='black')
ax3a.set_xlabel('Physical Energy (MeV)', fontsize=12)
ax3a.set_ylabel('Frequency', fontsize=12)
ax3a.set_title('Distribution of Physical Energy Values', fontsize=14)
ax3a.grid(True, alpha=0.3)

# 右子图: 每个能量组的样本数
group_counts = df_exp['E_group'].value_counts()
bars = ax3b.bar(range(len(group_counts)), group_counts.values, 
               color=[color_dict[group] for group in group_counts.index], 
               alpha=0.8)
ax3b.set_xlabel('Energy Group', fontsize=12)
ax3b.set_ylabel('Number of Samples', fontsize=12)
ax3b.set_title('Sample Count per Energy Group', fontsize=14)
ax3b.set_xticks(range(len(group_counts)))
ax3b.set_xticklabels(group_counts.index, rotation=45, ha='right')
ax3b.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, count in zip(bars, group_counts.values):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
              f'{count}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig3.savefig('results/experimental/energy_distribution.png', dpi=150, bbox_inches='tight')
print(f"  ✓ 图3保存: results/experimental/energy_distribution.png")

# ========== 5. 保存分析结果 ==========
print("\n[5/5] 保存分析结果...")

# 保存处理后的实验数据
df_exp.to_csv('results/experimental/experimental_data_processed.csv', index=False)
print(f"  ✓ 处理后的实验数据保存: results/experimental/experimental_data_processed.csv")

# 保存聚合数据
df_sum_by_A.to_csv('results/experimental/experimental_sum_by_A.csv', index=False)
df_sum_by_Z.to_csv('results/experimental/experimental_sum_by_Z.csv', index=False)
print(f"  ✓ 聚合数据保存: results/experimental/experimental_sum_by_[A|Z].csv")

# 创建详细的分析报告
report = f"""
实验数据裂变产额分布分析报告
{'='*50}

1. 数据概览
   数据文件: {csv_path}
   总样本数: {len(df_exp)}
   原始列: {list(df_exp.columns[:5])}
   物理能量组数: {len(E_groups)}
   
2. 物理值范围
   - 原子序数(Z): {Z_physical.min():.1f} 到 {Z_physical.max():.1f}
   - 质量数(A): {A_physical.min():.1f} 到 {A_physical.max():.1f}
   - 物理能量(E): {E_physical.min():.3f} 到 {E_physical.max():.3f} MeV
   - 产额(Yield): {Yield_physical.min():.3e} 到 {Yield_physical.max():.3e}
   
3. 能量分组详情
"""
for i, group in enumerate(E_groups, 1):
    group_data = df_exp[df_exp['E_group'] == group]
    E_values = group_data['E_physical'].unique()
    E_range = f"{E_values.min():.3f}-{E_values.max():.3f} MeV" if len(E_values) > 1 else f"{E_values[0]:.3f} MeV"
    report += f"   {i:2d}. {group}: {len(group_data)} 样本, 能量范围: {E_range}\n"

report += f"""
4. 产额统计
   零值产额数量: {(df_exp['Yield'] == 0).sum()} ({(df_exp['Yield']==0).sum()/len(df_exp)*100:.1f}%)
   平均产额: {df_exp['Yield_physical'].mean():.3e}
   中位数产额: {df_exp['Yield_physical'].median():.3e}
   
5. 生成文件
   - 实验数据分布图: results/experimental/experimental_yield_by_[A|Z].png
   - 能量分布图: results/experimental/energy_distribution.png
   - 处理后的实验数据: results/experimental/experimental_data_processed.csv
   - 聚合数据: results/experimental/experimental_sum_by_[A|Z].csv
   
6. 分析完成时间: {pd.Timestamp.now()}
"""

with open('results/experimental/experimental_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✓ 分析报告保存: results/experimental/experimental_analysis_report.txt")

# 创建能量组详细统计
group_details = df_exp.groupby('E_group').agg({
    'E_physical': ['min', 'max', 'mean', 'std'],
    'Yield_physical': ['count', 'mean', 'std', 'min', 'max', 'median']
}).round(6)

group_details.to_csv('results/experimental/energy_group_statistics.csv')
print(f"  ✓ 能量组统计保存: results/experimental/energy_group_statistics.csv")

print("\n" + "="*60)
print("实验数据分析完成!")
print("="*60)
print(f"关键发现:")
print(f"  1. 分析了 {len(df_exp)} 个实验样本")
print(f"  2. 识别出 {len(E_groups)} 个不同的物理能量组")
print(f"  3. 创建了 3 张可视化图表")
print(f"  4. 所有结果保存在 results/experimental/ 目录中")

print(f"\n能量组详情:")
for i, group in enumerate(E_groups, 1):
    group_data = df_exp[df_exp['E_group'] == group]
    unique_E = np.unique(np.round(group_data['E_physical'].values, 3))
    E_str = ', '.join([f"{E:.3f}" for E in unique_E])
    print(f"  {i:2d}. {group}: {len(group_data)} 样本, 包含能量: {E_str}")

print(f"\n下一步建议:")
print(f"  1. 比较不同能量下的产额分布差异")
print(f"  2. 与模型预测结果进行对比分析")
print(f"  3. 分析特定能量下的产额异常值")
print("="*60)

# 显示图片预览
plt.show()