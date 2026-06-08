"""
特征重要性统计分析脚本（适配无表头 CSV）
数据来源：data/GEF.csv（无列名，第一行即为数据）
使用方法：请先确认 GEF.csv 中各列的顺序，然后在下面 COLUMN_NAMES 中对应修改。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ========== ★ 请根据实际列顺序修改此处 ==========
COLUMN_NAMES = ['Z', 'A', 'E', 'Yield']   # 例如：若第1列是质量数A，第2列是电荷数Z，则改为 ['A','Z','E','Yield']
# ================================================

print("="*60)
print("GEF 数据特征统计分析（无表头版）")
print("="*60)

# ========== 1. 加载数据 ==========
print("\n[1/4] 加载数据...")

csv_path = "data/GEF.csv"
try:
    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
    print(f"  ✓ 成功加载数据: {csv_path}")
    print(f"    数据形状: {df.shape}")
    print(f"    列名: {list(df.columns)}")
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    exit(1)

# 检查必要的列（至少需要 Z, A, E, Yield 四列，否则请修改下面的 required_columns）
required_columns = ['Z', 'A', 'E', 'Yield']
missing_cols = [c for c in required_columns if c not in df.columns]
if missing_cols:
    print(f"  ✗ 数据缺少列: {missing_cols}")
    print(f"    请检查 COLUMN_NAMES 是否与 GEF.csv 的实际列顺序一致。")
    exit(1)

# 显示数据基本信息
print("\n  数据基本信息:")
print(f"    - 样本数: {len(df)}")
print(f"    - Z范围: [{df['Z'].min():.3f}, {df['Z'].max():.3f}]")
print(f"    - A范围: [{df['A'].min():.3f}, {df['A'].max():.3f}]")
print(f"    - E范围: [{df['E'].min():.3f}, {df['E'].max():.3f}]")
print(f"    - Yield范围: [{df['Yield'].min():.3e}, {df['Yield'].max():.3e}]")
print(f"    - 零值Yield数量: {(df['Yield'] == 0).sum()} ({100*(df['Yield']==0).sum()/len(df):.1f}%)")

# ========== 2. 特征工程 ==========
print("\n[2/4] 计算物理特征...")

df['N'] = df['A'] - df['Z']

magic_numbers = np.array([2, 8, 20, 28, 50, 82, 126], dtype=np.float32)
sigma = 2.0

def compute_magic_proximity_vectorized(values):
    distances = np.abs(values[:, np.newaxis] - magic_numbers[np.newaxis, :])
    proximities = np.exp(-distances**2 / (2 * sigma**2)).sum(axis=1)
    return proximities

df['Z_parity'] = df['Z'] % 2
df['N_parity'] = df['N'] % 2
df['parity_product'] = df['Z_parity'] * df['N_parity']

def get_parity_category(z_parity, n_parity):
    if z_parity == 0 and n_parity == 0:
        return 'EE'
    elif z_parity == 1 and n_parity == 1:
        return 'OO'
    elif z_parity == 0 and n_parity == 1:
        return 'EO'
    else:
        return 'OE'

df['parity_category'] = df.apply(lambda x: get_parity_category(x['Z_parity'], x['N_parity']), axis=1)

print("  奇偶性特征统计:")
parity_counts = df['parity_category'].value_counts()
for category, count in parity_counts.items():
    print(f"    - {category}: {count}样本 ({100*count/len(df):.1f}%)")

print("  计算幻数接近度...")
df['Z_magic_prox'] = compute_magic_proximity_vectorized(df['Z'].values)
df['N_magic_prox'] = compute_magic_proximity_vectorized(df['N'].values)

def compute_closest_magic_distance(values):
    distances = np.abs(values[:, np.newaxis] - magic_numbers[np.newaxis, :])
    return distances.min(axis=1)

df['Z_magic_dist'] = compute_closest_magic_distance(df['Z'].values)
df['N_magic_dist'] = compute_closest_magic_distance(df['N'].values)

df['Z_shell_closure'] = (df['Z_magic_dist'] < 2).astype(int)
df['N_shell_closure'] = (df['N_magic_dist'] < 2).astype(int)
df['any_shell_closure'] = (df['Z_shell_closure'] | df['N_shell_closure']).astype(int)

df['N_over_Z'] = df['N'] / (df['Z'] + 1e-12)
df['symmetry_energy'] = (df['N'] - df['Z'])**2 / (4 * df['A'])
df['mass_excess'] = df['A'] - 2 * df['Z']

print(f"\n  特征计算完成，总特征数: {len(df.columns)}")
new_features = ['N', 'Z_parity', 'N_parity', 'parity_product', 'parity_category',
                'Z_magic_prox', 'N_magic_prox', 'Z_magic_dist', 'N_magic_dist',
                'Z_shell_closure', 'N_shell_closure', 'any_shell_closure',
                'N_over_Z', 'symmetry_energy', 'mass_excess']
for i, feat in enumerate(new_features, 1):
    print(f"    {i:2d}. {feat}")

# ========== 3. 统计分析 ==========
print("\n[3/4] 进行统计分析...")

target = 'Yield'

continuous_features = ['Z', 'A', 'N', 'E',
                       'Z_magic_prox', 'N_magic_prox',
                       'Z_magic_dist', 'N_magic_dist',
                       'N_over_Z', 'symmetry_energy', 'mass_excess']

print("\n  📊 连续特征与Yield的相关系数分析:")
print("  " + "="*50)

correlation_results = []
for feature in continuous_features:
    if feature in df.columns:
        valid_mask = (df[feature].notna()) & (df[target].notna())
        if valid_mask.sum() > 10:
            pearson_corr, pearson_p = stats.pearsonr(df.loc[valid_mask, feature],
                                                      df.loc[valid_mask, target])
            spearman_corr, spearman_p = stats.spearmanr(df.loc[valid_mask, feature],
                                                         df.loc[valid_mask, target])
            correlation_results.append({
                'feature': feature,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            })
            sig_pearson = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
            sig_spearman = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""
            print(f"    {feature:20s}: Pearson={pearson_corr:7.3f}{sig_pearson:3s} "
                  f"| Spearman={spearman_corr:7.3f}{sig_spearman:3s}")

corr_df = pd.DataFrame(correlation_results)

categorical_features = ['Z_parity', 'N_parity', 'parity_category',
                        'Z_shell_closure', 'N_shell_closure', 'any_shell_closure']

print("\n  📈 分类特征与Yield的关系分析:")
print("  " + "="*50)

for feature in categorical_features:
    if feature in df.columns:
        if feature == 'parity_category':
            groups = df.groupby(feature)[target]
        else:
            groups = df.groupby(feature)[target]

        group_stats = groups.agg(['mean', 'std', 'count'])
        print(f"\n    {feature}:")
        for idx, row in group_stats.iterrows():
            print(f"      {idx}: 均值={row['mean']:.3e}, 标准差={row['std']:.3e}, 样本数={int(row['count'])}")

        if feature in ['Z_parity', 'N_parity', 'Z_shell_closure', 'N_shell_closure', 'any_shell_closure']:
            group0 = df[df[feature] == 0][target]
            group1 = df[df[feature] == 1][target]
            if len(group0) > 1 and len(group1) > 1:
                t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
                print(f"      t检验: t={t_stat:.3f}, p={p_value:.3e}")

print("\n  🔍 特征重要性排序 (基于绝对相关系数):")
print("  " + "="*50)

corr_df['abs_spearman'] = corr_df['spearman_corr'].abs()
sorted_features = corr_df.sort_values('abs_spearman', ascending=False)

for i, (_, row) in enumerate(sorted_features.iterrows(), 1):
    sig = "***" if row['spearman_p'] < 0.001 else "**" if row['spearman_p'] < 0.01 else "*" if row['spearman_p'] < 0.05 else ""
    print(f"    {i:2d}. {row['feature']:20s}: |ρ|={row['abs_spearman']:.3f}{sig}")

# ========== 4. 可视化 ==========
print("\n[4/4] 生成可视化图表...")

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("results/GEF_analysis", exist_ok=True)

# 4.1 热力图
print("\n  生成相关系数热力图...")
top_features = sorted_features.head(10)['feature'].tolist()
top_features.append(target)
corr_matrix = df[top_features].corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='left')
ax.set_yticklabels(corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.7 else 'white')
ax.set_title('Spearman Correlation Matrix of Top Features (GEF)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('results/GEF_analysis/correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("  ✓ 热力图保存: results/GEF_analysis/correlation_heatmap.png")

# 4.2 散点图
print("\n  生成连续特征与Yield的散点图...")
top_continuous = sorted_features[sorted_features['feature'].isin(continuous_features)].head(4)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, (_, row) in enumerate(top_continuous.iterrows()):
    ax = axes[idx]
    feature = row['feature']
    log_yield = np.log10(df[target] + 1e-12)
    ax.scatter(df[feature], log_yield, alpha=0.6, s=10, c='blue')
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('log10(Yield)', fontsize=12)
    if len(df[feature].unique()) > 1:
        z = np.polyfit(df[feature], log_yield, 1)
        p = np.poly1d(z)
        ax.plot(df[feature].sort_values(), p(df[feature].sort_values()),
                "r--", alpha=0.8, linewidth=2, label=f'ρ={row["spearman_corr"]:.3f}')
        ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{feature} vs Yield (ρ={row["spearman_corr"]:.3f})', fontsize=12)
plt.suptitle('Top Continuous Features vs Yield (log scale) – GEF', fontsize=16)
plt.tight_layout()
plt.savefig('results/GEF_analysis/top_features_scatter.png', dpi=150, bbox_inches='tight')
print("  ✓ 散点图保存: results/GEF_analysis/top_features_scatter.png")

# 4.3 箱线图
print("\n  生成分类特征的箱线图...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, feature in enumerate(categorical_features[:6]):
    if feature in df.columns:
        ax = axes[idx]
        df_log = df.copy()
        df_log['log_Yield'] = np.log10(df_log[target] + 1e-12)
        if feature == 'parity_category':
            category_order = ['EE', 'EO', 'OE', 'OO']
            data_to_plot = [df_log[df_log[feature] == cat]['log_Yield'] for cat in category_order]
            ax.boxplot(data_to_plot, labels=category_order)
        else:
            df_log.boxplot(column='log_Yield', by=feature, ax=ax)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('log10(Yield)', fontsize=12)
        ax.set_title(f'Yield by {feature}', fontsize=12)
        ax.grid(True, alpha=0.3)
plt.suptitle('Yield Distribution by Categorical Features – GEF', fontsize=16)
plt.tight_layout()
plt.savefig('results/GEF_analysis/categorical_features_boxplot.png', dpi=150, bbox_inches='tight')
print("  ✓ 箱线图保存: results/GEF_analysis/categorical_features_boxplot.png")

# 4.4 特征重要性条形图
print("\n  生成特征重要性条形图...")
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(sorted_features))
bars = ax.barh(y_pos, sorted_features['abs_spearman'])
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_features['feature'])
ax.invert_yaxis()
ax.set_xlabel('Absolute Spearman Correlation (|ρ|)', fontsize=12)
ax.set_title('Feature Importance Ranking – GEF', fontsize=16)
for i, (bar, p_val) in enumerate(zip(bars, sorted_features['spearman_p'])):
    width = bar.get_width()
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}{sig}', ha='left', va='center')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/GEF_analysis/feature_importance_bar.png', dpi=150, bbox_inches='tight')
print("  ✓ 特征重要性图保存: results/GEF_analysis/feature_importance_bar.png")

# 保存结果
print("\n  💾 保存统计分析结果...")
corr_df.to_csv('results/GEF_analysis/correlation_analysis.csv', index=False)
df.to_csv('results/GEF_analysis/enhanced_dataset.csv', index=False)

with open('results/GEF_analysis/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("GEF 特征重要性统计分析报告\n")
    f.write("="*60 + "\n\n")
    f.write("1. 数据概览\n")
    f.write(f"   原始数据文件: {csv_path}\n")
    f.write(f"   样本数量: {len(df)}\n")
    f.write(f"   原始特征数: 4 (Z, A, E, Yield)\n")
    f.write(f"   新增特征数: {len(new_features)}\n")
    f.write(f"   总特征数: {len(df.columns)}\n\n")
    f.write("2. 关键发现\n")
    f.write("   2.1 与Yield相关性最强的特征 (Spearman |ρ|):\n")
    for i, (_, row) in enumerate(sorted_features.head(5).iterrows(), 1):
        f.write(f"      {i}. {row['feature']:20s}: |ρ| = {row['abs_spearman']:.3f} ")
        if row['spearman_p'] < 0.001:
            f.write("(p < 0.001 ***)\n")
        elif row['spearman_p'] < 0.01:
            f.write("(p < 0.01 **)\n")
        elif row['spearman_p'] < 0.05:
            f.write("(p < 0.05 *)\n")
        else:
            f.write("(p = {:.3f})\n".format(row['spearman_p']))
    f.write("\n   2.2 奇偶性效应:\n")
    for category, count in parity_counts.items():
        mean_yield = df[df['parity_category'] == category][target].mean()
        f.write(f"      {category}: {count}样本, 平均Yield = {mean_yield:.3e}\n")
    f.write("\n   2.3 幻数效应:\n")
    shell_closure_mean = df[df['any_shell_closure'] == 1][target].mean()
    non_shell_mean = df[df['any_shell_closure'] == 0][target].mean()
    f.write(f"      壳闭合核: {df['any_shell_closure'].sum()}样本, 平均Yield = {shell_closure_mean:.3e}\n")
    f.write(f"      非壳闭合核: {(df['any_shell_closure'] == 0).sum()}样本, 平均Yield = {non_shell_mean:.3e}\n")
    f.write("\n3. 建议\n")
    f.write("   基于统计分析，建议在机器学习模型中优先考虑以下特征:\n")
    for i, (_, row) in enumerate(sorted_features.head(5).iterrows(), 1):
        f.write(f"     {i}. {row['feature']}\n")
    f.write("\n   对于分类特征，建议进行独热编码或目标编码。\n")
    f.write("   对于连续特征，注意检查与现有特征(Z, A, E)的多重共线性。\n")
    f.write("\n4. 生成文件\n")
    f.write("   - correlation_analysis.csv\n")
    f.write("   - enhanced_dataset.csv\n")
    f.write("   - 4张可视化图表 (PNG格式)\n")
    f.write("   - 本报告文件\n")

print("  ✓ 分析报告保存: results/GEF_analysis/analysis_report.txt")

print("\n" + "="*60)
print("GEF 特征统计分析完成!")
print("="*60)
print("关键发现:")
print(f"  1. 分析了 {len(continuous_features)} 个连续特征和 {len(categorical_features)} 个分类特征")
print(f"  2. 生成 {len(sorted_features)} 个特征的相关系数排名")
print(f"  3. 创建了 4 张可视化图表")
print(f"  4. 所有结果保存在 results/GEF_analysis/ 目录中")

print("\n最重要的3个特征 (基于|Spearman ρ|):")
for i, (_, row) in enumerate(sorted_features.head(3).iterrows(), 1):
    print(f"  {i}. {row['feature']:20s}: ρ = {row['spearman_corr']:.3f} (p = {row['spearman_p']:.3e})")

print("\n下一步建议:")
print("  1. 查看 results/GEF_analysis/analysis_report.txt 获取详细分析")
print("  2. 根据特征重要性选择top-K特征进行模型训练")
print("  3. 考虑特征组合或交互项")
print("  4. 对分类特征进行适当的编码处理")
print("="*60)