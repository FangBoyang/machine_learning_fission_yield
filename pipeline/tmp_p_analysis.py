# 临时统计脚本：为 KAN 的目标幂次变换 t=(y+eps)^p 选择最佳 p。
# 目标：兼顾高低产额，但更关注高产额区。y 为 minimax 归一化后的 GEF 第 4 列（无表头）。
#
# 数学依据（核心一行）：
#   训练在 t = y^p 空间做普通 MSE。某点在原始 y 上的小误差 δy 在 t 空间造成的平方误差为
#       (Δt)^2 = (d(y^p)/dy · δy)^2 = (p · y^{p-1} · δy)^2 = p^2 · y^{2p-2} · (δy)^2
#   因此每个样本对优化的“有效损失权重”为  w(y) ∝ y^{2p-2}  （p^2 对全体是常数，比较时可约去）。
#   含义：
#     - p = 1      → w(y)=1，标准 MSE，高低产额等权；
#     - p > 1      → 2p-2 > 0，权重随 y 增大 → 高产额区被放大（符合“更关注高产额”）；
#     - p < 1      → 2p-2 < 0，权重随 y 减小 → 低产额被放大（j 的 p=0.25 即此情形）。
#   另外：p<1 时在 y=0 处导数 d(y^p)/dy = p·y^{p-1} → ∞（奇异），低产额区拟合困难、数值不稳；
#        p≥1 在 [0,1] 上处处光滑，从可学习性/条件数角度也优于 p<1。
#   故结论方向：p 应当 ≥ 1（最好略大于 1），用权重公式从数据分布上定出具体数值。

import csv
import numpy as np

DATA = r'F:\computer_science\machine_learning_fission_yield\data\GEF.csv'
EPS = 1e-12
COL = 3  # 第 4 列（0-based）

def load_y(path):
    ys = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) > COL:
                try:
                    ys.append(float(row[COL]))
                except ValueError:
                    pass
    a = np.array(ys, dtype=np.float64)
    a = np.clip(a, 0.0, None)          # 与 01_preprocess 一致
    return a

def weight(y, p):
    # w(y) ∝ (y+eps)^{2p-2}
    return np.power(y + EPS, 2.0 * p - 2.0)

def high_share(y, p, thr):
    w = weight(y, p)
    mask = y >= thr
    if mask.sum() == 0:
        return float('nan')
    return w[mask].sum() / w.sum()

def solve_p_for_share(y, thr, target, p_lo=1.0, p_hi=8.0, tol=1e-3):
    # high_share(p) 随 p 单调递增 → 二分
    f_lo = high_share(y, p_lo, thr) - target
    f_hi = high_share(y, p_hi, thr) - target
    if f_lo > 0:
        return p_lo
    if f_hi < 0:
        return p_hi
    for _ in range(60):
        pm = 0.5 * (p_lo + p_hi)
        fm = high_share(y, pm, thr) - target
        if abs(fm) < tol:
            return pm
        if fm > 0:
            p_hi = pm
        else:
            p_lo = pm
    return 0.5 * (p_lo + p_hi)

def main():
    y = load_y(DATA)
    n = len(y)
    print('=' * 70)
    print('GEF 第4列（归一化 y）分布统计  (n = %d)' % n)
    print('=' * 70)
    print('min=%.6g  max=%.6g  mean=%.6g  median=%.6g' % (y.min(), y.max(), y.mean(), np.median(y)))
    print('std=%.6g  skew=%.4g' % (y.std(), float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + EPS))))
    for q in (50, 75, 90, 95, 99):
        print('  %2dth percentile y = %.6g' % (q, np.percentile(y, q)))
    print('区间占比:  y<0.1 = %.1f%%  0.1<=y<=0.5 = %.1f%%  y>0.5 = %.1f%%' %
          (100 * (y < 0.1).mean(), 100 * ((y >= 0.1) & (y <= 0.5)).mean(), 100 * (y > 0.5).mean()))

    # 高产额区定义：分位阈值 与 绝对阈值
    regions = []
    for q in (75, 90):
        thr = np.percentile(y, q)
        regions.append(('top%d%% (y>=%.4g)' % (100 - q, thr), thr))
    regions.append(('y>0.5 (峰区)', 0.5))
    print()
    for name, thr in regions:
        print('  高产额区 [%s]: 样本占比 = %.2f%%' % (name, 100 * (y >= thr).mean()))

    # 扫描 p，输出权重份额表
    print()
    print('=' * 70)
    print('不同 p 下：各高产额区获得的“有效损失权重占比” high_share(p)')
    print('（= 该区 w(y) 之和 / 全体 w(y) 之和；p=1 时恒等于样本占比）')
    print('=' * 70)
    ps = [0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    header = '  p     | ' + ' | '.join('%14s' % name for name, _ in regions) + ' |  高/中权重比(y=0.9 vs 中位)'
    print(header)
    for p in ps:
        cells = []
        for name, thr in regions:
            cells.append('%13.3f%%' % (100 * high_share(y, p, thr)))
        # 权重比：高产额点(y=0.9) 相对 中位点 的 w 之比
        w_hi = weight(np.array([0.9]), p)[0]
        w_med = weight(np.array([np.median(y)]), p)[0]
        ratio = w_hi / w_med if w_med > 0 else float('inf')
        print('  %5.2f | %s |  %s' % (p, ' | '.join(cells), ('%.2e×' % ratio)))

    # 推荐：以 top10% 区为目标，使其获得约 50% 的权重（> 其 ~10% 样本占比，明显偏向高产额，
    # 但仍给其余 90% 留一半权重 → “更关注高产额，但兼顾”）。同时给出 40%/60% 目标作参考。
    print()
    print('=' * 70)
    print('推荐 p（按“top10% 高产额区获得目标权重占比”反解）')
    print('=' * 70)
    thr10 = np.percentile(y, 90)
    for target in (0.40, 0.50, 0.60):
        p_rec = solve_p_for_share(y, thr10, target)
        print('  high_share 目标 %.0f%% → p = %.3f' % (100 * target, p_rec))

    # t 空间可学习性体检：p 过大时低产额 bulk 被压到近 0，网络难以分辨（仅作警示）
    print()
    print('=' * 70)
    print('t = y^p 空间体检（p 过大→低产额 bulk 塌缩到 ~0，网络对低/中产额失分辨力）')
    print('=' * 70)
    rec_p = solve_p_for_share(y, thr10, 0.50)
    for p in (1.0, round(rec_p, 3), 2.0, 3.0):
        t = np.power(y + EPS, p)
        print('  p=%5.2f | t.min=%.4g t.max=%.4g t.range=%.4g t.skew=%.3g | 中位t/最大t=%.4g'
              % (p, t.min(), t.max(), t.max() - t.min(),
                 float(((t - t.mean()) ** 3).mean() / (t.std() ** 3 + EPS)),
                 (np.median(t) / t.max()) if t.max() > 0 else float('nan')))

    print()
    print('结论方向：p 应 ≥ 1（p<1 在 y=0 奇异、且把权重压向已占多数的低产额）。')
    print('        建议先用 p≈%.2f（top10%% 高产额区获得 ~50%% 权重），偏向高产额同时兼顾其余。' % rec_p)
    print('        若希望更强偏向高产额，取 ~%.2f；若只想温和修正当前 p=0.25，取 ~%.2f。'
          % (solve_p_for_share(y, thr10, 0.60), solve_p_for_share(y, thr10, 0.40)))

if __name__ == '__main__':
    main()
