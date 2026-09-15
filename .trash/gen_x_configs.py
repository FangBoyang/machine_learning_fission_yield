# -*- coding: utf-8 -*-
"""一次性生成 x 系列配置（12 个）：x1..x6 的 GEF warmup + 235UALL finetune。

架构/训练超参全部继承 w，唯一变量是 data.augment。
用法：
    C:/Users/86138/.conda/envs/fpy_kan/python.exe -u .trash/gen_x_configs.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'pipeline', 'configs')

GEF_TPL = """\
# x{i}_gef_isomer —— 复制 w{i}_gef（无 delta_np），唯一实质改动：开启 GEF 产额噪声增广
# （学长/裴老师「第二条路」：在 GEF 数据中掺高斯噪声，产额大的部分多采样一些）。
#
# 分档（按原始产额分位数）：
#   低档  后 50%   : 0 副本、σ=0.00  —— 不触碰
#   中档  50%~90%  : 1 副本、σ=0.05  —— 轻度
#   高档  前 10%   : 2 副本、σ=0.10  —— 学长指定的 10%
# 结果：9333 -> 约 15005 行（1.6x）。高产区样本占比由 10% 升到约 19%（等效权重约 2x）。
#
# 增广 seed 固定 42：6 个 x_i 共用同一份增广数据，唯一变量是 model.seed，
#   从而与 w1..w6 逐 seed 配对，可做 paired 比较。
#
# 训练超参（epochs / patience / min_delta / lr / 调度器）全部沿用 w —— 受控实验只改数据。
# 噪声会抬高 loss 地板；若日志显示早停行为异常，再单独调 patience / min_delta。
#
# 注意：data.augment 与 split.mode=held_out 冲突（01 会直接报错拒绝），
#   故增广只用于 GEF full_train，finetune 侧不开。
inherit: w{i}_gef_isomer.yaml
experiment:
  name: x{i}_gef_isomer
  description: "复制w{i}_gef(GEF异构合并,[16,16]/grid15/k3,p=0.35,full_train,monitor=train_loss,epochs2000/patience400)，开启产额噪声增广(后50%不动/中40% 1副本σ5%/前10% 2副本σ10%)，seed={i}"
data:
  augment:
    enabled: true
    seed: 42                    # 6 个 seed 共用同一份增广数据
    quantile_edges: [0.5, 0.9]
    n_copies: [0, 1, 2]         # 副本数，不含原件
    rel_sigma: [0.0, 0.05, 0.10]
    noise_mode: multiplicative  # y~ = y*(1+sigma*z), clip 到 >=0
"""

FT_TPL = """\
# x{i}_ft_235UALL_power —— 复制 w{i}_ft，唯一改动：warmup 起点换成增广版 x{i}_gef_isomer。
#
# split.seed=42 与 w/v 一致 -> 验证集逐点对齐，可与 w{i}_ft 直接 paired 比较。
# finetune 侧不开增广：235UALL 是实验数据（本身已有噪声），不应再加人工噪声；
#   且 data.augment 与 split.mode=held_out 冲突，01 会直接报错拒绝。
inherit: w{i}_ft_235UALL_power.yaml
experiment:
  name: x{i}_ft_235UALL_power
  description: "复制w{i}_ft(235U finetune,[16,16]/grid15/k3,p=0.35,val前3096行,val_loss早停,patience100)，warmup改为增广版x{i}_gef_isomer，seed={i}"
finetune:
  init_from: x{i}_gef_isomer
  reuse_scalers_from: x{i}_gef_isomer
"""

for i in range(1, 7):
    for tpl, name in ((GEF_TPL, f'x{i}_gef_isomer.yaml'),
                      (FT_TPL, f'x{i}_ft_235UALL_power.yaml')):
        path = os.path.join(CFG, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tpl.format(i=i))
        print(f"写入 {name}")
print("完成：12 个配置")
