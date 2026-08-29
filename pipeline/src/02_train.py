# -*- coding: utf-8 -*-
"""
02_train.py — KAN 模型训练（配置驱动，支持断点续训 / 中断保存 / 渐进网格细化 / LBFGS 收尾）

流程：加载预处理 pkl → 构建 KAN 模型 → （可选）从 resume 检查点续训
      → Adam 阶段（含渐进网格细化）→ （可选）LBFGS 收尾
      → 保存 best/final/latest/resume

能力（均通过 config 控制，不影响 03/04，它们仍只读 kan_best_*.pth）：
- 断点续训：train.resume=true 时从 kan_resume_<variant>.pth 载入
  模型/优化器/调度器/epoch/早停计数/RNG/当前 grid/阶段，从断点接着训练。
- 中断保存：捕获 SIGINT/SIGTERM，在当前 epoch/step 边界保存后退出。
- 提前结束：train.stop_on_best_below 设定阈值，best_loss 一达标即停。
- 周期保存：train.checkpoint_every=N，每 N 轮落一次 latest+resume。
- 渐进网格细化：train.grid_schedule=[{epoch, grid}, ...]，到达 epoch 时
  model.refine(grid)（更密的 B-spline 网格），并重置 lr 调度器让 Adam 在新网格上
  重新收敛；早停计数同时清零（细化后 loss 会跳升再下降）。参考 KAN 论文的
  grid extension：粗网格快训 → 细化 → 精修。
- LBFGS 收尾：train.lbfgs.enabled=true 时，Adam 阶段结束后（或到达
  lbfgs.start_epoch）切换 LBFGS 做最终抛光，对光滑 B-spline 拟合常把 loss 再降一截。

resume 检查点含：model_state / optimizer_state / scheduler_state / epoch /
  best_loss / best_epoch / patience_counter / history / 当前 grid / grid_ptr /
  phase(adam|lbfgs) / lbfgs_step / 全部 RNG / config / variant / interrupted

用法：
    python src/02_train.py --config configs/<variant>.yaml
"""

import os
import argparse
import signal
import random
import time
import pickle
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kan import KAN
from common import (load_config, get_variant, output_path,
                    get_optimizer, get_scheduler, get_criterion,
                    load_pretrained_model, PROJECT_ROOT)


# ===================== 中断信号管理 =====================
_INTERRUPTED = False


def _signal_handler(signum, frame):
    """收到 SIGINT/SIGTERM 时置位标志，训练循环会在当前 epoch/step 边界退出并保存。"""
    global _INTERRUPTED
    if not _INTERRUPTED:
        _INTERRUPTED = True
        print(f"\n  [signal {signum}] 收到中断请求，将在本 epoch/step 结束后保存检查点并退出...")


def _suppress_act(model):
    """关闭 pykan 的自动建目录行为（与构造时的 save_act=False/auto_save=False 一致）。"""
    for attr in ('save_act', 'auto_save'):
        if hasattr(model, attr):
            try:
                setattr(model, attr, False)
            except Exception:
                pass
    return model


def _refine_and_rebuild(model, new_grid, cfg, device, X):
    """渐进网格细化：model.refine(new_grid) 返回含更密网格的新模型（插值保留已学模式），
    并重建优化器与 lr 调度器（调度器从头开始衰减，让 Adam 在新网格上重新收敛）。

    注意：pykan 的 refine() 依赖前向缓存的激活（spline_preacts / cache_data）。其
    forward 每次都会无条件清空这些缓存，且仅在 save_act=True 时才重新填充；而 refine
    内部会用 model.save_act 再跑一次前向。因此细化前必须把 save_act 置为 True（且细化后
    保持 True，使后续训练前向持续填充缓存，供下一轮 refine 使用），否则内部前向清空后不
    填充，initialize_from_another_model 会 IndexError。"""
    if not hasattr(model, 'refine'):
        raise RuntimeError("当前 pykan 版本不支持 model.refine()，无法做渐进网格细化；"
                           "请升级 pykan 或移除 train.grid_schedule。")
    model.save_act = True       # 关键：refine 内部前向需以此填充缓存
    model.get_act(X)            # 用全量训练数据填充 spline_preacts / cache_data
    model = model.refine(new_grid)
    model.to(device)
    model.train()
    model.save_act = True       # 保持 True：后续训练前向持续填充，供下一轮 refine
    model.auto_save = False
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)  # 重新创建 → lr 从初始值重新开始衰减
    return model, optimizer, scheduler


def _maybe_resume(variant, model, optimizer, scheduler, cfg, initial_grid):
    """若存在 resume 检查点且 cfg['train']['resume']=true，则载入全部状态续训。

    返回 (start_epoch, best_loss, best_epoch, patience_counter, history, resumed,
          current_grid, grid_ptr, phase, lbfgs_step)。
    """
    resume_path = output_path(variant, 'models', f'kan_resume_{variant}.pth')
    init = (0, float('inf'), 0, 0, {'train_loss': [], 'lr': []}, False,
            initial_grid, 0, 'adam', 0)
    if not cfg['train'].get('resume', False):
        return init
    if not os.path.exists(resume_path):
        print(f"[注] 未找到 resume 检查点 {resume_path}，按全新训练开始。")
        return init

    ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None and ckpt.get('scheduler_state') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    torch.set_rng_state(ckpt['torch_rng'])
    np.random.set_state(ckpt['numpy_rng'])
    random.setstate(ckpt['python_rng'])

    start_epoch = int(ckpt.get('epoch', 0))
    best_loss = float(ckpt.get('best_loss', float('inf')))
    best_epoch = int(ckpt.get('best_epoch', 0))
    patience_counter = int(ckpt.get('patience_counter', 0))
    history = ckpt.get('history', {'train_loss': [], 'lr': []})
    if 'train_loss' not in history:
        history['train_loss'] = []
    if 'lr' not in history:
        history['lr'] = []
    current_grid = int(ckpt.get('current_grid', initial_grid))
    grid_ptr = int(ckpt.get('grid_ptr', 0))
    phase = ckpt.get('phase', 'adam')
    lbfgs_step = int(ckpt.get('lbfgs_step', 0))
    print(f"[续训] 载入 resume 检查点：已完成 {start_epoch} 轮，phase={phase}，"
          f"grid={current_grid}，历史最佳 loss={best_loss:.4e} (epoch {best_epoch})")
    return (start_epoch, best_loss, best_epoch, patience_counter, history, True,
            current_grid, grid_ptr, phase, lbfgs_step)


def _save_checkpoint(model, cfg, variant, epoch, train_loss, path, extra=None):
    """保存模型权重检查点（best / latest 用），内嵌完整配置。"""
    # 内嵌 config 的 model.grid 必须反映模型当前实际网格（训练中可能经
    # grid_schedule/refine 改变），否则 03/04 按初始 grid 重建会因尺寸不匹配失败。
    save_cfg = copy.deepcopy(cfg)
    try:
        save_cfg['model']['grid'] = int(model.grid)
    except Exception:
        pass
    ckpt = {
        'model_state': model.state_dict(),
        'config': save_cfg,
        'config_yaml': yaml.dump(save_cfg, allow_unicode=True),
        'epoch': epoch,
        'train_loss': float(train_loss),
        'variant': variant,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def _save_resume_checkpoint(model, optimizer, scheduler, cfg, variant, epoch,
                            best_loss, best_epoch, patience_counter, history,
                            path, interrupted=False, current_grid=None,
                            grid_ptr=0, phase='adam', lbfgs_step=0, extra=None):
    """保存续训检查点：含模型 + 优化器 + 调度器 + epoch + 早停计数 + RNG + 网格/阶段等。"""
    # 内嵌 config 的 model.grid 反映当前实际网格
    save_cfg = copy.deepcopy(cfg)
    try:
        save_cfg['model']['grid'] = int(model.grid)
    except Exception:
        pass
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_loss': float(best_loss),
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'history': history,
        'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
        'current_grid': int(current_grid),
        'grid_ptr': int(grid_ptr),
        'phase': phase,
        'lbfgs_step': int(lbfgs_step),
        'config': save_cfg,
        'config_yaml': yaml.dump(save_cfg, allow_unicode=True),
        'variant': variant,
        'interrupted': bool(interrupted),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def _run_lbfgs(model, train_loader, criterion, cfg, variant, best_path, history,
              best_loss, best_epoch, best_phase, min_delta, lbfgs_cfg,
              lbfgs_step_start, current_grid, grid_ptr, patience_counter,
              checkpoint_every, resume_path):
    """LBFGS 收尾阶段：在最终网格上做抛光。返回 (best_loss, best_epoch, best_phase, lbfgs_step, avg_loss)。"""
    lbfgs_lr = float(lbfgs_cfg.get('lr', 1.0))
    lbfgs_max_iter = int(lbfgs_cfg.get('max_iter', 20))
    lbfgs_hist = int(lbfgs_cfg.get('history_size', 100))
    lbfgs_steps_total = int(lbfgs_cfg.get('steps', 300))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr,
                                  history_size=lbfgs_hist, max_iter=lbfgs_max_iter)
    model.train()
    lbfgs_step = lbfgs_step_start
    avg_loss = float('inf')
    while lbfgs_step < lbfgs_steps_total and not _INTERRUPTED:
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            def closure():
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                return loss
            optimizer.step(closure)
            with torch.no_grad():
                pred = model(bx)
                loss = criterion(pred, by)
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        lbfgs_step += 1
        history['train_loss'].append(avg_loss)
        history['lr'].append(lbfgs_lr)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_epoch = lbfgs_step
            best_phase = 'lbfgs'
            _save_checkpoint(model, cfg, variant, f'lbfgs{lbfgs_step}', best_loss,
                             best_path, extra={'history': history, 'phase': 'lbfgs'})

        if lbfgs_step % 20 == 0 or lbfgs_step == 1 or lbfgs_step == lbfgs_steps_total:
            print(f"    LBFGS step {lbfgs_step:4d}/{lbfgs_steps_total} | loss: {avg_loss:.4e}")

        if checkpoint_every > 0 and lbfgs_step % checkpoint_every == 0:
            _save_resume_checkpoint(model, optimizer, None, cfg, variant,
                                    f'lbfgs{lbfgs_step}', best_loss, best_epoch,
                                    patience_counter, history, resume_path,
                                    interrupted=False, current_grid=current_grid,
                                    grid_ptr=grid_ptr, phase='lbfgs',
                                    lbfgs_step=lbfgs_step)

    return best_loss, best_epoch, best_phase, lbfgs_step, avg_loss


def main():
    # 1. 解析 --config
    parser = argparse.ArgumentParser(description="KAN 模型训练（配置驱动，支持续训/网格细化/LBFGS）")
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)

    print("=" * 60)
    print("KAN 模型训练（配置驱动，支持续训/网格细化/LBFGS）")
    print("=" * 60)

    # 2. 加载预处理 pkl
    pkl_path = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到预处理文件: {pkl_path}（请先运行 01_preprocess.py）")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[1] 加载预处理数据: {pkl_path}")
    print(f"    样本数: {data['X_train'].shape[0]}, 特征维度: {data['X_train'].shape[1]}")
    print(f"    特征: {data['feature_names']}")

    # 3. 构建 KAN 模型（initial grid 取自 model.grid；续训时若 resume 文件存在则按其记录 grid 重建）
    input_dim = len(cfg['data']['features'])
    hidden = cfg['model']['hidden_layers']
    width = [input_dim] + hidden + [1]
    initial_grid = int(cfg['model'].get('grid', 5))
    resume_path = output_path(variant, 'models', f'kan_resume_{variant}.pth')
    build_grid = initial_grid
    if bool(cfg['train'].get('resume', False)) and os.path.exists(resume_path):
        try:
            _ck = torch.load(resume_path, map_location='cpu', weights_only=False)
            build_grid = int(_ck.get('current_grid', initial_grid))
        except Exception:
            pass
    model = KAN(
        width=width,
        grid=build_grid,
        k=cfg['model']['k'],
        seed=cfg['model']['seed'],
        save_act=True, auto_save=False,
    )
    device = data['device']
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] 构建 KAN 模型: width={width} (grid={build_grid}, k={cfg['model']['k']})")
    print(f"    参数量: {n_params:,}, 设备: {device}")

    # 3b. 微调：从预训练权重初始化（区别于 resume 断点续训）
    # init_from 指向预训练变体名或 .pth 路径；直接复用其权重与（推断出的）真实 grid。
    ft = cfg.get('finetune', {}) or {}
    init_from = ft.get('init_from')
    if init_from:
        print(f"\n[2b] 从预训练权重初始化 (init_from={init_from}) ...")
        pre_model, pre_cfg, pre_grid, pre_path = load_pretrained_model(init_from, device)
        model = pre_model  # 已含正确 grid + 权重
        build_grid = pre_grid
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    已载入预训练模型: {pre_path} (grid={pre_grid}, 参数量 {n_params:,})")

    # 3c. 冻结策略（finetune.freeze 为参数名子串列表；匹配的层 requires_grad=False）
    freeze = ft.get('freeze', []) or []
    if freeze:
        frozen = 0
        for name, p in model.named_parameters():
            if any(fr in name for fr in freeze):
                p.requires_grad = False
                frozen += 1
        print(f"    冻结参数组: {frozen}（匹配子串 {freeze}）；可训练参数: "
              f"{sum(1 for p in model.parameters() if p.requires_grad):,}")

    # 4–6. 优化器 / 调度器 / 损失函数
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    criterion = get_criterion(cfg)
    print(f"[3] 优化器: {type(optimizer).__name__}, 调度器: "
          f"{type(scheduler).__name__ if scheduler is not None else 'None'}, "
          f"损失: {type(criterion).__name__}")

    # 7. 转为 tensor，送 device
    X = torch.tensor(data['X_train'], dtype=torch.float32).to(device)
    y = torch.tensor(data['y_train'], dtype=torch.float32).to(device).reshape(-1, 1)
    train_ds = TensorDataset(X, y)
    batch_size = int(cfg['train']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 验证集（held-out）：用于 val_loss 早停监控（finetune 时由 01 划分产生）
    X_val_t = y_val_t = None
    if 'X_val' in data and data.get('X_val') is not None:
        X_val_t = torch.tensor(data['X_val'], dtype=torch.float32).to(device)
        y_val_t = torch.tensor(data['y_val'], dtype=torch.float32).to(device).reshape(-1, 1)
        print(f"    验证集: {X_val_t.shape[0]} 样本（held-out，用于 val_loss 监控）")

    # 8. 训练超参
    epochs = int(cfg['train']['epochs'])
    grad_clip = float(cfg['train'].get('gradient_clip', 0.0))
    es = cfg['train']['early_stopping']
    patience = int(es['patience'])
    min_delta = float(es['min_delta'])
    monitor = es.get('monitor', 'train_loss')
    scheduler_per_batch = False
    resume_enabled = bool(cfg['train'].get('resume', False))
    checkpoint_every = int(cfg['train'].get('checkpoint_every', 0))
    stop_on_best_below = cfg['train'].get('stop_on_best_below', None)
    stop_on_best_below = float(stop_on_best_below) if stop_on_best_below is not None else None

    # 渐进网格细化调度
    grid_schedule = cfg['train'].get('grid_schedule', []) or []
    grid_schedule = sorted([(int(g['epoch']), int(g['grid'])) for g in grid_schedule],
                           key=lambda x: x[0])

    # LBFGS 收尾配置
    lbfgs_cfg = cfg['train'].get('lbfgs', {}) or {}
    lbfgs_enabled = bool(lbfgs_cfg.get('enabled', False))
    lbfgs_start = int(lbfgs_cfg.get('start_epoch', 10 ** 9))
    lbfgs_steps_total = int(lbfgs_cfg.get('steps', 300))

    # 路径
    best_path = output_path(variant, 'models', f'kan_best_{variant}.pth')
    final_path = output_path(variant, 'models', f'kan_final_{variant}.pth')
    latest_path = output_path(variant, 'models', f'kan_latest_{variant}.pth')
    resume_path = output_path(variant, 'models', f'kan_resume_{variant}.pth')

    # 注册中断信号
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 续训载入
    (start_epoch, best_loss, best_epoch, patience_counter, history, resumed,
     current_grid, grid_ptr, phase, lbfgs_step) = _maybe_resume(
        variant, model, optimizer, scheduler, cfg, initial_grid)
    best_phase = phase if resumed else 'adam'
    if resumed:
        print(f"    续训将从 epoch {start_epoch + 1} 继续（phase={phase}, grid={current_grid}）。")
    else:
        print(f"\n[4] 开始训练（共 {epochs} 轮，batch={batch_size}，"
              f"早停 patience={patience}, min_delta={min_delta}, monitor={monitor}）"
              + (f"；续训关闭" if not resume_enabled else "")
              + (f"；网格细化 {grid_schedule}" if grid_schedule else "")
              + (f"；LBFGS 收尾 steps={lbfgs_steps_total}" if lbfgs_enabled else ""))

    model.train()
    start = time.time()
    interrupted = False
    stopped_by_threshold = False
    run_adam = not (resumed and phase == 'lbfgs')

    try:
        # ---------- Adam 阶段（含渐进网格细化）----------
        if run_adam:
            for epoch in range(start_epoch, epochs):
                epoch_loss = 0.0
                n_batches = 0
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    _step_scheduler(scheduler, optimizer, scheduler_per_batch, n_batches)
                    epoch_loss += loss.item()
                    n_batches += 1
                avg_loss = epoch_loss / n_batches
                history['train_loss'].append(avg_loss)
                history['lr'].append(optimizer.param_groups[0]['lr'])

                # 调度器按 epoch step
                if scheduler is not None and not scheduler_per_batch:
                    if type(scheduler).__name__ == 'ReduceLROnPlateau':
                        scheduler.step(avg_loss)
                    else:
                        scheduler.step()

                # 早停监控量：val_loss（held-out，优先）或 train_loss
                monitor_loss = avg_loss
                if monitor == 'val_loss' and X_val_t is not None:
                    model.eval()
                    with torch.no_grad():
                        vpred = model(X_val_t)
                        val_loss = criterion(vpred, y_val_t).item()
                    model.train()
                    monitor_loss = val_loss
                    history.setdefault('val_loss', []).append(val_loss)
                elif monitor == 'val_loss' and X_val_t is None:
                    print("  ⚠️ monitor=val_loss 但 pkl 无 X_val，回退为 train_loss 监控")
                    monitor = 'train_loss'

                if monitor_loss < best_loss - min_delta:
                    best_loss = monitor_loss
                    best_epoch = epoch + 1
                    best_phase = 'adam'
                    patience_counter = 0
                    _save_checkpoint(model, cfg, variant, best_epoch, best_loss,
                                     best_path, extra={'history': history})
                else:
                    patience_counter += 1

                if (epoch + 1) % 20 == 0 or epoch == start_epoch or epoch + 1 == epochs:
                    val_str = (f" | val_loss: {val_loss:.4e}"
                               if (monitor == 'val_loss' and X_val_t is not None) else "")
                    print(f"    Epoch {epoch + 1:4d}/{epochs} | loss: {avg_loss:.4e}{val_str} | "
                          f"lr: {optimizer.param_groups[0]['lr']:.2e} | grid: {current_grid}"
                          + (f" | 早停计数: {patience_counter}/{patience}"
                             if monitor in ('train_loss', 'val_loss') else ""))

                # 渐进网格细化：到达调度点时 refine 并重置 lr 调度器
                while grid_ptr < len(grid_schedule) and (epoch + 1) >= grid_schedule[grid_ptr][0]:
                    new_grid = grid_schedule[grid_ptr][1]
                    if new_grid != current_grid:
                        print(f"\n  [网格细化] epoch {epoch + 1}: grid {current_grid} -> {new_grid}"
                              f"（重置 lr 调度器，早停计数清零）")
                        model, optimizer, scheduler = _refine_and_rebuild(model, new_grid, cfg, device, X)
                        current_grid = new_grid
                        patience_counter = 0  # 细化后 loss 跳升再下降，重置早停
                    grid_ptr += 1

                # 周期保存 latest + resume
                if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                    _save_checkpoint(model, cfg, variant, epoch + 1, avg_loss, latest_path,
                                     extra={'history': history})
                    _save_resume_checkpoint(model, optimizer, scheduler, cfg, variant, epoch + 1,
                                            best_loss, best_epoch, patience_counter, history,
                                            resume_path, interrupted=False,
                                            current_grid=current_grid, grid_ptr=grid_ptr,
                                            phase='adam', lbfgs_step=0)

                # 达标提前停
                if stop_on_best_below is not None and best_loss <= stop_on_best_below:
                    print(f"\n  ✓ 达到 stop_on_best_below 阈值 ({stop_on_best_below:.4e})，"
                          f"best_loss={best_loss:.4e}，提前结束。")
                    stopped_by_threshold = True
                    break

                # 早停触发
                if monitor in ('train_loss', 'val_loss') and patience_counter >= patience:
                    print(f"\n  ⏹ 早停触发：连续 {patience} 轮损失未改善（最佳 @ epoch {best_epoch}）")
                    break

                # Adam -> LBFGS 切换点
                if lbfgs_enabled and (epoch + 1) >= lbfgs_start:
                    print(f"\n  [阶段切换] epoch {epoch + 1} 到达 lbfgs.start_epoch={lbfgs_start}，"
                          f"准备进入 LBFGS 收尾。")
                    break

                # 用户中断
                if _INTERRUPTED:
                    interrupted = True
                    print(f"\n  ⏸ 收到中断，当前 epoch {epoch + 1} 已完成，准备保存后退出。")
                    break

        # ---------- LBFGS 收尾阶段 ----------
        if lbfgs_enabled and not interrupted and phase == 'adam':
            phase = 'lbfgs'
            print(f"\n[5] 进入 LBFGS 收尾阶段（grid={current_grid}，steps={lbfgs_steps_total}）...")
            best_loss, best_epoch, best_phase, lbfgs_step, avg_loss = _run_lbfgs(
                model, train_loader, criterion, cfg, variant, best_path, history,
                best_loss, best_epoch, best_phase, min_delta, lbfgs_cfg,
                0, current_grid, grid_ptr, patience_counter,
                checkpoint_every, resume_path)
        elif resumed and phase == 'lbfgs':
            # 续训直接进入 LBFGS（从断点 step 继续）
            phase = 'lbfgs'
            print(f"\n[5] 续训进入 LBFGS 收尾阶段（grid={current_grid}，从 step {lbfgs_step} 继续）...")
            best_loss, best_epoch, best_phase, lbfgs_step, avg_loss = _run_lbfgs(
                model, train_loader, criterion, cfg, variant, best_path, history,
                best_loss, best_epoch, best_phase, min_delta, lbfgs_cfg,
                lbfgs_step, current_grid, grid_ptr, patience_counter,
                checkpoint_every, resume_path)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n  ⏸ 捕获 KeyboardInterrupt，准备保存后退出。")

    # 9–10. 保存 latest / resume / final 检查点
    final_epoch_disp = epoch + 1 if 'epoch' in dir() else '?'
    try:
        final_epoch_disp = epoch + 1
    except NameError:
        final_epoch_disp = (f'lbfgs{lbfgs_step}' if phase == 'lbfgs' else '?')
    _save_checkpoint(model, cfg, variant, final_epoch_disp, avg_loss, latest_path,
                     extra={'history': history, 'phase': phase})
    _save_resume_checkpoint(model, optimizer, scheduler, cfg, variant, final_epoch_disp,
                            best_loss, best_epoch, patience_counter, history,
                            resume_path, interrupted=interrupted,
                            current_grid=current_grid, grid_ptr=grid_ptr,
                            phase=phase, lbfgs_step=lbfgs_step)
    if best_epoch == 0:
        best_epoch = final_epoch_disp
        best_loss = avg_loss
        _save_checkpoint(model, cfg, variant, best_epoch, best_loss, best_path,
                         extra={'history': history, 'phase': phase})
    _save_checkpoint(model, cfg, variant, final_epoch_disp, avg_loss, final_path,
                     extra={'history': history,
                            'early_stopped': (monitor == 'train_loss' and patience_counter >= patience) or stopped_by_threshold,
                            'interrupted': interrupted,
                            'phase': phase})

    # 11. 摘要
    total_time = time.time() - start
    print(f"\n    训练完成：最佳损失 {best_loss:.4e}"
          + (f"（{best_phase} epoch/step {best_epoch}）" if best_phase else "")
          + (f"（中断退出）" if interrupted else "")
          + (f"（阈值达标）" if stopped_by_threshold else "")
          + f"，用时 {total_time:.1f}s")
    print("\n" + "=" * 60)
    print("训练完成！摘要信息")
    print("=" * 60)
    print(f"变体名    : {variant}")
    print(f"最终 grid : {current_grid}")
    print(f"最终 loss  : {avg_loss:.4e}")
    print(f"最佳 loss  : {best_loss:.4e} ({best_phase} epoch/step {best_epoch})")
    print(f"best 模型  : {best_path}")
    print(f"latest 模型: {latest_path}")
    print(f"resume 点 : {resume_path}" + ("（可设 train.resume=true 续训）" if not resumed else ""))
    print(f"final 模型 : {final_path}")
    print("=" * 60)


def _step_scheduler(scheduler, optimizer, per_batch, step_count):
    """根据调度器类型决定 step 时机：ReduceLROnPlateau 需按 epoch 在外部 step，
       其余（Cosine/Step/Exponential）按配置 per_batch 决定每个 batch 或每个 epoch 调用。"""
    if scheduler is None:
        return
    t = type(scheduler).__name__
    if t == 'ReduceLROnPlateau':
        return  # 在 epoch 末统一 step（传入当前 loss）
    if per_batch:
        scheduler.step()
    # 非 per_batch 的情况在 epoch 末统一 step


if __name__ == '__main__':
    main()
