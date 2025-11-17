#!/usr/bin/env python3
"""
简单剪枝脚本：对训练好的 UNetLightningModule (SegFormer/MobileMamba) ckpt 文件执行结构化剪枝。

- 支持：
    - PyTorch 结构化剪枝（按输出通道/输出行, 使用 prune.ln_structured）
    - Torch-Pruning 自动化通道剪枝（依赖第三方库 torch_pruning，可自动分析依赖并重构模型）
- 剪枝比例、输入/输出路径等参数直接在脚本内硬编码。
- 剪枝后保存 state_dict 到新文件。
"""
import os
import torch
import torch.nn.utils.prune as prune
from UNet import UNetLightningModule

# ====== 配置区 ======
CKPT_PATH = '/home/cbm/deRFI/checkpoints/best_model-epoch=21-fgF1_val_fg_macro_f1=0.7905.ckpt'  # 输入ckpt
PRUNED_PATH = os.path.join('checkpoints', 'prune', 'pruned_best_model-epoch=21-fgF1_val_fg_macro_f1=0.7905.pt')  # 剪枝后输出
PRUNE_AMOUNT = 0.3  # 全局剪枝比例（0.3=剪掉30%最小权重）
PRUNE_TYPE = 'structured'  # 'structured' or 'torch_pruning'
# ====================

def load_model(ckpt_path):
    model = UNetLightningModule.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.cpu()
    return model

# NOTE: 非结构化剪枝（按元素置零）已移除，若需要此功能请使用版本历史或手动添加


def apply_structured_pruning(model, amount=0.3):
    """按输出通道（Conv2d dim=0）和输出行（Linear dim=0）进行结构化 L1 剪枝。

    注意：此函数仅将整通道/整行的权重置 0，并不会物理上删除通道（要获得真实加速需额外重构模型）。
    """
    # 记录成功剪枝的层与统计
    pruned_layers = []
    # 定义关键关键模块关键词，匹配到则跳过剪枝（避免影响 classifier / decode head）
    CRITICAL_KEYWORDS = ["classifier", "decode_head", "decoder", "linear_fuse", "linear_c"]

    for name, module in model.named_modules():
        # 如果模块路径中包含关键关键词则跳过
        if any(k in name.lower() for k in CRITICAL_KEYWORDS):
            print(f"[StructuredPrune] 跳过关键模块: {name}")
            continue
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # 打印尝试修剪信息
            print(f"[StructuredPrune] 尝试剪枝: {name}: {module.__class__.__name__}")
            try:
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            except Exception as e:
                print(f"  跳过 (异常): {name}: {e}")
                continue
    # (注) 已通过上面的条件覆盖 Conv2d 与 Linear

    # 移除 re-param（将 mask 合并到权重），这样保存的 state_dict 更易移植
    # 统计被剪掉的通道数（如果存在 mask）
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if any(k in name.lower() for k in CRITICAL_KEYWORDS):
                # 再次保护 — 移除 mask 前不要对关键层做任何操作
                print(f"[StructuredPrune] 跳过关键模块（移除mask阶段）: {name}")
                continue
            mask = getattr(module, 'weight_mask', None)
            if mask is not None:
                # mask 形状同 weight; 对 Conv2d mask shape (out_ch, in_ch, k, k)
                # 对 Linear mask shape (out_features, in_features)
                # 统计哪些输出通道被全部置零
                try:
                    chwise = mask.view(mask.shape[0], -1).sum(dim=1)
                    pruned_idx = (chwise == 0).nonzero(as_tuple=False).view(-1).tolist()
                    if len(pruned_idx) > 0:
                        pruned_layers.append((name, module.__class__.__name__, len(pruned_idx), module.weight.shape[0]))
                        print(f"  剪枝结果: {name}: 剪掉 {len(pruned_idx)}/{module.weight.shape[0]} 输出通道")
                    else:
                        print(f"  剪枝结果: {name}: 未剪掉输出通道 (0/{module.weight.shape[0]})")
                except Exception as e:
                    print(f"  统计失败: {name}: {e}")
            # 移除 mask，将剪枝结果合并到权重
            if hasattr(module, 'weight') and hasattr(module.weight, 'mask'):
                prune.remove(module, 'weight')
    return model


def apply_torch_pruning(model, amount=0.3, example_shape=(1,1,512,512), iterative_steps=1, global_pruning=True):
    """使用 torch_pruning 自动分析并执行结构化剪枝（channel pruning）。

    - 需要额外依赖：torch_pruning。请使用 `pip install torch-pruning` 安装。
    - 该方法会尝试自动按重要性（默认 L1 magnitude）判定每层要剪掉的通道并执行 prune。
    - 默认只做一次步骤（iterative_steps=1）。若想渐进式剪枝可增加 steps。
    """
    try:
        import torch_pruning as tp
    except Exception as e:
        raise RuntimeError("torch_pruning 未安装，请运行: pip install torch-pruning") from e

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else 'cpu'
    example_inputs = torch.randn(*example_shape, device='cpu')

    # 重要性度量：Magnitude (L1)
    imp = tp.importance.MagnitudeImportance(p=1)

    # 忽略很小的输出层或分类器（如果检测到）
    ignored_layers = []
    for name, m in model.named_modules():
        # 忽略输出分类器(小channel数)
        if isinstance(m, torch.nn.Linear) and getattr(m, 'out_features', 0) <= 4:
            ignored_layers.append(m)
        if isinstance(m, torch.nn.Conv2d) and getattr(m, 'out_channels', 0) <= 4:
            ignored_layers.append(m)
        # 保护 classifier / decoder / decode_head
        if any(k in name.lower() for k in ("classifier", "decode_head", "decoder", "linear_fuse", "linear_c")):
            print(f"[TP] 忽略关键模块 for tp pruning: {name}")
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=amount,
        ignored_layers=ignored_layers,
        global_pruning=global_pruning,
    )

    # 打印剪枝前计算量
    try:
        _count_res = tp.utils.count_ops_and_params(model, example_inputs)
        base_macs, base_nparams = _count_res[0], _count_res[1]
        print(f"[TP] Before: MACs={base_macs:.2f}, Params={base_nparams:.2f}")
        try:
            tp.utils.print_tool.before_pruning(model)
        except Exception:
            pass
    except Exception:
        pass

    # 执行一次（或多次）剪枝
    # 如果 iterable 返回 groups, 需要逐个执行g.prune()
    groups = pruner.step(interactive=True)
    if groups is not None:
        for g in groups:
            g.prune()
    else:
        # fallback to non-interactive step (may apply immediately)
        pruner.step()

    try:
        _count_res = tp.utils.count_ops_and_params(model, example_inputs)
        pruned_macs, pruned_nparams = _count_res[0], _count_res[1]
        print(f"[TP] After:  MACs={pruned_macs:.2f}, Params={pruned_nparams:.2f}")
    except Exception:
        pass

    # 打印 after_pruning 信息
    try:
        tp.utils.print_tool.after_pruning(model, do_print=True)
    except Exception:
        pass

    return model

def save_pruned_state_dict(model, path):
    # 确保输出目录存在
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Save] 剪枝后模型 state_dict 已保存: {path}")


def count_sparsity(model):
    total = 0
    zero = 0
    layer_stats = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        numel = p.numel()
        total += numel
        z = torch.sum(p == 0).item()
        zero += z
        layer_stats.append((name, numel, z, z / numel if numel else 0.0))
    sparsity = zero / total if total else 0.0
    return sparsity, layer_stats

def main():
    print(f"[Info] 加载模型: {CKPT_PATH}")
    model = load_model(CKPT_PATH)
    print(f"[Info] 执行{PRUNE_TYPE}剪枝, 剪枝比例: {PRUNE_AMOUNT}")
    # 打印剪枝前稀疏度
    before_s, before_stats = count_sparsity(model)
    print(f"[Before] 全局稀疏度: {before_s*100:.2f}%")
    # 输出前几层统计
    for name, numel, z, ratio in before_stats[:8]:
        print(f"  {name}: {z}/{numel} ({ratio*100:.2f}%)")
    if PRUNE_TYPE == 'structured':
        model = apply_structured_pruning(model, amount=PRUNE_AMOUNT)
    elif PRUNE_TYPE == 'torch_pruning':
        # 需要 torch_pruning 库
        model = apply_torch_pruning(model, amount=PRUNE_AMOUNT)
    else:
        raise ValueError(f"未知的 PRUNE_TYPE: {PRUNE_TYPE}")
    save_pruned_state_dict(model, PRUNED_PATH)
    # 保存后输出稀疏统计
    after_s, after_stats = count_sparsity(model)
    print(f"[After] 全局稀疏度: {after_s*100:.2f}%")
    for name, numel, z, ratio in after_stats[:8]:
        print(f"  {name}: {z}/{numel} ({ratio*100:.2f}%)")
    print("[Done] 剪枝完成。")

if __name__ == '__main__':
    main()
