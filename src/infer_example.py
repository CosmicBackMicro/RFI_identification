#!/usr/bin/env python3
"""
统一推理脚本（简化版）：
- PyTorch 路线：--checkpoint 指定 Lightning .ckpt，自动用 GPU（若可用）或 CPU 推理
- ONNX 路线：--onnx 指定 .onnx，优先 TensorRT，其次 CUDA，最后 CPU Provider

示例：
    PyTorch: python src/infer_example.py --checkpoint <ckpt> --dataset <dataset_dir>
    ONNX  : python src/infer_example.py --onnx <model.onnx> --dataset <dataset_dir>

若同时给出 --checkpoint 与 --onnx，优先使用 --onnx。
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端，避免线程问题
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(__file__))

from UNet import UNetLightningModule, FITSDataset

try:
    import onnxruntime as ort  # 可选，仅在使用 --onnx 时需要
except Exception:
    ort = None

def _pick_ort_providers():
    """按优先顺序返回 ORT Providers 列表。"""
    providers = []
    if ort is None:
        return providers
    try:
        avail = ort.get_available_providers()
        if 'TensorrtExecutionProvider' in avail:
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in avail:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
    except Exception:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return providers

def _get_class_names(dataset, num_classes_hint=None):
    try:
        names = list(getattr(dataset, 'classes', []))
    except Exception:
        names = []
    if num_classes_hint is None:
        return names if names else FITSDataset.CLASSES
    # 对齐长度到提示的类数
    if not names:
        names = FITSDataset.CLASSES
    if len(names) < int(num_classes_hint):
        names = names + [f'class_{i}' for i in range(len(names), int(num_classes_hint))]
    return names[:int(num_classes_hint)]

def _save_sample_visual(i, image_np, mask_np, pred_np, class_names, save_dir, prefix=""):
    """通用可视化：1x4 视图，原图/GT/Pred/Overlay。"""
    os.makedirs(save_dir, exist_ok=True)
    nc = len(class_names)
    mask_vmax = max(nc - 1, int(np.max(mask_np)) if getattr(mask_np, 'size', 0) > 0 else 0)
    pred_vmax = max(nc - 1, int(np.max(pred_np)) if getattr(pred_np, 'size', 0) > 0 else 0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_np, cmap='gist_heat'); axes[0].set_title('Image'); axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='viridis', vmin=0, vmax=mask_vmax); axes[1].set_title('GT'); axes[1].axis('off')
    axes[2].imshow(pred_np, cmap='viridis', vmin=0, vmax=pred_vmax); axes[2].set_title('Pred'); axes[2].axis('off')
    axes[3].imshow(image_np, cmap='gist_heat'); axes[3].imshow(pred_np, cmap='viridis', alpha=0.5, vmin=0, vmax=pred_vmax); axes[3].set_title('Overlay'); axes[3].axis('off')
    fname = f"{prefix}inference_sample_{i}.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    return path


def run_pytorch_inference(model_path, dataset, num_samples=3, save_dir="results/pytorch", batch_size=4):
    """
    可视化推理结果：显示原始图像、真实掩码、预测掩码和叠加结果。
    
    Args:
        model_path (str): 模型 checkpoint 路径
        dataset (Dataset): 验证数据集
        num_samples (int): 要可视化的样本数量
        save_dir (str): 保存图像的目录
        batch_size (int): 批处理大小，用于并行推理
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    total_start_time = time.time()  # 记录程序总开始时间
    
    # 加载模型
    model = UNetLightningModule.load_from_checkpoint(model_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = next(model.parameters()).device
    # 显示当前推理设备信息（CPU/GPU）
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            dev_index = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_index)
            major, minor = torch.cuda.get_device_capability(dev_index)
            print(f"🖥️ 推理设备: GPU - {dev_name} (cc {major}.{minor})")
        except Exception:
            print("🖥️ 推理设备: GPU")
    else:
        print("🖥️ 推理设备: CPU")
    # 类别名称（根据模型 num_classes 对齐）
    num_classes = int(getattr(model, 'num_classes', 1))
    dataset_class_names = _get_class_names(dataset, num_classes)
    
    total_inference_time = 0.0  # 总推理时间
    
    # 批处理推理循环
    for start_idx in range(0, min(num_samples, len(dataset)), batch_size):
            end_idx = min(start_idx + batch_size, min(num_samples, len(dataset)))
            batch_images = []
            batch_masks = []
            
            # 收集批次数据
            for i in range(start_idx, end_idx):
                image, mask = dataset[i]
                batch_images.append(torch.tensor(image, dtype=torch.float32).unsqueeze(0))
                batch_masks.append(mask)
            
            # 堆叠成批次 tensor
            batch_images_tensor = torch.stack(batch_images).to(device)  # (batch_size, 1, H, W)
            
            # 并行推理并计时（只计推理时间）
            start_time = time.time()
            with torch.no_grad():
                batch_outputs = model(batch_images_tensor)
                # 兼容模型 forward 返回 (logits, probs) 的新结构；若只返回 logits 则自行 softmax
                if isinstance(batch_outputs, tuple):
                    # 期望格式: (logits, probs)
                    if len(batch_outputs) == 2:
                        logits, probs = batch_outputs
                    else:
                        # 多元素时取第一个为 logits
                        logits = batch_outputs[0]
                        probs = torch.softmax(logits, dim=1)
                elif isinstance(batch_outputs, dict):
                    logits = batch_outputs.get('logits')
                    if logits is None:
                        raise ValueError("字典输出缺少 'logits' 键，无法推理。")
                    probs = torch.softmax(logits, dim=1)
                else:
                    # 仅 logits
                    logits = batch_outputs
                    probs = torch.softmax(logits, dim=1)

                batch_preds = torch.argmax(probs, dim=1).cpu().numpy()  # (batch_size, H, W)
            end_time = time.time()
            batch_inference_time = end_time - start_time  # 批次推理时间
            total_inference_time += batch_inference_time
            
            # 计算每个样本平均推理时间
            num_samples_in_batch = len(batch_images)
            avg_sample_inference_time = batch_inference_time / num_samples_in_batch if num_samples_in_batch > 0 else 0.0
            
            print(f"📊 批次 {start_idx//batch_size + 1}: 推理 {num_samples_in_batch} 个样本，用时 {batch_inference_time:.4f}s (平均每样本 {avg_sample_inference_time:.4f}s)")
            
            # 收集批次样本数据并同步保存
            for idx_in_batch, i in enumerate(range(start_idx, end_idx)):
                image_np = batch_images[idx_in_batch].squeeze(0).cpu().numpy()
                mask_np = batch_masks[idx_in_batch]
                pred_np = batch_preds[idx_in_batch]
                try:
                    saved_path = _save_sample_visual(i+1, image_np, mask_np, pred_np, dataset_class_names, save_dir, prefix="pt_")
                    print(f"✅ 保存(Pytorch)结果: {saved_path}")
                except Exception as e:
                    print(f"❌ 保存失败: {e}")
    
    total_end_time = time.time()  # 记录程序总结束时间
    total_time = total_end_time - total_start_time  # 计算总运行时间
    
    print(f"🎉 PyTorch 推理完成！共处理 {min(num_samples, len(dataset))} 个样本，总推理时间: {total_inference_time:.4f}s，总运行时间: {total_time:.4f}s，结果保存至 {save_dir}/")

def run_onnx_inference(onnx_path, dataset, num_samples=3, save_dir="results/onnx", batch_size=4):
    """ONNX Runtime 推理与可视化（与 PyTorch 版本保持输出结构一致）。"""
    if ort is None:
        raise RuntimeError("onnxruntime 未安装，无法使用 --onnx。请先 pip install onnxruntime-gpu 或 onnxruntime。")
    os.makedirs(save_dir, exist_ok=True)
    total_start_time = time.time()
    providers = _pick_ort_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers) if providers else ort.InferenceSession(onnx_path)
    used = sess.get_providers()
    print(f"[ORT] Providers: {used}")

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    # 类别名称
    dataset_class_names = _get_class_names(dataset)
    num_classes = len(dataset_class_names)

    # 推理循环
    total_inference_time = 0.0
    for start_idx in range(0, min(num_samples, len(dataset)), batch_size):
        end_idx = min(start_idx + batch_size, min(num_samples, len(dataset)))
        batch_imgs = []
        batch_masks = []
        for i in range(start_idx, end_idx):
            image, mask = dataset[i]
            img_np = np.expand_dims(np.array(image, dtype=np.float32), axis=0)  # (1,H,W)
            batch_imgs.append(img_np)
            batch_masks.append(mask)
        x = np.stack(batch_imgs, axis=0)  # (B,1,H,W)
        feed = {input_name: x}
        start_time = time.time()
        outs = sess.run(output_names, feed)
        end_time = time.time()
        total_inference_time += (end_time - start_time)
        if 'probs' in output_names:
            probs = outs[output_names.index('probs')]
        elif 'logits' in output_names:
            logits = outs[output_names.index('logits')]
            # numpy softmax
            shift = logits - np.max(logits, axis=1, keepdims=True)
            expv = np.exp(shift)
            probs = expv / np.sum(expv, axis=1, keepdims=True)
        else:
            probs = outs[0]
        preds = np.argmax(probs, axis=1)
        for bi, i in enumerate(range(start_idx, end_idx)):
            img_np = batch_imgs[bi].squeeze(0)
            mask_np = batch_masks[bi]
            pred_np = preds[bi]
            try:
                saved_path = _save_sample_visual(i+1, img_np, mask_np, pred_np, dataset_class_names, save_dir, prefix="onnx_")
                print(f"✅ 保存(ONNX)结果: {saved_path}")
            except Exception as e:
                print(f"❌ 保存失败: {e}")
    total_time = time.time() - total_start_time
    print(f"🎉 ONNX 推理完成！共处理 {min(num_samples, len(dataset))} 个样本，总推理时间: {total_inference_time:.4f}s，总运行时间: {total_time:.4f}s，结果保存至 {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="统一推理脚本(Pytorch 或 ONNX)")
    parser.add_argument('--dataset', required=True, help='数据集顶层目录 (包含 image/mask)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Lightning .ckpt 路径')
    parser.add_argument('--onnx', type=str, default=None, help='ONNX 模型路径 (指定后优先使用)')
    parser.add_argument('--num-samples', type=int, default=32, help='推理样本数')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--save-dir', type=str, default='results', help='保存目录 (defaults to results/, or you can set a custom path that will be used for both modes)')
    args = parser.parse_args()

    dataset_top_dir = args.dataset
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=None,
        augmentation=None,
        preprocessing=None
    )
    print("🔍 开始推理可视化...")
    print(f"📂 数据集: {dataset_top_dir}")
    print(f"📊 样本数量: {len(val_dataset)}")

    if args.onnx:
        print(f"🧠 使用 ONNX 模型: {args.onnx}")
        run_onnx_inference(
            onnx_path=args.onnx,
            dataset=val_dataset,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            batch_size=args.batch_size
        )
    elif args.checkpoint:
        print(f"🧠 使用 PyTorch Lightning Checkpoint: {args.checkpoint}")
        run_pytorch_inference(
            model_path=args.checkpoint,
            dataset=val_dataset,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            batch_size=args.batch_size
        )
    else:
        raise SystemExit("必须提供 --checkpoint 或 --onnx 其中之一。")