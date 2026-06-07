#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as albu
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import segmentation_models_pytorch as smp

# 导入现有的 Dataset 逻辑 (确保从 UNet 导入，因为用户指定要按照 UNet.py 的逻辑)
from SegFormer_StrategyAltered import FITSDataset, get_validation_augmentation, get_preprocessing

def get_validation_augmentation(height=640, width=640):
    """Validation augmentation: Use Reflect padding to avoid edge artifacts being misclassified."""
    test_transform = []
    if height is not None and width is not None:
        test_transform.append(albu.PadIfNeeded(
            min_height=height, 
            min_width=width, 
            border_mode=cv2.BORDER_REFLECT_101
        ))
    return albu.Compose(test_transform)

class TRTInference:
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 预先获取绑定名称
        try:
            num_io = self.engine.num_io_tensors
            self.tensor_names = [self.engine.get_tensor_name(i) for i in range(num_io)]
        except AttributeError:
            self.tensor_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        import tensorrt as trt
        import pycuda.driver as cuda
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        # 兼容不同版本的 TensorRT API
        if hasattr(self.engine, 'num_io_tensors'):
            for name in self.tensor_names:
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                # 处理动态 Batch
                if shape[0] == -1: shape[0] = 4 # 默认设为 4
                
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                
                # 设置 Tensor 地址 (TRT 10 核心步骤)
                self.context.set_tensor_address(name, int(device_mem))
                
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
        else:
            for i in range(self.engine.num_bindings):
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                if self.engine.binding_is_input(i):
                    inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
        return inputs, outputs, bindings, stream

    def infer(self, img_batch):
        import pycuda.driver as cuda
        # 拷贝数据到 Input Buffer (Host -> Device)
        host_input = self.inputs[0]['host']
        np.copyto(host_input, img_batch.astype(host_input.dtype).ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], host_input, self.stream)
        
        # 执行推理
        if hasattr(self.context, "execute_v2"):
            self.context.execute_v2(bindings=self.bindings)
        elif hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute(batch_size=img_batch.shape[0], bindings=self.bindings)

        # 拷贝结果 (Device -> Host)
        host_output = self.outputs[0]['host']
        cuda.memcpy_dtoh_async(host_output, self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return host_output

def calculate_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    metrics = []
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append((iou, precision, recall, f1))
    return metrics

def save_debug_images(img_np, pred_np, mask_np, count, output_dir="temp/val_debug"):
    """
    保存可视化对比图，检查预处理和推理是否正确。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色映射 (B, G, R)
    colors = [
        [0, 0, 0],       # Background: Black
        [255, 0, 0],     # Horizontal: Blue
        [0, 255, 0],     # Vertical: Green
        [0, 0, 255],     # Point: Red
        [255, 255, 0],   # Block: Yellow/Cyan
        [255, 0, 255]    # Pulsar: Purple
    ]
    
    # 1. 还原输入图 (img_np 是 [1, H, W] 且在 [0,1] 之间)
    img_viz = (img_np[0] * 255).astype(np.uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2BGR)
    
    # 2. 绘制 GT Mask
    gt_viz = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        gt_viz[mask_np == i] = color
        
    # 3. 绘制 Pred Mask
    pred_viz = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        pred_viz[pred_np == i] = color
        
    # 拼合 [Image | GT | Pred]
    combined = np.hstack([img_viz, gt_viz, pred_viz])
    
    # 添加文字标注
    cv2.putText(combined, "Input (Normalized)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(combined, "Ground Truth", (img_viz.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(combined, "Prediction", (img_viz.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    save_path = os.path.join(output_dir, f"val_{count:04d}.png")
    cv2.imwrite(save_path, combined)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine')
    parser.add_argument('--val_dir', type=str, default='Datasets/SynthesizedDataset/image/val', help='Path to validation images')
    parser.add_argument('--mask_dir', type=str, default='Datasets/SynthesizedDataset/mask/val', help='Path to validation masks')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # 初始化 TRT 获取 Engine 尺寸
    model = TRTInference(args.engine)
    
    # 兼容 TRT 10.x 接口获取输入尺寸
    if hasattr(model.engine, 'get_tensor_shape'):
        input_name = model.tensor_names[0]
        input_shape = list(model.engine.get_tensor_shape(input_name))
        # 处理动态 Batch
        if input_shape[0] == -1: input_shape[0] = args.batch_size
        engine_b, engine_c, engine_h, engine_w = input_shape
    else:
        input_shape = list(model.engine.get_binding_shape(0))
        engine_b, engine_c, engine_h, engine_w = input_shape

    # 初始化 Dataset
    dataset = FITSDataset(
        image_dir=args.val_dir, 
        mask_dir=args.mask_dir, 
        crop_size=engine_h, 
        point_oversample_factor=1.0, 
        point_crop_prob=0.0,
        augmentation=get_validation_augmentation(engine_h, engine_w),
        preprocessing=get_preprocessing(None),
        normalization_method="median_sigma"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if engine_b != args.batch_size:
        print(f"[警告] Engine batch size ({engine_b}) 与命令行参数 ({args.batch_size}) 不匹配。")

    num_classes = 6
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_names = ['Background', 'Horizontal', 'Vertical', 'Point', 'Block', 'Pulsar']

    # 新增：复刻训练时的 smp 指标累加器
    all_tp = []
    all_fp = []
    all_fn = []
    all_tn = []

    num_samples_to_validate = 10000
    print(f"🚀 Starting validation on {num_samples_to_validate} samples (subset of {len(dataset)}) using Engine shape: {input_shape}")

    count = 0
    debug_dir = "temp/val_debug"
    print(f"📸 Debug images will be saved to: {debug_dir}")
    
    for images, masks in tqdm(loader, total=num_samples_to_validate // args.batch_size):
        if count >= num_samples_to_validate:
            break
        
        img_np = images.numpy() if isinstance(images, torch.Tensor) else images
        
        # 补全 Batch 维度以匹配 Engine
        current_b = img_np.shape[0]
        if current_b < engine_b:
            pad = np.zeros((engine_b - current_b, *img_np.shape[1:]), dtype=img_np.dtype)
            img_input = np.concatenate([img_np, pad], axis=0)
        else:
            img_input = img_np
            
        # TRT 推理
        output = model.infer(img_input)
        
        # 将输出 reshape 并恢复成 logits [B, C, H, W]
        logits = output.reshape(engine_b, num_classes, engine_h, engine_w)
        logits_pt = torch.from_numpy(logits).float()
        
        # 获取预测
        pred_pt = torch.argmax(logits_pt, dim=1).long()
        target_pt = masks.long()

        # --- 完全复刻 UNet.py 中的 smp.metrics 逻辑 ---
        tp_smp, fp_smp, fn_smp, tn_smp = smp.metrics.get_stats(
            pred_pt[:current_b], 
            target_pt[:current_b], 
            mode='multiclass', 
            num_classes=num_classes
        )
        all_tp.append(tp_smp.sum(dim=0).long()) # (C,)
        all_fp.append(fp_smp.sum(dim=0).long())
        all_fn.append(fn_smp.sum(dim=0).long())
        all_tn.append(tn_smp.sum(dim=0).long())
        # --------------------------------------------

        # 原有的混淆矩阵统计 (供对比)
        pred = pred_pt.numpy()
        target = target_pt.numpy()
        
        # 每隔 50 个样本保存一张调试图
        if count % 50 == 0:
            save_debug_images(img_np[0], pred[0], target[0], count, debug_dir)

        # 更新混淆矩阵
        for b in range(current_b):
            p = pred[b].flatten()
            t = target[b].flatten()
            # 过滤掉标签 255 (忽略位)
            keep = (t >= 0) & (t < num_classes)
            bin_counts = np.bincount(t[keep].astype(int) * num_classes + p[keep].astype(int), minlength=num_classes**2)
            total_conf_matrix += bin_counts.reshape(num_classes, num_classes)
        
        count += current_b

    # 计算并显示指标
    print("\n" + "="*72)
    print(f"{'Class':<15} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("="*72)
    
    results = calculate_metrics(total_conf_matrix)
    for i, name in enumerate(class_names):
        iou, prec, rec, f1 = results[i]
        print(f"{name:<15} {iou:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    print("="*72)

    # 计算复刻指标
    print("\n" + "🔍" + " " + "="*30 + " SMP REPLICATED METRICS " + "="*30)
    final_tp = torch.stack(all_tp).sum(dim=0)
    final_fp = torch.stack(all_fp).sum(dim=0)
    final_fn = torch.stack(all_fn).sum(dim=0)
    final_tn = torch.stack(all_tn).sum(dim=0)

    replicated_iou = smp.metrics.iou_score(final_tp.long(), final_fp.long(), final_fn.long(), final_tn.long(), reduction="none")
    replicated_f1  = smp.metrics.f1_score(final_tp.long(), final_fp.long(), final_fn.long(), final_tn.long(), reduction="none")
    
    # 按照 UNet.py 逻辑计算 FG Macro F1
    fg_macro_f1 = replicated_f1[1:].mean()
    
    print(f"{'Class':<15} {'SMP-IoU':<12} {'SMP-F1':<12}")
    for i, name in enumerate(class_names):
        print(f"{name:<15} {replicated_iou[i]:<12.4f} {replicated_f1[i]:<12.4f}")
    
    print("-" * 72)
    print(f"VAL FG MACRO F1 (REPLICATED): {fg_macro_f1:.4f}")
    print("=" * 72)

    # ...原有打印混淆矩阵指标的代码...

if __name__ == "__main__":
    main()
