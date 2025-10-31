#!/usr/bin/env python3
"""
推理示例：加载训练好的模型并可视化推理结果
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端，避免线程问题
import matplotlib.pyplot as plt
import time  # 添加时间模块
from concurrent.futures import ThreadPoolExecutor  # 添加并行模块
sys.path.append(os.path.dirname(__file__))

from UNet import UNetLightningModule, FITSDataset


def visualize_inference_results(model_path, dataset, num_samples=3, save_dir="inference_results", batch_size=4):
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
    
    total_inference_time = 0.0  # 总推理时间
    
    # 定义保存函数
    def save_visualization(sample_data):
        # 在子线程中确保matplotlib后端
        matplotlib.use('Agg')
        
        i, image_np, mask_np, pred_np, num_classes, save_dir = sample_data
        
        # 调试：检查数据
        # print(f"调试样本 {i}: image shape {image_np.shape}, max {np.max(image_np):.4f}, mask max {np.max(mask_np)}, pred max {np.max(pred_np)}")
        
        # 动态计算vmax，确保掩码值在范围内
        mask_vmax = max(num_classes - 1, np.max(mask_np))
        pred_vmax = max(num_classes - 1, np.max(pred_np))
        
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
        
        # 原始图像
        axes[0, 0].imshow(image_np, cmap='gist_heat')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 叠加：原始图像 + 预测掩码透明叠加 + 轮廓
        axes[0, 1].imshow(image_np, cmap='gist_heat')
        axes[0, 1].imshow(pred_np, cmap='viridis', alpha=0.5, vmin=0, vmax=pred_vmax)
        # 添加细轮廓线
        axes[0, 1].contour(pred_np, colors='white', linewidths=0.1, alpha=0.8)
        axes[0, 1].set_title('Overlay (Image + Pred Mask + Contour)')
        axes[0, 1].axis('off')
        
        # 真实掩码
        im1 = axes[0, 2].imshow(mask_np, cmap='viridis', vmin=0, vmax=mask_vmax)
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].axis('off')
        
        # 预测掩码
        im2 = axes[0, 3].imshow(pred_np, cmap='viridis', vmin=0, vmax=pred_vmax)
        axes[0, 3].set_title('Predicted Mask')
        axes[0, 3].axis('off')
        
        # 创建类别标签
        classes = ["bkg", "chan_rfi", "point_rfi"]  # 从FITSDataset.CLASSES获取
        
        # 在底部添加颜色条和类别标签
        # 为每个类别创建颜色条
        for j in range(num_classes):
            # 创建一个小的颜色条
            cbar_data = np.full((1, 100), j, dtype=int)
            im_cbar = axes[1, j].imshow(cbar_data, cmap='viridis', vmin=0, vmax=max(num_classes-1, mask_vmax), aspect='auto')
            axes[1, j].set_title(f'{classes[j]} (Class {j})', fontsize=10)
            axes[1, j].axis('off')
        
        # 隐藏多余的子图
        for j in range(num_classes, 4):
            axes[1, j].axis('off')
        
        # plt.tight_layout()  # 移除以避免布局警告
        plt.savefig(os.path.join(save_dir, f'inference_sample_{i}.png'), dpi=150, bbox_inches='tight')  # 降低DPI到150
        plt.close()
        
        return os.path.join(save_dir, f'inference_sample_{i}.png')
    
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
                batch_preds = torch.argmax(batch_outputs, dim=1).cpu().numpy()  # (batch_size, H, W)
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
                sample_data = (i+1, image_np, mask_np, pred_np, model.num_classes, save_dir)
                
                try:
                    saved_path = save_visualization(sample_data)
                    print(f"✅ 保存推理可视化结果: {saved_path}")
                except Exception as e:
                    print(f"❌ 保存失败: {str(e)}")
    
    total_end_time = time.time()  # 记录程序总结束时间
    total_time = total_end_time - total_start_time  # 计算总运行时间
    
    print(f"🎉 推理可视化完成！共处理 {min(num_samples, len(dataset))} 个样本，总推理时间: {total_inference_time:.4f}s，总运行时间: {total_time:.4f}s，结果保存至 {save_dir}/")


if __name__ == "__main__":
    # ============ 配置参数 ============
    # 数据集路径（根据你的数据集调整）
    # dataset_top_dir = "/home/cbm/deRFI/Datasets/Dataset_G200.48+2.54_5978_2classes_NoSubMed_ThreshCorrected"
    dataset_top_dir = "/home/cbm/deRFI/Datasets/Dataset_G184.63-5.93_6018_2classes"
    # 模型路径
    model_path = "/home/cbm/deRFI/checkpoints/best_model-epoch=66-val_iou=0.8297.ckpt"
    # 保存目录
    save_dir = "/home/cbm/deRFI/IntermediateResults"
    # =================================
    
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    # 创建验证数据集
    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=["bkg", "chan_rfi", "point_rfi"],  # 根据训练时的类别
        augmentation=None,  # 推理时不使用数据增强
        preprocessing=None
    )
    
    print("🔍 开始推理可视化...")
    print(f"📂 数据集: {dataset_top_dir}")
    print(f"🧠 模型: {model_path}")
    print(f"📊 样本数量: {len(val_dataset)}")
    
    # 可视化推理结果
    visualize_inference_results(
        model_path=model_path,
        dataset=val_dataset,
        num_samples=978,  # 可视化前50个样本
        save_dir=save_dir,
        batch_size=8  # 批处理大小，可根据GPU内存调整
    )