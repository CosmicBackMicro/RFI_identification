#!/usr/bin/env python3
"""
推理示例：加载训练好的模型并可视化推理结果
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))

from UNet import UNetLightningModule, FITSDataset


def visualize_inference_results(model_path, dataset, num_samples=3, save_dir="inference_results"):
    """
    可视化推理结果：显示原始图像、真实掩码、预测掩码和叠加结果。
    
    Args:
        model_path (str): 模型 checkpoint 路径
        dataset (Dataset): 验证数据集
        num_samples (int): 要可视化的样本数量
        save_dir (str): 保存图像的目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = UNetLightningModule.load_from_checkpoint(model_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = next(model.parameters()).device
    
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        
        # 转换为 tensor 并添加 batch 维度
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        # 转换为 numpy
        image_np = image  # 已经是 (H, W)
        mask_np = mask
        pred_np = preds
        
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [4, 1]})
        
        # 原始图像
        axes[0, 0].imshow(image_np, cmap='gist_heat')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 叠加：原始图像 + 预测掩码透明叠加 + 轮廓
        axes[0, 1].imshow(image_np, cmap='gist_heat')
        axes[0, 1].imshow(pred_np, cmap='viridis', alpha=0.5, vmin=0, vmax=model.num_classes-1)
        # 添加细轮廓线
        axes[0, 1].contour(pred_np, colors='white', linewidths=0.1, alpha=0.8)
        axes[0, 1].set_title('Overlay (Image + Pred Mask + Contour)')
        axes[0, 1].axis('off')
        
        # 真实掩码
        im1 = axes[0, 2].imshow(mask_np, cmap='viridis', vmin=0, vmax=model.num_classes-1)
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].axis('off')
        
        # 预测掩码
        im2 = axes[0, 3].imshow(pred_np, cmap='viridis', vmin=0, vmax=model.num_classes-1)
        axes[0, 3].set_title('Predicted Mask')
        axes[0, 3].axis('off')
        
        # 创建类别标签
        classes = ["bkg", "chan_rfi", "point_rfi"]  # 从FITSDataset.CLASSES获取
        
        # 在底部添加颜色条和类别标签
        # 为每个类别创建颜色条
        for j in range(model.num_classes):
            # 创建一个小的颜色条
            cbar_data = np.full((1, 100), j, dtype=int)
            im_cbar = axes[1, j].imshow(cbar_data, cmap='viridis', vmin=0, vmax=model.num_classes-1, aspect='auto')
            axes[1, j].set_title(f'{classes[j]} (Class {j})', fontsize=10)
            axes[1, j].axis('off')
        
        # 隐藏多余的子图
        for j in range(model.num_classes, 4):
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'inference_sample_{i+1}.png'), dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 保存推理可视化结果: {os.path.join(save_dir, f'inference_sample_{i+1}.png')}")
    
    print(f"🎉 推理可视化完成！共处理 {min(num_samples, len(dataset))} 个样本，结果保存至 {save_dir}/")


if __name__ == "__main__":
    # ============ 配置参数 ============
    # 数据集路径（根据你的数据集调整）
    dataset_top_dir = "/home/cbm/deRFI/Datasets/Dataset_G200.48+2.54_5978_2classes_NoSubMed_ThreshCorrected"
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
        num_samples=50,  # 可视化前5个样本
        save_dir=save_dir
    )