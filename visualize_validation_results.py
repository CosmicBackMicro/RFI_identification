#!/usr/bin/env python3
"""
可视化验证集的识别效果
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
import fitsio
import albumentations as albu
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# 导入之前定义的类
import sys
sys.path.append('/home/cbm/deRFI/src')
from UNet import UNetLightningModule, FITSDataset, get_validation_augmentation, get_preprocessing

def visualize_validation_predictions(model, val_loader, num_samples=6, save_path="validation_results.png"):
    """
    可视化验证集的预测结果
    
    Args:
        model: 训练好的模型
        val_loader: 验证集数据加载器
        num_samples: 要可视化的样本数量
        save_path: 保存图片的路径
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 准备图像显示
    cols = 4  # 原图，真实掩码，预测掩码，叠加图
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if sample_count >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.numpy()
            
            # 模型预测
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            # 获取概率图用于可视化
            rfi_probs = probs[:, 1].cpu().numpy()  # RFI类别的概率
            
            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                    
                # 原始图像 (去归一化显示)
                img = images[i].cpu().numpy().squeeze()
                img_display = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                
                # 真实掩码
                true_mask = masks[i]
                
                # 预测掩码
                pred_mask = predictions[i]
                
                # RFI概率图
                rfi_prob = rfi_probs[i]
                
                # 第一列：原始图像
                axes[sample_count, 0].imshow(img_display, cmap='gray', vmin=0, vmax=255)
                axes[sample_count, 0].set_title(f'Original Image {sample_count+1}')
                axes[sample_count, 0].axis('off')
                
                # 第二列：真实掩码
                axes[sample_count, 1].imshow(true_mask, cmap='jet', vmin=0, vmax=1)
                axes[sample_count, 1].set_title('Ground Truth')
                axes[sample_count, 1].axis('off')
                
                # 第三列：预测掩码
                axes[sample_count, 2].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
                axes[sample_count, 2].set_title('Prediction')
                axes[sample_count, 2].axis('off')
                
                # 第四列：RFI概率热图叠加
                axes[sample_count, 3].imshow(img_display, cmap='gray', alpha=0.7)
                im = axes[sample_count, 3].imshow(rfi_prob, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                axes[sample_count, 3].set_title('RFI Probability Overlay')
                axes[sample_count, 3].axis('off')
                
                sample_count += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"验证结果已保存到: {save_path}")
    plt.show()

def calculate_detailed_metrics(model, val_loader):
    """
    计算详细的评估指标
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks_device = masks.to(device)
            
            # 模型预测
            outputs = model(images)
            
            # 计算损失
            loss = model.joint_loss(outputs, masks_device)
            total_loss += loss.item()
            
            # 获取预测
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(masks.numpy().flatten())
            
            num_batches += 1
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')
    cm = confusion_matrix(all_targets, all_predictions)
    
    avg_loss = total_loss / num_batches
    
    print("=== 验证集详细评估结果 ===")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"总体准确率: {accuracy:.4f}")
    print(f"RFI检测精确率: {precision:.4f}")
    print(f"RFI检测召回率: {recall:.4f}")
    print(f"RFI检测F1分数: {f1:.4f}")
    print("\n混淆矩阵:")
    print("        预测")
    print("实际    背景  RFI")
    print(f"背景    {cm[0,0]:>6} {cm[0,1]:>4}")
    print(f"RFI     {cm[1,0]:>6} {cm[1,1]:>4}")
    
    # 计算IoU
    intersection = cm[1,1]
    union = cm[1,1] + cm[1,0] + cm[0,1]
    iou = intersection / union if union > 0 else 0
    print(f"\nRFI检测IoU: {iou:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'confusion_matrix': cm
    }

def compare_predictions_with_raw_fits(dataset, model, sample_indices=[0, 1, 2], save_path="comparison_with_raw.png"):
    """
    对比原始FITS数据、预处理后的数据和模型预测
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    cols = 5  # 原始FITS，预处理图像，真实掩码，预测掩码，概率图
    rows = len(sample_indices)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, sample_idx in enumerate(sample_indices):
        # 获取原始FITS数据
        try:
            with fitsio.FITS(dataset.image_list[sample_idx], 'r') as fits:
                fits_header = fits[1].read_header()
                fits_data = fits[1].read()
                
            nsamp = fits_header["NBLOCKS"] * fits_header["NSBLK"]
            nchan = fits_header["NCHAN"]
            data = fits_data[0]["DATA"].reshape(nsamp, nchan)
            dat_scl = fits_data[0]["DAT_SCL"]
            dat_offs = fits_data[0]["DAT_OFFS"]
            raw_image = (data.astype(np.float32) + dat_offs[np.newaxis, :]) * dat_scl[np.newaxis, :]
            
            # 获取预处理后的数据
            processed_img, true_mask = dataset[sample_idx]
            
            # 模型预测
            with torch.no_grad():
                img_tensor = torch.from_numpy(processed_img).unsqueeze(0).to(device)
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_mask = torch.argmax(probs, dim=1).cpu().numpy().squeeze()
                rfi_prob = probs[0, 1].cpu().numpy()
            
            # 显示原始FITS数据
            raw_min, raw_max = np.percentile(raw_image, [1, 99])
            raw_display = np.clip((raw_image - raw_min) / (raw_max - raw_min) * 255, 0, 255).astype(np.uint8)
            axes[row_idx, 0].imshow(raw_display, cmap='gray')
            axes[row_idx, 0].set_title(f'Raw FITS {sample_idx+1}')
            axes[row_idx, 0].axis('off')
            
            # 显示预处理后的图像
            if isinstance(processed_img, np.ndarray):
                proc_img = processed_img.squeeze()
            else:
                proc_img = processed_img
            proc_display = np.clip(proc_img * 255.0, 0, 255).astype(np.uint8)
            axes[row_idx, 1].imshow(proc_display, cmap='gray')
            axes[row_idx, 1].set_title('Processed Image')
            axes[row_idx, 1].axis('off')
            
            # 显示真实掩码
            axes[row_idx, 2].imshow(true_mask, cmap='jet', vmin=0, vmax=1)
            axes[row_idx, 2].set_title('Ground Truth')
            axes[row_idx, 2].axis('off')
            
            # 显示预测掩码
            axes[row_idx, 3].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
            axes[row_idx, 3].set_title('Prediction')
            axes[row_idx, 3].axis('off')
            
            # 显示RFI概率图
            im = axes[row_idx, 4].imshow(rfi_prob, cmap='Reds', vmin=0, vmax=1)
            axes[row_idx, 4].set_title('RFI Probability')
            axes[row_idx, 4].axis('off')
            
            # 添加颜色条
            if row_idx == 0:
                plt.colorbar(im, ax=axes[row_idx, 4], fraction=0.046, pad=0.04)
            
        except Exception as e:
            print(f"处理样本 {sample_idx} 时出错: {e}")
            # 填充空白图像
            for col in range(cols):
                axes[row_idx, col].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[row_idx, col].transAxes)
                axes[row_idx, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比结果已保存到: {save_path}")
    plt.show()

def main():
    """主函数"""
    # 设置路径
    dataset_top_dir = "/home/cbm/deRFI/dataset"
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")
    
    # 加载最佳模型
    checkpoint_path = "/home/cbm/deRFI/final_model.ckpt"  # 或者使用最佳检查点
    
    if not os.path.exists(checkpoint_path):
        # 查找最新的检查点
        checkpoint_dir = "/home/cbm/deRFI/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoints:
                # 选择最佳模型（通过文件名中的IoU值）
                best_checkpoint = None
                best_iou = 0
                for ckpt in checkpoints:
                    if 'val_iou' in ckpt:
                        try:
                            iou_str = ckpt.split('val_iou=')[1].split('.ckpt')[0].split('-')[0]
                            iou_val = float(iou_str)
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_checkpoint = ckpt
                        except:
                            continue
                
                if best_checkpoint:
                    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
                    print(f"使用最佳检查点: {checkpoint_path} (IoU: {best_iou:.4f})")
                else:
                    # 使用last.ckpt
                    checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
                    print(f"使用最后的检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"找不到模型检查点: {checkpoint_path}")
        return
    
    # 加载模型
    print("加载模型...")
    model = UNetLightningModule.load_from_checkpoint(checkpoint_path)
    print("模型加载完成")
    
    # 创建验证数据集
    print("创建验证数据集...")
    CLASSES = ["background", "rfi"]
    
    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(None),
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"验证集包含 {len(val_dataset)} 个样本")
    
    # 计算详细指标
    print("\n计算验证集指标...")
    metrics = calculate_detailed_metrics(model, val_loader)
    
    # 可视化预测结果
    print("\n生成预测可视化...")
    visualize_validation_predictions(model, val_loader, num_samples=6, save_path="validation_predictions.png")
    
    # 对比原始数据与预测
    print("\n生成原始数据对比...")
    compare_predictions_with_raw_fits(val_dataset, model, sample_indices=[0, 1, 2], save_path="raw_vs_prediction.png")
    
    print("\n可视化完成！")

if __name__ == "__main__":
    main()
