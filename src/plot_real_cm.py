import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# 导入项目中定义的模型和数据集类
from SegFormer import UNetLightningModule, FITSDataset, get_preprocessing, get_validation_augmentation

def main():
    # 1. 配置路径
    CKPT_PATH = "/home/cbm/deRFI/checkpoints/train/best_model-epoch=24-fgmIoU_val_fg_miou=0.7285.ckpt"
    VAL_IMAGE_DIR = "/home/cbm/deRFI/Datasets/SynthesizedDataset/image/val"
    VAL_MASK_DIR = "/home/cbm/deRFI/Datasets/SynthesizedDataset/mask/val"
    
    # 根据数据集定义类别映射
    # FITSDataset 内部 class_mapping 逻辑: 0:0(bkg), 1:1(horiz), 2:2(vert), 6:3(point), 7:4(block), 8:5(pulsar)
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block", "pulsar"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    # 2. 加载模型
    print(f"Loading checkpoint: {CKPT_PATH}")
    # map_location 确保在不同机器上加载正常
    model = UNetLightningModule.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    # 3. 准备验证集
    print("Preparing validation dataset...")
    val_dataset = FITSDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        classes=CLASSES,
        augmentation=get_validation_augmentation(640, 640),
        preprocessing=get_preprocessing(None)
    )
    
    # batch_size=1 确保不会因为图像大小不一报错，同时方便调试
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 4. 执行推理并收集结果
    all_preds = []
    all_masks = []
    
    print(f"Starting inference on {len(val_dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, masks = batch
            images = images.to(DEVICE)
            
            # 推理获得概率并取预测类
            logits, probs = model(images)
            preds = torch.argmax(probs, dim=1).cpu().numpy().flatten()
            masks_np = masks.numpy().flatten()
            
            # 特别重要：过滤 ignore_index = 255
            valid_mask = (masks_np != 255)
            if np.any(valid_mask):
                all_preds.append(preds[valid_mask])
                all_masks.append(masks_np[valid_mask])

    # 5. 计算混淆矩阵
    print("Merging results and computing confusion matrix...")
    y_true = np.concatenate(all_masks)
    y_pred = np.concatenate(all_preds)

    # 这里的 labels 必须与 CLASSES 顺序一致 (0-5)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    
    # 归一化 (Recall 视角: 按行归一化)
    # 表示：真实的 RFI 类别中，有多少比例被正确识别，有多少比例被错划为其他类
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # 6. 可视化绘图
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap='Greens',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Label (Output)', fontsize=12)
    plt.ylabel('True Label (Ground Truth)', fontsize=12)
    plt.title(f'Pixel-Level Confusion Matrix (Recall Normalized)\nModel: ep24_valFGmIoU0.7285', fontsize=14)
    
    save_img = "/home/cbm/deRFI/results/inference_plots/cm_ep24_precise.png"
    os.makedirs(os.path.dirname(save_img), exist_ok=True)
    plt.savefig(save_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 真实的混淆矩阵已生成：{save_img}")
    print("非对角线区域现在显示了真实的误报/漏报分布，不再是平均分配的值。")

if __name__ == "__main__":
    main()
