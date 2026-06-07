import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as albu
from UNetOnServer import UNetLightningModule, FITSDataset, get_preprocessing

# Import the plotting function from plot_custom_cm for consistency
from plot_custom_cm import plot_custom_cm

def main():
    # --- 1. 配置路径 ---
    # 根据你的描述，设置 checkpoint 和数据集路径
    checkpoint_path = "/home/cbm/checkpoints_UNetBaseline_20260508/UNetBaseline_20260509/UNet_epepoch=14_FGmIoUval_fg_miou=0.5927.ckpt"
    dataset_top_dir = "/home/cbm/deRFI/Datasets/SynthesizedDataset"
    val_image_dir = os.path.join(dataset_top_dir, "image/val")
    val_mask_dir = os.path.join(dataset_top_dir, "mask/val")
    
    output_npy = "results/confusion_matrix_unet.npy"
    output_pdf = "results/ConfusionMatrix_UNet_HighPrec.pdf"
    os.makedirs("results", exist_ok=True)

    # --- 2. 加载模型 ---
    print(f"Loading checkpoint: {checkpoint_path}")
    # 注意：UNetLightningModule 构造函数中的某些参数需要与训练时一致
    model = UNetLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    # --- 3. 准备验证集 ---
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block", "pulsar"]
    val_dataset = FITSDataset(
        val_image_dir, val_mask_dir, classes=CLASSES,
        augmentation=albu.Resize(512, 512), # 与验证逻辑保持一致
        preprocessing=get_preprocessing()
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # --- 4. 运行推断并手动收集混淆矩阵 ---
    print("Running validation and collecting confusion matrix...")
    
    # 建立一个独立的混淆矩阵统计器，避免与 Lightning 内部状态竞争
    from torchmetrics import ConfusionMatrix as CM_Metric
    raw_cm_collector = CM_Metric(task="multiclass", num_classes=len(CLASSES)).to(model.device)
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(model.device), y.to(model.device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            # 排除忽略索引 255
            valid_mask = (y != 255)
            if valid_mask.any():
                raw_cm_collector.update(preds[valid_mask], y[valid_mask])
    
    # 获取最终的混淆矩阵张量
    cm = raw_cm_collector.compute().cpu().numpy()
    np.save(output_npy, cm)
    print(f"Raw confusion matrix saved to {output_npy}")

    # --- 5. 使用 plot_custom_cm 逻辑绘图 ---
    labels = ['Bkg', 'Horizontal', 'Vertical', 'Point', 'Block', 'Pulsar']
    title = "Normalized Confusion Matrix (UNet Baseline)"
    
    # 调用 plot_custom_cm (它内部已经实现了 3 位小数和样式调整)
    plot_custom_cm(cm, labels, title, output_pdf)
    print(f"High precision confusion matrix plotted to {output_pdf}")

if __name__ == "__main__":
    main()
