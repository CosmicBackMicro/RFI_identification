import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from typing import Optional, List
from UNet import UNetLightningModule, FITSDataset, get_stable_training_augmentation, get_preprocessing, get_validation_augmentation

class DistillationLoss(nn.Module):
    """
    Task Loss (CE + FT) + Distillation Loss (KL Divergence)
    """
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels, task_loss_fn):
        # 1. 基础任务损失 (由教师/学生共同定义的 CE + FocalTversky)
        # 这里为了简化，我们直接调用 LightningModule 里的 joint_loss
        
        # 2. KL 散度蒸馏损失
        # T 越高，概率分布越平滑，学生能学到更多“类间相似度”信息
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return distill_loss

class DistillationModule(pl.LightningModule):
    def __init__(
        self, 
        teacher_ckpt_path: str,
        student_encoder: str = "timm-mobilenetv3_small_100", 
        classes: int = 6,
        lr: float = 1e-3,
        temperature: float = 3.0,
        distill_weight: float = 0.7  # 蒸馏损失的权重
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 加载教师模型 (冻结权重)
        # 假设教师是 MiT-B2 U-Net
        self.teacher = UNetLightningModule.load_from_checkpoint(teacher_ckpt_path)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. 初始化学生模型 (采用更轻量的 MobileNetV3-Small 骨干)
        # MobileNetV3 擅长快速下采样，对尺度敏感
        self.student = smp.Unet(
            encoder_name=student_encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=classes,
            activation=None
        )

        self.distill_loss_fn = DistillationLoss(temperature=temperature)

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # 教师推理 (不记录梯度)
        with torch.no_grad():
            teacher_logits = self.teacher.model(images)
        
        # 学生推理
        student_logits = self.student(images)
        student_probs = torch.softmax(student_logits, dim=1)

        # 计算基础损失 (CE + FT)
        base_loss = self.teacher.joint_loss(student_logits, student_probs, masks)
        
        # 计算蒸馏损失
        distill_loss = self.distill_loss_fn(student_logits, teacher_logits, masks, None)

        total_loss = (1 - self.hparams.distill_weight) * base_loss + self.hparams.distill_weight * distill_loss
        
        self.log("train_total_loss", total_loss, prog_bar=True)
        self.log("train_distill_loss", distill_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # 验证逻辑与原 UNet 脚本保持一致，仅评估学生模型
        images, masks = batch
        logits = self.student(images)
        probs = torch.softmax(logits, dim=1)
        # ... 复用评价指标代码 ...
        return self.teacher.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]

if __name__ == "__main__":
    # 配置
    TEACHER_CKPT = "/home/cbm/deRFI/PaperExperiments/ReviseWork/BESTRevised_MiTB2UNet_ep24_valFGmIoU0.7285.ckpt"
    DATASET_DIR = "/home/cbm/deRFI/Datasets/SynthesizedDataset"
    
    # 准备数据 (复用 UNet.py 中的 FITSDataset)
    train_dataset = FITSDataset(
        os.path.join(DATASET_DIR, "image/train"),
        os.path.join(DATASET_DIR, "mask/train"),
        classes=["bkg", "horizontal", "vertical", "point", "block", "pulsar"],
        augmentation=get_stable_training_augmentation(640),
        preprocessing=get_preprocessing(None)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    
    # 初始化蒸馏训练
    model = DistillationModule(teacher_ckpt_path=TEACHER_CKPT)
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_fg_miou", mode="max")]
    )
    
    trainer.fit(model, train_loader)
