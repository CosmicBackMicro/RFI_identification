import os, re, json, hashlib, io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchmetrics import ConfusionMatrix
import warnings

# 限制底层库线程数，防止在 WSL2 / 多进程 DataLoader 下发生线程争用。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")  # 允许访问多张 GPU（如可用）
os.environ['SMP_HUB_MODE'] = "original"
os.environ["TORCHINDUCTOR_USE_CUDA_GRAPH"] = "0"
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
os.environ["TORCHINDUCTOR_USE_CUDAGRAPHS"] = "0"
os.environ["TORCH_CUDAGRAPHS"] = "0"

import torch
import cv2
import fitsio
import numpy as np
from typing import Any, cast, Optional, List
import torch.multiprocessing as mp
# Suppress albumentations network warning when checking version (offline/air-gapped environment)
warnings.filterwarnings("ignore", message="Error fetching version info")
# Suppress Lightning / Fabric SLURM hint when not running under srun
warnings.filterwarnings(
    "ignore",
    message=r"The `srun` command is available on your system but is not used\..*",
    category=UserWarning,
)
# 静音 Lightning 关于 num_workers 过少的提示
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
)
# 静音 DDP 选项下的 grad strides mismatch warning
warnings.filterwarnings(
    "ignore",
    message=r"Grad strides do not match bucket view strides.*",
    category=UserWarning,
)
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('medium')

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.4,
        beta: float = 0.6,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        assert 0 <= label_smoothing < 1.0
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        y_pred: torch.Tensor,  # (N, C, H, W) 
        y_true: torch.Tensor,  # (N, H, W)
        class_weights: Optional[torch.Tensor] = None,  # (C,)
    ) -> torch.Tensor:
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        N, C, H, W = y_pred.shape
        y_true_oh = F.one_hot(y_true.clamp_min(0), num_classes=C).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            ignore_mask = (y_true == self.ignore_index).unsqueeze(1)
            y_true_oh = torch.where(ignore_mask, torch.zeros_like(y_true_oh), y_true_oh)
            y_pred = torch.where(ignore_mask, torch.zeros_like(y_pred), y_pred)

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            y_true_oh = (1 - eps) * y_true_oh + eps / C

        y_pred = y_pred.clamp(min=1e-7, max=1.0 - 1e-7)
        dims = (0, 2, 3)
        TP = (y_true_oh * y_pred).sum(dim=dims)
        FP = (y_pred * (1 - y_true_oh)).sum(dim=dims)
        FN = ((1 - y_pred) * y_true_oh).sum(dim=dims)
        TI = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth + 1e-7)
        loss_c = (1.0 - TI + 1e-7).pow(self.gamma)

        if class_weights is not None:
            cw = class_weights.to(loss_c.device).float()
            if cw.numel() == C:
                loss_c = loss_c * cw

        if self.reduction == 'mean':
            return loss_c.mean()
        elif self.reduction == 'sum':
            return loss_c.sum()
        else:
            return loss_c

class UNetLightningModule(pl.LightningModule):
    def __init__(self, encoder_name="resnet34", encoder_weights: Optional[str] = "imagenet", classes=2, learning_rate=0.0001, scheduler_type="cosine_warmup", class_weights=None, focal_gamma: float = 2.0, loss_alpha: float = 0.7, loss_beta: float = 0.3, ce_weight: float = 0.5, ft_weight: float = 0.5, use_compile: bool = False, ft_warmup_epochs: int = 5, class_names: Optional[List[str]] = None, point_class_index: int = 3, point_aux_weight: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_type = scheduler_type
        self.num_classes = classes
        self.use_compile = use_compile
        self.ft_warmup_epochs = ft_warmup_epochs
        self.point_class_index = point_class_index
        self.point_aux_weight = point_aux_weight

        # 使用 U-Net 模型架构，采用 ResNet-34 作为 Encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=classes,
            activation=None,
        )
        
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.learning_rate = learning_rate
        self.ce_weight = ce_weight
        self.ft_weight = ft_weight

        self.focal_tversky = FocalTverskyLoss(
            num_classes=self.num_classes,
            alpha=loss_alpha,
            beta=loss_beta,
            gamma=focal_gamma,
            smooth=1e-6,
            label_smoothing=0.01,
        )
        
        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
            self.class_names = class_names if len(class_names) == self.num_classes else [f"class_{i}" for i in range(self.num_classes)]

        if use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass
        
        def _make_confmat(num_classes: int):
            try:
                return ConfusionMatrix(task="multiclass", num_classes=num_classes)
            except TypeError:
                return ConfusionMatrix(num_classes=num_classes)

        self.val_conf_matrix = _make_confmat(self.num_classes)

        # --- 显式初始化：在无预训练权重时非常重要 ---
        if encoder_weights is None:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

    def joint_loss(self, logits: torch.Tensor, probs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss_ce = F.cross_entropy(logits, y_true, weight=self.class_weights, ignore_index=255)
        loss_ft = self.focal_tversky(probs, y_true, class_weights=self.class_weights)
        
        current_epoch = self.current_epoch
        if current_epoch < self.ft_warmup_epochs:
            ft_w = (current_epoch / self.ft_warmup_epochs) * self.ft_weight
        else:
            ft_w = self.ft_weight
        
        return self.ce_weight * loss_ce + ft_w * loss_ft

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        
        loss = self.joint_loss(logits, probs, y)
        
        # 记录训练损失
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # --- 诊断：每 50 步记录一次各类的平均预测概率 ---
        if batch_idx % 50 == 0:
            with torch.no_grad():
                avg_probs = probs.mean(dim=(0, 2, 3))
                for i, name in enumerate(self.class_names):
                    self.log(f"train_prob_avg_{name}", avg_probs[i], on_step=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        loss = self.joint_loss(logits, probs, y)
        preds = torch.argmax(probs, dim=1)
        
        # --- 诊断：记录验证集各类的平均预测概率 ---
        avg_probs_val = probs.mean(dim=(0, 2, 3))
        for i, name in enumerate(self.class_names):
            self.log(f"val_prob_avg_{name}", avg_probs_val[i], on_epoch=True, sync_dist=True)

        valid_mask = (y != 255)
        if valid_mask.any():
            self.val_conf_matrix.update(preds[valid_mask], y[valid_mask])
        
        # --- 新增：命令行进度条监视 block IoU ---
        if self.num_classes > 4:
            y_block = (y == 4)
            p_block = (preds == 4)
            if y_block.any():
                it = (y_block & p_block).sum().float()
                un = (y_block | p_block).sum().float()
                iou_block = it / (un + 1e-7)
                self.log("v_block_iou", iou_block, prog_bar=True, sync_dist=False)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        conf_matrix = self.val_conf_matrix.compute()
        self.val_conf_matrix.reset()
        
        cm = conf_matrix.float()
        
        # 计算 Recall, Precision, F1 和 IoU
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        
        # 避免除以 0
        denominator_iou = tp + fp + fn + 1e-7
        denominator_prec = tp + fp + 1e-7
        denominator_rec = tp + fn + 1e-7
        
        iou = tp / denominator_iou
        precision = tp / denominator_prec
        recall = tp / denominator_rec
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # 整体指标
        miou = iou.mean()
        mf1 = f1.mean()
        
        # 前景指标 (排除背景，假设 class 0 是背景)
        fg_iou = iou[1:]
        fg_miou = fg_iou.mean()
        fg_mf1 = f1[1:].mean()
        
        self.log("val_miou", miou, prog_bar=True, sync_dist=True)
        self.log("val_fg_miou", fg_miou, prog_bar=True, sync_dist=True)
        self.log("val_mf1", mf1, prog_bar=False, sync_dist=True)
        self.log("val_fg_mf1", fg_mf1, prog_bar=True, sync_dist=True)
        
        # 详细记录每一类的各项指标
        for i, name in enumerate(self.class_names):
            self.log(f"val_iou_{name}", iou[i], sync_dist=True)
            self.log(f"val_f1_{name}", f1[i], sync_dist=True)
            self.log(f"val_recall_{name}", recall[i], sync_dist=True)
            self.log(f"val_precision_{name}", precision[i], sync_dist=True)

        # 可视化混淆矩阵并记录到 TensorBoard (可选，模仿 SegFormerOnServer)
        if self.global_rank == 0:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                cm_np = cm.cpu().numpy()
                # 归一化以便观察比例
                cm_norm = cm_np / (cm_np.sum(axis=1, keepdims=True) + 1e-7)
                sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                            xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
                ax.set_title(f"Normalized Confusion Matrix (Epoch {self.current_epoch})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                
                if self.logger and hasattr(self.logger, "experiment"):
                    self.logger.experiment.add_figure("ConfusionMatrix", fig, global_step=self.global_step)
                plt.close(fig)
            except Exception as e:
                print(f"Error logging confusion matrix figure: {e}")

    def configure_optimizers(self):
        # 优化器分组：BN 和 Bias 不进行权重衰减，使训练更稳定
        no_decay = ["bias", "bn", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 1e-4,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        if self.scheduler_type == "cosine_warmup":
            # 简单的线性预热 + 余弦退火
            def lr_lambda(current_step):
                warmup_steps = 1000
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (self.trainer.estimated_stepping_batches - warmup_steps)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_fg_miou"},
        }

class FITSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, classes, augmentation=None, preprocessing=None, normalization_method="median_sigma", crop_size=512, point_class_value=6, pulsar_class_value=8, point_oversample_factor=2.0, pulsar_oversample_factor=2.0, point_crop_prob=0.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = [f for f in os.listdir(image_dir) if f.endswith('.fits')]
        self.image_list = [os.path.join(image_dir, f) for f in self.ids]
        self.mask_list = [os.path.join(mask_dir, f.replace('.fits', '.png')) for f in self.ids]
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.normalization_method = normalization_method
        
        # 类别映射：支持两种可能的 block 标签 (4 和 7)
        # 0:bkg, 1:horizontal, 2:vertical, 6:point, 4&7:block, 8:pulsar
        self.class_mapping = {0: 0, 1: 1, 2: 2, 6: 3, 4: 4, 7: 4, 8: 5}
        
        # 计算样本权重用于 Sampler
        self.sample_weights = self._compute_sample_weights(point_class_value, pulsar_class_value, point_oversample_factor, pulsar_oversample_factor)

    def _compute_sample_weights(self, point_val, pulsar_val, point_factor, pulsar_factor):
        weights = np.ones(len(self.ids))
        # 简单实现：检查文件名或预先扫描。这里为了简化假设所有样本权重为1，或根据需求在此扩展。
        return torch.from_numpy(weights)

    @staticmethod
    def normalize_image_mean_std(image, k=5.0):
        mean = np.mean(image)
        std = np.std(image)
        if std < 1e-6:
            return (image - mean)
        return (image - mean) / (k * std)

    @staticmethod
    def load_fits_image(fits_path):
        with fitsio.FITS(fits_path, 'r') as fits:
            fits_header = fits[1].read_header()
            fits_data = fits[1].read()
        nchan = int(fits_header["NCHAN"])
        raw = np.asarray(fits_data[0]["DATA"], dtype=np.uint8)
        nsamp = raw.size // nchan
        data = raw.reshape(nsamp, nchan).astype(np.float32)
        dat_scl = np.asarray(fits_data[0]["DAT_SCL"], dtype=np.float32)
        dat_offs = np.asarray(fits_data[0]["DAT_OFFS"], dtype=np.float32)
        data *= dat_scl[np.newaxis, :]
        data += dat_offs[np.newaxis, :]
        image = np.flipud(data.T)
        return image

    def __getitem__(self, i):
        raw_image = self.load_fits_image(self.image_list[i])
        raw_mask = cv2.imread(self.mask_list[i], cv2.IMREAD_UNCHANGED)
        mask_labels = np.zeros_like(raw_mask, dtype=np.uint8)
        for mask_val, class_idx in self.class_mapping.items():
            mask_labels[raw_mask == mask_val] = class_idx

        image = self.normalize_image_mean_std(raw_image)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_labels)
            image, mask_labels = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask_labels)
            image, mask_labels = sample['image'], sample['mask']

        return image, mask_labels

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def compute_class_weights(image_dir, mask_dir, classes, class_mapping, num_samples=500):
        # 恢复原始权重，以便在不干预的情况下诊断 block 类别指标为 0 的原因
        # 0:bkg, 1:horizontal, 2:vertical, 3:point, 4:block, 5:pulsar
        weights = torch.tensor([1.0, 2.0, 2.0, 5.0, 2.0, 3.0])
        return weights

def get_stable_training_augmentation(size: int = 512):
    return albu.Compose([
        albu.Resize(size, size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(4, 16), hole_width_range=(4, 16), fill=0, p=0.2),
        albu.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        albu.RandomBrightnessContrast(p=0.5),
    ])

def to_tensor(x, **kwargs):
    if x.dtype in [np.uint8, np.int32, np.int64]:
        return torch.from_numpy(np.ascontiguousarray(x.astype(np.int64)))
    else:
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))

def get_preprocessing():
    return albu.Compose([albu.Lambda(image=to_tensor, mask=to_tensor)])

if __name__ == "__main__":
    print("🚀 开始 U-Net 射电干扰分割训练脚本")
    
    dataset_top_dir = "/home/bmcao/deRFI/Datasets/SynthesizedDataset"
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    train_image_dir = os.path.join(image_dir, "train")
    train_mask_dir = os.path.join(mask_dir, "train")
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    TOTAL_MAX_EPOCHS = 15
    ENCODER = "resnet34" # U-Net 常用且鲁棒的基线 Encoder
    # 鉴于服务器无法连接外网下载预训练权重，将 encoder_weights 设为 None 以使用随机初始化
    ENCODER_WEIGHTS = None 
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block", "pulsar"]
    
    CONFIG_MODE = "server"
    configs = {
        "server": {
            "LEARNING_RATE": 1e-4, # U-Net (ResNet) 通常可以使用稍高的学习率
            "BATCH_SIZE": 5,     # 保持与 SegFormer 一致的 Batch Size
            "NUM_WORKERS": 8,
            "PRECISION": "16-mixed",
            "DEVICES": 2,
            "ACCUMULATE_GRAD_BATCHES": 1,
        }
    }
    cfg = configs[CONFIG_MODE]

    train_dataset = FITSDataset(
        train_image_dir, train_mask_dir, classes=CLASSES,
        augmentation=get_stable_training_augmentation(512),
        preprocessing=get_preprocessing()
    )
    val_dataset = FITSDataset(
        val_image_dir, val_mask_dir, classes=CLASSES,
        augmentation=albu.Resize(512, 512),
        preprocessing=get_preprocessing()
    )

    class_weights = FITSDataset.compute_class_weights(train_image_dir, train_mask_dir, CLASSES, None)

    # --- 快速验证：统计 500 张图出现的 label ---
    print("🔍 验证数据集 Label 分布 (500张)...")
    check_labels = set()
    for i in range(min(500, len(train_dataset))):
        _, mask = train_dataset[i]
        # mask 此时已经是映射后的 torch tensor (由 preprocessing 转换)
        check_labels.update(torch.unique(mask).cpu().numpy().tolist())
    print(f"✅ 映射后的 Label 索引: {sorted(list(check_labels))} (应包含 4)")

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=cfg["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=cfg["NUM_WORKERS"], pin_memory=True)

    model = UNetLightningModule(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        class_names=CLASSES,
        learning_rate=cfg["LEARNING_RATE"],
        class_weights=class_weights,
    )

    # 配置 TensorBoard 日志记录器
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir='training_logs',
        name='unet_training',
        version=None,
        log_graph=False,
        default_hp_metric=False
    )

    callbacks = [
        ModelCheckpoint(dirpath='checkpoints/train', filename='UNet_ep{epoch:02d}_FGmIoU{val_fg_miou:.4f}', monitor='val_fg_miou', mode='max', save_top_k=5, save_last=True),
        EarlyStopping(monitor='val_fg_miou', patience=10, mode='max'),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        max_epochs=TOTAL_MAX_EPOCHS,
        accelerator='gpu',
        devices=cfg["DEVICES"],
        callbacks=callbacks,
        logger=logger,  # 启用 Logger
        precision=cfg["PRECISION"],
        gradient_clip_val=1.0,  # 梯度裁剪：防止训练初期或 Focal Loss 导致的梯度爆炸
        accumulate_grad_batches=cfg["ACCUMULATE_GRAD_BATCHES"],
        strategy='ddp_find_unused_parameters_true' if cfg["DEVICES"] > 1 else 'auto'
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("🎉 U-Net 训练完成！")
