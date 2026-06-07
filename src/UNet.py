import torch
import cv2
# 优化点: 禁用 OpenCV 的内部多线程，防止在多进程环境下线程数溢出 (同步自 TrainingOnServer)
cv2.setNumThreads(0)
import os, re

# 限制底层库线程数，防止线程爆炸 (同步自 TrainingOnServer)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import fitsio
import numpy as np
from typing import Any, cast, Optional, List
import albumentations as albu

import sys

# 💡 针对国内网络环境优化：添加 Hugging Face 镜像站地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 避免 cudagraph 池相关错误
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['SMP_HUB_MODE'] = "original"
os.environ["TORCHINDUCTOR_USE_CUDA_GRAPH"] = "0" 
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
os.environ["TORCHINDUCTOR_USE_CUDAGRAPHS"] = "0"
os.environ["TORCH_CUDAGRAPHS"] = "0"

import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # 隐藏无关告警
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

#确保禁用 Inductor 的 CUDA Graphs
try:
    import torch._inductor.config as inductor_config
    if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
        inductor_config.triton.cudagraphs = False
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
try:
    # 启用 Flash / Mem-Efficient SDPA，显著提升性能
    from torch.backends.cuda import sdp_kernel
except Exception:
    pass


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for multi-class segmentation with probability inputs.

    Features:
    - Accepts probabilities (from a softmax output)
    - Handles class imbalance via alpha/beta and optional class_weights
    - More robust to imperfect labels with focal focusing (gamma) and label smoothing

    Args:
        num_classes: number of classes (C)
        alpha: weight for FN in Tversky index (typically 0.7)
        beta: weight for FP in Tversky index (typically 0.3)
        gamma: focal parameter (>1 increases focus on hard examples; e.g., 1.333)
        smooth: numerical stability term
        reduction: 'mean' | 'sum' | 'none'
        ignore_index: optional label to ignore in targets
        label_smoothing: small epsilon to soften one-hot targets, improves noise robustness
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.333,
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
        assert C == self.num_classes, f"num_classes mismatch: {C} vs {self.num_classes}"

        # One-hot encode targets: (N, H, W) -> (N, H, W, C) -> (N, C, H, W)
        y_true_oh = F.one_hot(y_true.clamp_min(0), num_classes=C).permute(0, 3, 1, 2).float()

        # Handle ignore_index by zeroing contributions
        if self.ignore_index is not None:
            ignore_mask = (y_true == self.ignore_index).unsqueeze(1)  # (N,1,H,W)
            y_true_oh = torch.where(ignore_mask, torch.zeros_like(y_true_oh), y_true_oh)
            y_pred = torch.where(ignore_mask, torch.zeros_like(y_pred), y_pred)

        # Optional label smoothing: soften hard labels to reduce overconfidence on noisy pixels
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            y_true_oh = (1 - eps) * y_true_oh + eps / C

        # Clamp probabilities for numerical stability
        y_pred = y_pred.clamp(min=0.0, max=1.0)

        # Compute TP, FP, FN per class
        dims = (0, 2, 3)  # sum over N, H, W
        TP = (y_true_oh * y_pred).sum(dim=dims)
        FP = (y_pred * (1 - y_true_oh)).sum(dim=dims)
        FN = ((1 - y_pred) * y_true_oh).sum(dim=dims)

        # Tversky index per class
        TI = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Focal Tversky loss per class
        loss_c = (1.0 - TI).pow(self.gamma)

        # Apply optional class weights to reduce background dominance
        if class_weights is not None:
            cw = class_weights.to(loss_c.device).float()
            if cw.numel() == C:
                loss_c = loss_c * cw

        # Reduction
        if self.reduction == 'mean':
            return loss_c.mean()
        elif self.reduction == 'sum':
            return loss_c.sum()
        else:
            return loss_c  # (C,)


# MobileMamba 已移除 —— 保持代码库为 SegFormer-only，以简化运行并消除对本地 CUDA 扩展的依赖。

class UNetLightningModule(pl.LightningModule):
    def __init__(self, encoder_name="mit_b2", classes=2, learning_rate=0.0001, scheduler_type="cosine_warmup", class_weights=None, focal_gamma: float = 2.0, loss_alpha: float = 0.7, loss_beta: float = 0.3, ce_weight: float = 0.5, ft_weight: float = 0.5, use_compile: bool = False, ft_warmup_epochs: int = 5, class_names: Optional[List[str]] = None, point_class_index: int = 3, point_aux_weight: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_type = scheduler_type
        self.num_classes = classes
        self.use_compile = use_compile
        self.ft_warmup_epochs = ft_warmup_epochs  # 前若干epoch逐步增加FT权重，稳定早期训练
        self.point_class_index = point_class_index
        self.point_aux_weight = point_aux_weight

        # 用于全局混淆矩阵统计
        self._epoch_preds = []
        self._epoch_targets = []

        # 使用 U-Net 架构 + MiT-B2 编码器
        # 不加载预训练权重，直接从随机初始化开始训练
        self.backbone_type = "mit_b2"
        self.model = smp.Unet(
            encoder_name="mit_b2",        # MiT-B2 编码器 (Mix Transformer)
            encoder_weights=None,         # 禁用 ImageNet 预训练，完全由射电数据驱动
            in_channels=1,                # 射电 2D 数据为单通道
            classes=classes,              # 输出类别数
            activation=None,              # 后续 joint_loss 需要 raw logits
        )

        # 类别权重（用于缓解类别不平衡）
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.learning_rate = learning_rate

    # 训练全程使用像素级交叉熵 + FocalTversky 组合损失
        self.ce_weight = ce_weight
        self.ft_weight = ft_weight

        # FocalTverskyLoss更好保留细节、小目标
        # label_smoothing减少标签噪声影响
        self.focal_tversky = FocalTverskyLoss(
            num_classes=self.num_classes,
            alpha=loss_alpha,
            beta=loss_beta,
            gamma=focal_gamma,
            smooth=1e-6,
            label_smoothing=0.01,
        )
        # 保存类别名称（可为 None）；用于在日志键上显示类名
        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
            # 如果长度不匹配则回退为索引名
            if len(class_names) != self.num_classes:
                print(f"[Warning] class_names length ({len(class_names)}) != num_classes ({self.num_classes}), fallback to numeric names")
                self.class_names = [f"class_{i}" for i in range(self.num_classes)]
            else:
                self.class_names = class_names

        # 可选：使用 torch.compile 编译以减少调度开销
        if use_compile and hasattr(torch, "compile"):
            try:
                # 显式在编译选项中关闭 cudagraphs，避免图池生命周期相关错误
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    options={
                        "triton.cudagraphs": False,
                        "cudagraphs": False,
                    },
                )
            except Exception:
                pass
        
    def joint_loss(self, logits: torch.Tensor, probs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        组合损失：像素级交叉熵 + FocalTversky。

        Args:
            logits: (N, C, H, W) 未归一化网络输出
            probs:  (N, C, H, W) softmax 后概率
            y_true: (N, H, W) 真实标签

        Returns:
            加权损失标量
        """
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        ce_weight = None
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            ce_weight = self.class_weights.to(logits.device)

        nll = F.cross_entropy(logits, y_true, weight=ce_weight)
        ft = self.focal_tversky(probs, y_true, class_weights=self.class_weights)

        try:
            epoch = int(self.current_epoch)
        except Exception:
            epoch = 0
        scale = 1.0
        if isinstance(self.ft_warmup_epochs, int) and self.ft_warmup_epochs > 0:
            scale = min(1.0, max(0.0, epoch / float(self.ft_warmup_epochs)))

        loss = self.ce_weight * nll + (self.ft_weight * scale) * ft
        return loss
    
    def forward(self, x):
        """
        前向传播。
        smp.Unet 输出的分辨率直接与输入相同 (512x512)，无需手动上采样。
        """
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
    # 返回 logits 与 probs，分别用于基于 logits 的 CE 和基于概率的 FocalTversky
        return logits, probs
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits, probs = self(images)
        loss = self.joint_loss(logits, probs, masks)
        preds = torch.argmax(probs, dim=1)

        preds_long = cast(torch.LongTensor, preds.long())
        masks_long = cast(torch.LongTensor, masks.long())
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds_long, masks_long, mode='multiclass', num_classes=self.num_classes
        )
        # 💡 重要：将 stats 按 batch 维度求和，得到 (C,)。
        # 使用 .long() 显式转换以满足 smp 指标对 LongTensor 的类型要求，并确保 per_class_f1[i] 为标量。

        tp, fp, fn, tn = tp.sum(dim=0), fp.sum(dim=0), fn.sum(dim=0), tn.sum(dim=0)
        tp, fp, fn, tn = tp.long(), fp.long(), fn.long(), tn.long()

        # Micro（整体）与 Macro（逐类平均）指标
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        macro_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # 前景宏平均：忽略背景类0
        per_class_f1  = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")

        if self.num_classes > 1:
            train_fg_macro_f1 = per_class_f1[1:].mean()
            train_fg_miou = per_class_iou[1:].mean()
            # 在进度条显示无背景（前景宏平均）F1 的 step 值
            self.log('train_fg_f1', train_fg_macro_f1, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_fg_miou', train_fg_miou, on_step=False, on_epoch=True, prog_bar=False)
            # 💡 记录每个类别的 F1 (Epoch 级)
            for i, name in enumerate(self.class_names):
                self.log(f'train_f1_cls_{i}_{name}', per_class_f1[i], on_epoch=True, prog_bar=False)
                self.log(f'train_iou_cls_{i}_{name}', per_class_iou[i], on_epoch=True, prog_bar=False)
        # 记录整体指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_macro_f1', macro_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_miou', miou, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        if torch.any(masks < 0) or torch.any(masks >= self.num_classes):
            print(f"Invalid mask values found: min={masks.min()}, max={masks.max()}")
            masks = torch.clamp(masks, 0, self.num_classes - 1)
        logits, probs = self(images)
        loss = self.joint_loss(logits, probs, masks)
        preds = torch.argmax(probs, dim=1)

        preds_long = cast(torch.LongTensor, preds.long())
        masks_long = cast(torch.LongTensor, masks.long())
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds_long, masks_long, mode='multiclass', num_classes=self.num_classes
        )
        # 💡 重要：聚合为 (C,) 确保 per_class_f1[i] 为标量，避免维度不匹配导致的 log 错误。
        tp, fp, fn, tn = tp.sum(dim=0).long(), fp.sum(dim=0).long(), fn.sum(dim=0).long(), tn.sum(dim=0).long()

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        macro_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # 前景宏平均：忽略背景类0，作为主监控指标
        per_class_f1  = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        per_class_acc = smp.metrics.accuracy(cast(torch.LongTensor, tp.long()), cast(torch.LongTensor, fp.long()), cast(torch.LongTensor, fn.long()), cast(torch.LongTensor, tn.long()), reduction="none")
        per_class_pre = smp.metrics.precision(cast(torch.LongTensor, tp.long()), cast(torch.LongTensor, fp.long()), cast(torch.LongTensor, fn.long()), cast(torch.LongTensor, tn.long()), reduction="none")
        per_class_rec = smp.metrics.recall(cast(torch.LongTensor, tp.long()), cast(torch.LongTensor, fp.long()), cast(torch.LongTensor, fn.long()), cast(torch.LongTensor, tn.long()), reduction="none")

        if self.num_classes > 1:
            val_fg_macro_f1 = per_class_f1[1:].mean()
            val_fg_miou = per_class_iou[1:].mean()
            
            # 💡 记录验证集每个类别的 F1 / IoU / Accuracy / Precision / Recall，便于定位模型在哪些 RFI 类型上较弱
            for i, name in enumerate(self.class_names):
                self.log(f'val_f1_cls_{i}_{name}', per_class_f1[i], on_epoch=True, prog_bar=False)
                self.log(f'val_iou_cls_{i}_{name}', per_class_iou[i], on_epoch=True, prog_bar=False)
                self.log(f'val_acc_cls_{i}_{name}', per_class_acc[i], on_epoch=True, prog_bar=False)
                self.log(f'val_pre_cls_{i}_{name}', per_class_pre[i], on_epoch=True, prog_bar=False)
                self.log(f'val_rec_cls_{i}_{name}', per_class_rec[i], on_epoch=True, prog_bar=False)
        else:
            val_fg_macro_f1 = per_class_f1.mean()
            val_fg_miou = per_class_iou.mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_fg_macro_f1', val_fg_macro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_fg_miou', val_fg_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_macro_f1', macro_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_miou', miou, on_step=False, on_epoch=True, prog_bar=False)
        
        # ====== 全局混淆矩阵收集 ======
        # flatten & detach to cpu
        preds_np = preds.detach().cpu().numpy().flatten()
        masks_np = masks.detach().cpu().numpy().flatten()
        # 过滤 ignore_index=255（如有）
        valid_mask = (masks_np != 255)
        if np.any(valid_mask):
            self._epoch_preds.append(preds_np[valid_mask])
            self._epoch_targets.append(masks_np[valid_mask])

        return loss
    def validation_epoch_end(self, outputs):
        """
        在每个epoch结束时，基于全量像素统计混淆矩阵，并计算IoU、F1、Precision、Recall等指标，保证与推理脚本一致。
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix

        if len(self._epoch_preds) == 0 or len(self._epoch_targets) == 0:
            return
        y_pred = np.concatenate(self._epoch_preds)
        y_true = np.concatenate(self._epoch_targets)
        # 清空缓存，防止内存泄漏
        self._epoch_preds.clear()
        self._epoch_targets.clear()

        num_classes = self.num_classes
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

        # 计算 per-class 指标
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        # 避免除零
        eps = 1e-9
        precision = (TP + eps) / (TP + FP + eps)
        recall = (TP + eps) / (TP + FN + eps)
        iou = (TP + eps) / (TP + FP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # log per-class指标
        for i, name in enumerate(self.class_names):
            self.log(f'val_global_f1_cls_{i}_{name}', f1[i], on_epoch=True, prog_bar=False)
            self.log(f'val_global_iou_cls_{i}_{name}', iou[i], on_epoch=True, prog_bar=False)
            self.log(f'val_global_pre_cls_{i}_{name}', precision[i], on_epoch=True, prog_bar=False)
            self.log(f'val_global_rec_cls_{i}_{name}', recall[i], on_epoch=True, prog_bar=False)

        # log 宏平均（不含背景）
        if num_classes > 1:
            fg_slice = slice(1, None)
            macro_f1 = f1[fg_slice].mean()
            macro_iou = iou[fg_slice].mean()
        else:
            macro_f1 = f1.mean()
            macro_iou = iou.mean()
        self.log('val_global_fg_macro_f1', macro_f1, on_epoch=True, prog_bar=True)
        self.log('val_global_fg_miou', macro_iou, on_epoch=True, prog_bar=True)

        # 可选：保存混淆矩阵到文件
        # np.save(f'confusion_matrix_epoch{self.current_epoch}.npy', cm)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler_config = self.get_scheduler_config(optimizer, scheduler_type=self.scheduler_type)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config,
        }
    
    def get_scheduler_config(self, optimizer, scheduler_type="cosine_warmup"):
        """
    获取学习率调度器配置（warmup + cosine annealing）。

        Args:
            optimizer: torch optimizer
            scheduler_type: 目前仅支持 "cosine_warmup"（线性 warmup -> 余弦退火）

        Returns:
            scheduler配置字典，供 Lightning 使用
        """

        # 当前项目中仅保留简单稳定的 warmup + cosine 调度策略
        if scheduler_type != "cosine_warmup":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type} (only 'cosine_warmup' allowed)")

        # 使用 1 个 epoch 的线性 warmup，再衔接余弦退火到训练结束。
        warmup_epochs = 1
        # 默认回退，但如果 Trainer 已附着到 model，使用 trainer.max_epochs 更精确
        total_epochs = 25
        if hasattr(self, 'trainer') and getattr(self.trainer, 'max_epochs', None) is not None:
            try:
                total_epochs = int(cast(int, self.trainer.max_epochs))
            except Exception:
                pass
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=5e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
        

class FITSDataset(Dataset):
    """
    This class implements a custom dataset for loading 8-bit grayscale images from 
    radio astronomy FITS files, and corresponding masks from PNG files. It also 
    performs data augmentation and preprocessing.
    and their corresponding masks.

    We assume that the dataset structure is as follows:
    dataset/
        ├── image/
        │   ├── train/
        │   │   ├── image1.fits
        │   │   ├── image2.fits
        │   │   └── ...
        │   └── val/
        │       ├── image1.fits
        │       ├── image2.fits
        │       └── ...
        └── mask/
            ├── train/
            │   ├── mask1.png
            │   ├── mask2.png
            │   └── ...
            └── val/
                ├── mask1.png
                ├── mask2.png
                └── ...
    The FITS files are expected to contain 2D grayscale images, and the masks are
    expected to be 2D Label-Encoding masks, where each pixel value corresponds to a class label.
    """

    # 默认类别列表（可被 __main__ 中的 CLASSES 覆盖）
    # 当前训练投入 6 类（含背景）：bkg, horizontal, vertical, point, block, pulsar
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block", "pulsar"]

    @staticmethod
    def extract_id(filename):
        match = re.search(r'(\d+)\.fits$', filename)
        return match.group(1) if match else None

    def __init__(
            self,
            image_dir,
            mask_dir,
            classes = None,
            augmentation = None,
            preprocessing = None,
            normalization_method = "percentile",  # 新增参数
            crop_size: int = 768,
            point_class_value: int = 6,
            point_oversample_factor: float = 3.0,
            point_crop_prob: float = 0.7,
    ):
        # ============ 新的配对策略：按文件基名匹配 ============
        all_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.fits')]
        all_mask_files  = [f for f in os.listdir(mask_dir)  if f.lower().endswith('.png')]

        image_basenames = {os.path.splitext(f)[0] for f in all_image_files}
        mask_basenames  = {os.path.splitext(f)[0] for f in all_mask_files}
        common_basenames = sorted(image_basenames & mask_basenames)

        if not common_basenames:
            print(f"[警告] 名称交集为空：image_dir={image_dir}, mask_dir={mask_dir}。检查命名或路径。")

        self.ids = common_basenames  # ids 使用无扩展的基名
        self.image_list = [os.path.join(image_dir, bn + '.fits') for bn in self.ids]
        self.mask_list  = [os.path.join(mask_dir,  bn + '.png')  for bn in self.ids]
        # =====================================================

        self.normalization_method = normalization_method  # 保存归一化方法
        self.crop_size = int(crop_size)
        self.point_class_value = int(point_class_value)
        self.point_oversample_factor = float(max(1.0, point_oversample_factor))
        self.point_crop_prob = float(np.clip(point_crop_prob, 0.0, 1.0))
        self.classes = classes if classes is not None else self.CLASSES
        self.class_mapping = {
            0: 0,  # bkg
            1: 1,  # horizontal
            2: 2,  # vertical
            6: 3,  # point
            7: 4,  # block
            8: 5,  # pulsar
        }
        print(f"Using class mapping: {self.class_mapping}")
        
        # 验证文件可读性（快速检测），不再重新构造配对，只报告坏文件
        readable_images = 0
        readable_masks = 0
        for img_path, msk_path in zip(self.image_list, self.mask_list):
            if os.path.exists(img_path):
                try:
                    with fitsio.FITS(img_path, 'r') as fits:
                        _ = fits[1].read_header()
                    readable_images += 1
                except Exception as e:
                    print(f"[警告] 图像不可读，跳过计数（仍保留）: {img_path} - {e}")
            else:
                print(f"[缺失] 图像文件不存在: {img_path}")
            if os.path.exists(msk_path):
                test_mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
                if test_mask is not None:
                    readable_masks += 1
                else:
                    print(f"[警告] 掩码不可读: {msk_path}")
            else:
                print(f"[缺失] 掩码文件不存在: {msk_path}")

        print(f"名称交集样本数: {len(self.ids)}，可读图像: {readable_images}，可读掩码: {readable_masks}")

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.sample_weights = self._build_sample_weights()

    def _build_sample_weights(self) -> np.ndarray:
        weights = np.ones(len(self.mask_list), dtype=np.float32)
        point_hits = 0
        if len(self.mask_list) == 0:
            return weights
        for idx, msk_path in enumerate(self.mask_list):
            try:
                mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
                if mask is not None and np.any(mask == self.point_class_value):
                    weights[idx] = self.point_oversample_factor
                    point_hits += 1
            except Exception:
                continue
        print(f"[PointOversample] 含 point 样本数: {point_hits}/{len(weights)}, oversample_factor={self.point_oversample_factor}")
        return weights

    def _point_focused_crop(self, image: np.ndarray, mask_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = mask_labels.shape[:2]
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        if h == crop_h and w == crop_w:
            return image, mask_labels

        point_index = self.class_mapping.get(self.point_class_value, 3)
        point_coords = np.argwhere(mask_labels == point_index)
        focus_on_point = point_coords.size > 0 and np.random.rand() < self.point_crop_prob

        if focus_on_point:
            center_y, center_x = point_coords[np.random.randint(0, len(point_coords))]
            top = int(np.clip(center_y - crop_h // 2, 0, max(0, h - crop_h)))
            left = int(np.clip(center_x - crop_w // 2, 0, max(0, w - crop_w)))
        else:
            top = 0 if h == crop_h else np.random.randint(0, h - crop_h + 1)
            left = 0 if w == crop_w else np.random.randint(0, w - crop_w + 1)

        return image[top:top + crop_h, left:left + crop_w], mask_labels[top:top + crop_h, left:left + crop_w]

    @staticmethod
    def normalize_image_mean_std(image: np.ndarray, k: float = 5.0) -> np.ndarray:
        """
        按照全图均值和±k倍标准差进行归一化到[0,1]，避免依赖任何标签信息。
        先将超出±kσ的像素值clamp到±kσ，然后线性缩放到[0,1]。
        当标准差为0时，返回常数0.5的图像以避免数值问题。
        """
        img = image.astype(np.float32)
        mean = float(img.mean())
        std = float(img.std())
        if std <= 0:
            return np.full_like(img, 0.5, dtype=np.float32)
        lo = mean - k * std
        hi = mean + k * std
        # 先clamp到[lo, hi]
        img_clamped = np.clip(img, lo, hi)
        # 然后线性缩放到[0,1]
        scaled = (img_clamped - lo) / (hi - lo)
        return scaled.astype(np.float32)

    @staticmethod
    def compute_class_weights(
        image_dir,
        mask_dir,
        classes,
        class_mapping,
        num_samples: int = 100,
        smoothing_method: str = 'sqrt',
        cap_factor: float = 3.0,
        min_floor: float = 0.1,
        ratio_cap: float = 1e3,
    sample_seed: Optional[int] = None,
    ):
        """
        计算训练集的类别权重，用于平衡类别不平衡。
        
        Args:
            image_dir: 图像目录路径
            mask_dir: 掩码目录路径
            classes: 类别列表
            class_mapping: 从掩码值到类别索引的映射
            num_samples: 采样数量，避免加载所有数据
        
        Returns:
            class_weights: 类别权重列表
        """
        import random
        
        # 获取所有图像文件并随机抽样（总是随机，哪怕数量不超过 num_samples）
        image_files_all = [f for f in os.listdir(image_dir) if f.endswith('.fits')]
        if sample_seed is not None:
            random.seed(sample_seed)
        # 抽样数量：不超过实际数量
        sample_k = min(num_samples, len(image_files_all))
        image_files = random.sample(image_files_all, sample_k)
        
        class_counts = np.zeros(len(classes))
        total_pixels = 0
        
        for image_file in image_files:
            # 与C侧一致：用图像基名替换扩展名为.png
            base_name, _ = os.path.splitext(image_file)
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            if not os.path.exists(mask_path):
                continue
            
            # 加载掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue

            # 根据映射关系统计每个最终类别索引的像素数
            for mask_val, class_idx in class_mapping.items():
                if class_idx < len(classes):  # 确保索引在范围内
                    class_pixels = np.sum(mask == mask_val)
                    class_counts[class_idx] += class_pixels
            
            total_pixels += mask.size
        
        # 原始权重：总像素 / (类别像素 * 类别数)
        raw_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_pixels / (count * len(classes))
            else:
                weight = 0.0
            raw_weights.append(weight)
        raw_weights = np.array(raw_weights, dtype=np.float64)

        # ============ 合并后的平滑 + 下限 + 中位数cap + 比率cap 逻辑 ============
        def apply_smoothing(arr: np.ndarray) -> np.ndarray:
            if smoothing_method == 'sqrt':
                return np.sqrt(arr + 1e-12)
            if smoothing_method == 'log':
                return np.log1p(arr)
            if smoothing_method == 'pow':
                return np.power(arr + 1e-12, 0.6)
            if smoothing_method == 'none':
                return arr
            print(f"[警告] 未知的 smoothing_method={smoothing_method}，回退为 'none'")
            return arr

        smoothed = apply_smoothing(raw_weights.copy())

        # 统一第一次 clip：确保非正/零样本得到 min_floor 下限
        smoothed[smoothed < min_floor] = min_floor

        # 计算基于中位数的上限
        if smoothed.size == 0:
            smoothed = np.ones(len(classes), dtype=np.float64) * min_floor
        median_val = float(np.median(smoothed)) if smoothed.size > 0 else 1.0
        upper_cap_median = median_val * cap_factor

        # 暂时按 median cap 截断
        smoothed = np.clip(smoothed, min_floor, upper_cap_median)

        # 比率 cap：使用当前下限(可能就是 min_floor) 计算最大允许值
        current_min = float(smoothed.min()) if smoothed.size > 0 else min_floor
        upper_cap_ratio = current_min * ratio_cap
        # 取两个上限中更严格的一个
        final_upper = min(upper_cap_median, upper_cap_ratio)
        if final_upper <= current_min:
            # 防御：避免出现 final_upper < lower 导致全等
            final_upper = current_min * 1.000001
        smoothed = np.clip(smoothed, current_min, final_upper)

        # 一步归一化到“平均为1”尺度：和==类别数
        total = smoothed.sum()
        if total <= 0:
            smoothed = np.ones(len(classes), dtype=np.float64)
        else:
            smoothed = smoothed * len(classes) / total

        max_min_ratio = (smoothed.max() / smoothed.min()) if smoothed.min() > 0 else float('inf')

        # 调试输出（合并说明）
        print(
            "[ClassWeights] 分布/处理摘要:\n"
            f"  像素计数: {dict(zip(classes, class_counts.astype(int)))}\n"
            f"  原始权重 raw: {np.array2string(raw_weights, precision=4)}\n"
            f"  处理后权重 processed (method={smoothing_method}, cap_factor={cap_factor}, ratio_cap={ratio_cap}):\n"
            f"    {np.array2string(smoothed, precision=4)}\n"
            f"  范围: min={smoothed.min():.4e}, max={smoothed.max():.4e}, max/min={max_min_ratio:.2f}"
        )

        return smoothed.tolist()

    @staticmethod
    def load_fits_image(fits_path):
        """
        从FITS文件加载原始图像数据，不进行归一化。
        约束：每个样本的 FITS 只能有 1 个 SUBINT 行。
        """
        # 使用更安全的FITS文件读取方式
        with fitsio.FITS(fits_path, 'r') as fits:
            fits_header = fits[1].read_header()
            fits_data = fits[1].read()

        nchan = int(fits_header["NCHAN"])  # 频道数

        # rows 应该为 1（约束）。如不为 1，抛出清晰错误，方便定位生成阶段问题。
        rows = fits_data.shape[0] if hasattr(fits_data, 'shape') else len(fits_data)
        if rows != 1:
            raise ValueError(f"Expected exactly 1 SUBINT row, but got {rows} in {os.path.basename(fits_path)}")

        raw = np.asarray(fits_data[0]["DATA"], dtype=np.uint8)
        total = int(raw.size)
        nsamp = total // nchan

        data = raw.reshape(nsamp, nchan).astype(np.float32)
        dat_scl = np.asarray(fits_data[0]["DAT_SCL"], dtype=np.float32)
        dat_offs = np.asarray(fits_data[0]["DAT_OFFS"], dtype=np.float32)

        # 注意：先乘后加
        data *= dat_scl[np.newaxis, :]
        data += dat_offs[np.newaxis, :]
        image = np.flipud(data.T)
        return image

    def __getitem__(self, i):
        try:
            raw_image = FITSDataset.load_fits_image(self.image_list[i])
            raw_mask = cv2.imread(self.mask_list[i], cv2.IMREAD_UNCHANGED)
            if raw_mask is None:
                raise IOError(f"无法读取掩码文件: {self.mask_list[i]}")

            # 进行类别索引映射
            mask_labels = np.zeros_like(raw_mask, dtype=np.uint8)
            for mask_val, class_idx in self.class_mapping.items():
                mask_labels[raw_mask == mask_val] = class_idx

            # 均值-标准差归一化
            image = FITSDataset.normalize_image_mean_std(raw_image, k=5.0)

            image, mask_labels = self._point_focused_crop(image, mask_labels)

            # 数据增强
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            # 预处理
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            # 某些增强会返回带有非标准/不可扩展底层 storage 的数组或张量。
            # 在 DataLoader worker 中 default_collate 组 batch 时，这类对象可能触发
            # "Trying to resize storage that is not resizable"。这里统一复制为连续内存。
            if isinstance(image, torch.Tensor):
                image = image.contiguous().clone()
            else:
                image = np.ascontiguousarray(image.copy())

            if isinstance(mask_labels, torch.Tensor):
                mask_labels = mask_labels.contiguous().clone()
            else:
                mask_labels = np.ascontiguousarray(mask_labels.copy())

            return image, mask_labels
            
        except Exception as e:
            print(f"Error reading file {self.image_list[i]}: {e}")
            return self.__getitem__((i + 1) % len(self.ids))
    
    def __len__(self):
        return len(self.ids)
    
def get_stable_training_augmentation(size: int = 640):
    """
    点源保护增强策略：禁用 Resize 以防止能量弥散。
    使用 PadIfNeeded + CenterCrop 确保尺寸对齐。
    """
    train_transform = [
        # 几何变换（不改变分辨率）
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        
        # 强力抗过拟合
        albu.CoarseDropout(
            num_holes_range=(1, 6), 
            hole_height_range=(4, 16), 
            hole_width_range=(4, 16), 
            fill=0, 
            p=0.2
        ),

        # 图像质量
        albu.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        albu.RandomBrightnessContrast(p=0.5),
        
        # 核心：使用 Pad + RandomCrop 替代 Resize，提升空间多样性并减轻中心偏置
        albu.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        albu.RandomCrop(size, size)
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    # 检查数据类型来区分图像和掩码
    if x.dtype in [np.uint8, np.int32, np.int64]:  # 标签掩码
        x = np.ascontiguousarray(x.astype(np.int64, copy=True))
        return torch.from_numpy(x)
    else:  # 图像
        if len(x.shape) == 2:  # (H, W) -> (1, H, W)
            x = np.expand_dims(x, axis=0)
        x = np.ascontiguousarray(x.astype(np.float32, copy=True))
        return torch.from_numpy(x)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_validation_augmentation(height=640, width=640):
    """Validation augmentation: Pad to target size instead of resizing."""
    test_transform = []
    if height is not None and width is not None:
        test_transform.append(albu.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT))
    return albu.Compose(test_transform)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始 U-Net (MiT-B2) 射电干扰分割训练脚本")
    print("=" * 60)
    
    dataset_top_dir = "/home/cbm/deRFI/Datasets/SynthesizedDataset"
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    train_image_dir = os.path.join(image_dir, "train")
    train_mask_dir = os.path.join(mask_dir, "train")
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    # ============ 超参数配置 ============
    # 将与硬件无关的模型/数据超参数放在顶层共享配置（便于与硬件相关参数分离）
    # 这些参数与训练机器（local/server）无关，便于统一修改和版本控制
    ENCODER = "mit_b2"
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block", "pulsar"]
    SCHEDULER_TYPE = "cosine_warmup"
    FOCAL_GAMMA = 3.0
    LOSS_ALPHA = 0.4  # 略回调 FN 权重，兼顾 point recall 与 pulsar 完整性
    LOSS_BETA = 0.6   # 保持对 FP 的抑制，但不要过度压制碎弱前景
    NORMALIZATION_METHOD = "median_sigma"
    TRAIN_CROP_SIZE = 640
    POINT_CLASS_NAME = "point"
    PULSAR_CLASS_NAME = "pulsar"
    POINT_WEIGHT_MULTIPLIER = 5.0  # 略收一点 point 偏置，给 pulsar 留容量
    PULSAR_WEIGHT_MULTIPLIER = 1.5  # 适度补偿 pointEnhance 稀释后的 pulsar 暴露
    POINT_OVERSAMPLE_FACTOR = 3.0
    POINT_CROP_PROB = 0.6
    POINT_AUX_LOSS_WEIGHT = 0.25
    TRAIN_SAMPLES_PER_EPOCH = 16384  # 提升每轮覆盖度，保留更多 pulsar/point 难例

    # 使用两套硬编码的配置集（`local` / `server`），通过 CONFIG_MODE 切换
    # local: 轻量级开发机配置；server: 高并发、多核、多GPU 训练服务器配置
    CONFIG_MODE = "local"  # 修改为 'local' 以适配 4060 显卡配置

    # 每套配置只包含与硬件/运行相关的参数（不再包含模型结构或损失超参）
    configs = {
        "local": {
            "LEARNING_RATE": 1.8e-4,
            "BATCH_SIZE": 4,
            "NUM_WORKERS": 8,
            "MAX_EPOCHS": 27,
            "ACCUMULATE_GRAD_BATCHES": 2,
            "USE_COMPILE": False,
            "PRECISION": "16-mixed",
            "DEVICES": 1,
        },
        "server": {
            "LEARNING_RATE": 2e-4,
            "BATCH_SIZE": 16,
            # 为避免把所有 CPU 核心占满，保留 1 个核给系统
            "NUM_WORKERS": 2,
            "MAX_EPOCHS": 100,
            "ACCUMULATE_GRAD_BATCHES": 8,
            # 在老 PyTorch/CUDA 环境上禁用 torch.compile，以免不兼容
            "USE_COMPILE": False,
            # V100 在比较稳定的环境下推荐使用 16-mixed
            "PRECISION": "16-mixed",
            "DEVICES": 1,  # 如果你想使用所有可见 GPU，设置为 torch.cuda.device_count() 或具体数字
        }
    }

    cfg = configs.get(CONFIG_MODE, configs["local"])
    LEARNING_RATE = cfg["LEARNING_RATE"]
    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_WORKERS = cfg["NUM_WORKERS"]
    MAX_EPOCHS = cfg["MAX_EPOCHS"]
    ACCUMULATE_GRAD_BATCHES = cfg["ACCUMULATE_GRAD_BATCHES"]
    USE_COMPILE = cfg.get("USE_COMPILE", False)
    PRECISION = cfg.get("PRECISION", 32)
    DEVICES = cfg.get("DEVICES", 1)
    RESUME_CKPT_PATH = "/home/cbm/deRFI/checkpoints/train/best_model-epoch=12-fgmIoU_val_fg_miou=0.6668.ckpt"
    # 预训练权重路径 (None表示随机初始化)
    # WEIGHTS_PATH = "/home/cbm/deRFI/pruned_best_model-epoch=21-fgF1_val_fg_macro_f1=0.7905.pt"
    WEIGHTS_PATH = None
    # ====================================
    print("📊 创建训练和验证数据集...")
    train_dataset = FITSDataset(
        train_image_dir,
        train_mask_dir,
        classes=CLASSES,
        augmentation=get_stable_training_augmentation(TRAIN_CROP_SIZE),
        preprocessing=get_preprocessing(None),
        normalization_method=NORMALIZATION_METHOD,
        crop_size=TRAIN_CROP_SIZE,
        point_oversample_factor=POINT_OVERSAMPLE_FACTOR,
        point_crop_prob=POINT_CROP_PROB,
    )
    print(f"✅ 训练数据集: {len(train_dataset)} 个样本")

    print("⚖️ 计算类别权重...")
    class_weights = FITSDataset.compute_class_weights(
        train_image_dir, 
        train_mask_dir, 
        CLASSES, 
        class_mapping=train_dataset.class_mapping,
        num_samples=200)
    point_class_index = CLASSES.index(POINT_CLASS_NAME)
    pulsar_class_index = CLASSES.index(PULSAR_CLASS_NAME)
    class_weights[point_class_index] *= POINT_WEIGHT_MULTIPLIER
    class_weights[pulsar_class_index] *= PULSAR_WEIGHT_MULTIPLIER
    print(f"[ClassWeights] point 类权重额外乘以 {POINT_WEIGHT_MULTIPLIER:.2f}，调整后={class_weights[point_class_index]:.4f}")
    print(f"[ClassWeights] pulsar 类权重额外乘以 {PULSAR_WEIGHT_MULTIPLIER:.2f}，调整后={class_weights[pulsar_class_index]:.4f}")

    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(TRAIN_CROP_SIZE, TRAIN_CROP_SIZE),
        preprocessing=get_preprocessing(None),
        normalization_method=NORMALIZATION_METHOD,
        crop_size=TRAIN_CROP_SIZE,
        point_oversample_factor=1.0,
        point_crop_prob=0.0)
    
    print(f"✅ 验证数据集: {len(val_dataset)} 个样本")

    print("🔍 检查第一个验证样本的mask类别编号...")
    if len(val_dataset) > 0:
        image, mask = val_dataset[0]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
            
        unique_values = np.unique(mask_np)
        print(f"第一个验证样本的mask唯一值（类别编号）: {unique_values}")
        print(f"对应的类别名称: {[CLASSES[i] if i < len(CLASSES) else f'unknown_{i}' for i in unique_values]}")
        
        for val in unique_values:
            count = np.sum(mask_np == val)
            percentage = (count / mask_np.size) * 100
            class_name = CLASSES[val] if val < len(CLASSES) else f'unknown_{val}'
            print(f"  {class_name} (编号{val}): {count} 像素 ({percentage:.2f}%)")
    else:
        print("验证数据集为空")

    print("🔄 创建数据加载器...")
    use_persistent_workers = NUM_WORKERS > 0
    train_sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=TRAIN_SAMPLES_PER_EPOCH,
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # 加速GPU数据传输
        persistent_workers=use_persistent_workers,  # 保持worker进程，提升 epoch 切换速度
        prefetch_factor=2,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2)

    print(f"✅ 训练数据加载器: {len(train_loader)} 个批次 (batch_size={BATCH_SIZE})")
    print(f"✅ 验证数据加载器: {len(val_loader)} 个批次 (batch_size={BATCH_SIZE})")
    print(f"✅ 每轮训练采样数: {TRAIN_SAMPLES_PER_EPOCH} (Dataset总数: {len(train_dataset)})")

    print("🏗️ 创建 U-Net 模型...")
    model = UNetLightningModule(
        encoder_name=ENCODER,
        classes=len(CLASSES),
        class_names=CLASSES,
        learning_rate=LEARNING_RATE,
        scheduler_type=SCHEDULER_TYPE,
        class_weights=class_weights,
        focal_gamma=FOCAL_GAMMA,
        loss_alpha=LOSS_ALPHA,
        loss_beta=LOSS_BETA,
        use_compile=USE_COMPILE,
        ft_warmup_epochs=2,
        point_class_index=point_class_index,
        point_aux_weight=POINT_AUX_LOSS_WEIGHT,
    )
    
    print(f"✅ 模型配置: {ENCODER} 编码器, {len(CLASSES)} 个类别 ({CLASSES})")
    # 使用 SegFormer，不需要 MobileMamba 特殊提示
    print(f"✅ 学习率: {LEARNING_RATE}, 批次大小: {BATCH_SIZE}, 梯度累积: {ACCUMULATE_GRAD_BATCHES}步")
    print(f"✅ 有效批次大小: {BATCH_SIZE * ACCUMULATE_GRAD_BATCHES}, 最大轮次: {MAX_EPOCHS}")
    print(f"✅ 恢复训练检查点: {RESUME_CKPT_PATH if os.path.exists(RESUME_CKPT_PATH) else 'None'}")
    print(f"✅ GPU: RTX 4060 Laptop 8GB - 已优化显存使用")
    
    # 设置示例输入数组以便 TensorBoard 记录计算图
    model.example_input_array = torch.randn(1, 1, TRAIN_CROP_SIZE, TRAIN_CROP_SIZE)
    
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints/train',
            filename='best_model-{epoch:02d}-fgmIoU_{val_fg_miou:.4f}',
            monitor='val_fg_miou',
            mode='max',
            save_top_k=3,
            save_last=False,  # 不再保存 last-vX.ckpt 文件
            verbose=True
        ),
        EarlyStopping(
            monitor='val_fg_miou',
            patience=10,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 配置 TensorBoard 日志记录器
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir='training_logs',  # 写到项目根的 training_logs 目录
        name='',                  # 取消额外的子目录（不再创建 'unetpp_training' 层）
        version=None,  # 自动递增版本号
        log_graph=False,  # 关闭图记录，避免与 torch.compile/FX 冲突
        default_hp_metric=False  # 避免默认超参数指标
    )
    
    print("⚙️ 配置训练器...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=DEVICES,
        callbacks=callbacks,
        logger=logger,  # 显式指定 TensorBoard 日志记录器
    log_every_n_steps=50,  # 进一步减少日志同步频率，提高吞吐
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision=PRECISION,
        enable_checkpointing=True,
        enable_model_summary=True,
        gradient_clip_val=2.0,  # 添加梯度裁剪防止训练不稳定
        deterministic=False,  # 允许非确定性操作以提高性能
        strategy='auto',
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,  # 梯度累积
        inference_mode=False,  # 避免在验证中使用推理模式的潜在图捕获
    )
    
    print("🚀 开始训练...")
    print("=" * 60)
    # ====== Load pruned weights if provided ======
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        print(f"[Weights] Loading weights from: {WEIGHTS_PATH}")
        try:
            loaded = torch.load(WEIGHTS_PATH, map_location="cpu")
            # Lightning .ckpt has a key 'state_dict'
            if isinstance(loaded, dict) and "state_dict" in loaded:
                state = loaded["state_dict"]
            else:
                state = loaded

            # 简单检查：如果当前选择 MobileMamba，但 checkpoint key 名称看起来像 SegFormer，则提醒
            try:
                if "mobilemamba" in ENCODER.lower() and isinstance(state, dict):
                    sample_keys = list(state.keys())[:30]
                    if any("segformer" in k.lower() or "mit" in k.lower() or "pixel" in k.lower() for k in sample_keys):
                        print("[Weights] Warning: the checkpoint appears to be for a SegFormer-like model and may not match MobileMamba backbone.")
            except Exception:
                pass

            model.load_state_dict(state, strict=False)
            print("[Weights] Loaded state_dict (strict=False)")
        except Exception as e:
            try:
                print(f"[Weights] Warning: direct load failed: {e}. Trying Lightning load_from_checkpoint...")
                model = UNetLightningModule.load_from_checkpoint(WEIGHTS_PATH, strict=False)
                print("[Weights] Loaded checkpoint via Lightning load_from_checkpoint.")
            except Exception as e2:
                print(f"[Weights] Failed to load weights: {e2}")
    # =============================================
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=RESUME_CKPT_PATH if os.path.exists(RESUME_CKPT_PATH) else None,
    )
    
    # 保存最终模型
    print("=" * 60)
    print("🎉 训练完成！模型已保存至checkpoints")