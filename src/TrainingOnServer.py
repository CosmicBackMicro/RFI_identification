import cv2
# 优化点 1: 禁用 OpenCV 的内部多线程，防止在多进程环境下线程数溢出
cv2.setNumThreads(0)
import os, re

# 彻底限制底层库的线程数，防止 DDP 模式下线程爆炸导致 "Resource temporarily unavailable"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import fitsio
import numpy as np
import gc  # 导入垃圾回收
from typing import Any, cast, Optional, List


import sys

# 避免 cudagraph 池相关错误
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 这行必须在导入albumentations之前
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # 缓解显存碎片化
os.environ['SMP_HUB_MODE'] = "original"
os.environ["TORCHINDUCTOR_USE_CUDA_GRAPH"] = "0" 
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
os.environ["TORCHINDUCTOR_USE_CUDAGRAPHS"] = "0"
os.environ["TORCH_CUDAGRAPHS"] = "0"

import albumentations as albu
import torch
import torch.multiprocessing as mp
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
    # 禁用 Flash / Mem-Efficient SDPA，避免某些环境下与 CUDA graphs/池的交互问题
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
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
    def __init__(self, encoder_name="segformer-b2", classes=2, learning_rate=0.0001, scheduler_type="cosine_warmup", class_weights=None, focal_gamma: float = 2.0, loss_alpha: float = 0.7, loss_beta: float = 0.3, ce_weight: float = 0.5, ft_weight: float = 0.5, use_compile: bool = False, ft_warmup_epochs: int = 5, class_names: Optional[List[str]] = None):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_type = scheduler_type
        self.num_classes = classes
        self.use_compile = use_compile
        self.ft_warmup_epochs = ft_warmup_epochs  # 前若干epoch逐步增加FT权重，稳定早期训练

        # 始终使用 SegFormer 作为 backbone（已移除 MobileMamba 支持）
        self.backbone_type = "segformer"
        enc = (encoder_name or "").lower()
        # 当用户选择含 b2 时使用较大配置，否则使用默认小型配置
        if "b2" in enc:
            config = SegformerConfig(
                num_labels=classes,
                num_channels=1,
                image_size=512,
                hidden_sizes=[64, 128, 320, 512],
                depths=[3, 4, 6, 3],
                num_attention_heads=[1, 2, 5, 8],
                intermediate_sizes=[256, 512, 1280, 2048],
                decoder_hidden_size=768,
            )
        else:
            config = SegformerConfig(
                num_labels=classes,
                num_channels=1,
                image_size=512,
            )
        self.model = SegformerForSemanticSegmentation(config)

        # 类别权重（用于缓解类别不平衡）
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.learning_rate = learning_rate

        # 交叉熵和FocalTversky
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
        # 用于记录每个类别的 FocalTversky Loss（不做 reduction）
        self.focal_tversky_perclass = FocalTverskyLoss(
            num_classes=self.num_classes,
            alpha=loss_alpha,
            beta=loss_beta,
            gamma=focal_gamma,
            smooth=1e-6,
            label_smoothing=0.01,
            reduction='none'
        )

        # 可选：使用 torch.compile 编译以减少调度开销
        if self.backbone_type == "segformer" and use_compile and hasattr(torch, "compile"):
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
        组合损失：像素级交叉熵（cross-entropy on logits） + FocalTversky（对小目标/细节更敏感）。

        Args:
            logits: (N, C, H, W) 未归一化网络输出（已上采样到输入分辨率）
            probs:  (N, C, H, W) softmax 后的概率
            y_true: (N, H, W) 真实标签

        Returns:
            加权损失标量
        """
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        device = logits.device

        # 准备类别权重用于交叉熵（如果有的话）
        ce_weight = None
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            ce_weight = self.class_weights.to(device)

        # 直接对 logits 计算 cross_entropy，数值上更稳定并支持 class weights
        nll = F.cross_entropy(logits, y_true, weight=ce_weight)

        ft = self.focal_tversky(probs, y_true, class_weights=self.class_weights)

        # 前若干个epoch对 FocalTversky 进行暖启动，减少早期优化不稳定
        try:
            epoch = int(self.current_epoch)
        except Exception:
            epoch = 0
        scale = 1.0
        if isinstance(self.ft_warmup_epochs, int) and self.ft_warmup_epochs > 0:
            scale = min(1.0, max(0.0, epoch / float(self.ft_warmup_epochs)))

        loss = self.ce_weight * nll + (self.ft_weight * scale) * ft

        # 不单独记录CE和FocalTversky子项，仅记录总loss（train_loss/val_loss）

        return loss
    
    def forward(self, x):
        # 仅保留 SegFormer 路径
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (N, C, h, w) 可能与输入分辨率不同
        # 始终上采样到输入分辨率以与标签对齐
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

        probs = torch.softmax(logits, dim=1)
        # 返回 logits 与 probs，方便使用基于 logits 的像素损失（CE）与基于概率的结构损失（FocalTversky）
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
        # Micro（整体）与 Macro（逐类平均）指标
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")  # micro IoU
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")    # micro F1
        miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")  # macro IoU (mIoU)
        macro_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        # 前景宏平均：忽略背景类0
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        per_class_f1  = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        if self.num_classes > 1:
            train_fg_macro_f1 = per_class_f1[1:].mean().item()
            train_fg_macro_iou = per_class_iou[1:].mean().item()
            # 在进度条显示无背景（前景宏平均）F1 的 step 值
            self.log('train_fg_macro_f1', train_fg_macro_f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            # 其余保持为 epoch 级统计
            self.log('train_fg_macro_iou', train_fg_macro_iou, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 记录每个类别的 FocalTversky loss（按 epoch 汇总）
        try:
                per_class_ft = self.focal_tversky_perclass(probs, masks, class_weights=self.class_weights)
                for c, v in enumerate(per_class_ft):
                    name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                    self.log(f"train_ft_{name}", v.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass

        # 记录整体指标
        self.log('train_loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_iou', iou.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_f1', f1.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_miou', miou.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_macro_f1', macro_f1.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_accuracy', accuracy.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

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
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")      # micro IoU
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")        # micro F1
        miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")      # macro IoU (mIoU)
        macro_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")   # macro F1
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        # 前景宏平均：忽略背景类0，作为主监控指标
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        per_class_f1  = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        if self.num_classes > 1:
            val_fg_macro_f1 = per_class_f1[1:].mean().item()
            val_fg_macro_iou = per_class_iou[1:].mean().item()
        else:
            val_fg_macro_f1 = per_class_f1.mean().item()
            val_fg_macro_iou = per_class_iou.mean().item()

        self.log('val_fg_macro_f1', val_fg_macro_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_fg_macro_iou', val_fg_macro_iou, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 记录整体指标
        self.log('val_loss', loss.detach().item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # 记录每个类别的 FocalTversky Loss（按 epoch 汇总）
        try:
            per_class_ft = self.focal_tversky_perclass(probs, masks, class_weights=self.class_weights)
            for c, v in enumerate(per_class_ft):
                name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                self.log(f'val_ft_{name}', v.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass
        self.log('val_iou', iou.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_miou', miou.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_macro_f1', macro_f1.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
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
        获取学习率调度器配置（精简版，当前仅保留 "cosine_warmup"）。

        Args:
            optimizer: torch optimizer
            scheduler_type: 目前仅支持 "cosine_warmup"（线性 warmup -> 余弦退火）

        Returns:
            scheduler配置字典，供 Lightning 使用
        """

        # 目前项目中只使用 "cosine_warmup"，其它分支已移除以简化行为
        if scheduler_type != "cosine_warmup":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type} (only 'cosine_warmup' allowed)")

        # 线性warmup若干epoch，然后余弦退火（不重启），更平滑
        warmup_epochs = 5
        # 默认回退为 100，但如果 Trainer 已附着到 model，使用 trainer.max_epochs 更精确
        total_epochs = 100
        if hasattr(self, 'trainer') and getattr(self.trainer, 'max_epochs', None) is not None:
            try:
                total_epochs = int(cast(int, self.trainer.max_epochs))
            except Exception:
                pass
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6)
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
    # 当前训练投入 5 类（含背景）：bkg, horizontal, vertical, point, block
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block"]

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
        self.classes = classes if classes is not None else self.CLASSES
        self.class_mapping = {
            0: 0,  # bkg
            1: 1,  # horizontal
            2: 2,  # vertical
            6: 3,  # point
            7: 4,  # block
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
            
            # 释放原始掩码内存
            del raw_mask

            # 均值-标准差归一化
            image = FITSDataset.normalize_image_mean_std(raw_image, k=5.0)
            del raw_image # 释放原始图像内存

            # 数据增强
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            # 预处理
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            return image, mask_labels
            
        except Exception as e:
            print(f"Error reading file {self.image_list[i]}: {e}")
            # 发生错误时尝试清理
            if 'raw_image' in locals(): del raw_image
            if 'raw_mask' in locals(): del raw_mask
            return self.__getitem__((i + 1) % len(self.ids))
    
    def __len__(self):
        return len(self.ids)
    
def get_stable_training_augmentation():
    """
    稳定的射电天文数据增强策略 - 平衡性能和质量的版本
    """
    train_transform = [
        albu.RandomScale(scale_limit=(0.8, 1.2), p=0.3),
        albu.Resize(512, 512),  # 下采样到512x512
        albu.Affine(
            scale=(0.9, 1.1),
            translate_percent=0,
            rotate=0,
            p=0.25),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.15),
        albu.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.2,
            p=0.5),
        albu.RandomGamma(gamma_limit=(90, 110), p=0.3),
        albu.GaussNoise(
            std_range=(0.005, 0.025),
            mean_range=(0, 0),
            per_channel=True,
            p=0.3),
        albu.GaussianBlur(blur_limit=(1, 2), p=0.2),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    # 检查数据类型来区分图像和掩码
    if x.dtype in [np.uint8, np.int32, np.int64]:  # 标签掩码
        return torch.from_numpy(x.astype(np.int64))
    else:  # 图像
        if len(x.shape) == 2:  # (H, W) -> (1, H, W)
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x.astype(np.float32))

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_validation_augmentation(height, width):
    """Validation resize to target size"""
    test_transform = [
        albu.Resize(height, width)
    ]
    return albu.Compose(test_transform)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization errors in DDP
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("=" * 60)
    print("🚀 开始 U-Net 射电干扰分割训练脚本")
    print("=" * 60)
    
    dataset_top_dir = "/home/bmcao/deRFI/Datasets/Dataset_G28.58+3.81_20220914_4classes_3.0_3.0_Downsamp1"
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    train_image_dir = os.path.join(image_dir, "train")
    train_mask_dir = os.path.join(mask_dir, "train")
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    # ============ 超参数配置 ============
    # 使用 SegFormer 作为默认编码器（回退到 SegFormer-only）
    ENCODER = "segformer-b2"
    CLASSES = ["bkg", "horizontal", "vertical", "point", "block"]
    LEARNING_RATE = 2e-4  # Prune微调学习率 * 0.1（原先5e-5）
    BATCH_SIZE = 16 # 减小 Batch Size 以解决 V100S 上的 OOM 问题
    NUM_WORKERS = 2 # 进一步降低 Worker 数量以节省系统内存 (RAM)
    MAX_EPOCHS = 100
    ACCUMULATE_GRAD_BATCHES = 8  # 增加梯度累积步数，保持有效 Batch Size 为 128 (16*8)
    SCHEDULER_TYPE = "cosine_warmup"  # 线性 warmup -> 余弦退火

    FOCAL_GAMMA = 2.0  # Focal Loss
    LOSS_ALPHA = 0.7  # FocalTversky
    LOSS_BETA = 0.3   # FocalTversky
    NORMALIZATION_METHOD = "median_sigma"
    # 预训练权重路径 (None表示随机初始化)
    # WEIGHTS_PATH = "/home/cbm/deRFI/pruned_best_model-epoch=21-fgF1_val_fg_macro_f1=0.7905.pt"
    WEIGHTS_PATH = None
    # ====================================
    print("📊 创建训练和验证数据集...")
    train_dataset = FITSDataset(
        train_image_dir,
        train_mask_dir,
        classes=CLASSES,
        augmentation=get_stable_training_augmentation(),
        preprocessing=get_preprocessing(None),
        normalization_method=NORMALIZATION_METHOD,
    )
    print(f"✅ 训练数据集: {len(train_dataset)} 个样本")

    print("⚖️ 计算类别权重...")
    class_weights = FITSDataset.compute_class_weights(
        train_image_dir, 
        train_mask_dir, 
        CLASSES, 
        class_mapping=train_dataset.class_mapping,
        num_samples=200)

    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(512, 512),
        preprocessing=get_preprocessing(None),
        normalization_method=NORMALIZATION_METHOD)
    
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,  # 加速GPU数据传输
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2)

    print(f"✅ 训练数据加载器: {len(train_loader)} 个批次 (batch_size={BATCH_SIZE})")
    print(f"✅ 验证数据加载器: {len(val_loader)} 个批次 (batch_size={BATCH_SIZE})")

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
        use_compile=False,
    )
    
    print(f"✅ 模型配置: {ENCODER} 编码器, {len(CLASSES)} 个类别 ({CLASSES})")
    # 使用 SegFormer，不需要 MobileMamba 特殊提示
    print(f"✅ 学习率: {LEARNING_RATE}, 批次大小: {BATCH_SIZE}, 梯度累积: {ACCUMULATE_GRAD_BATCHES}步")
    print(f"✅ 有效批次大小: {BATCH_SIZE * ACCUMULATE_GRAD_BATCHES}, 最大轮次: {MAX_EPOCHS}")
    print(f"✅ GPU: RTX 4060 Laptop 8GB - 已优化显存使用")
    
    # 设置示例输入数组以便 TensorBoard 记录计算图
    model.example_input_array = torch.randn(1, 1, 512, 512)
    
    # Get project root (one level up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(project_root, 'checkpoints', 'train'),
            filename='best_model-{epoch:02d}-fgF1_{val_fg_macro_f1:.4f}',
            monitor='val_fg_macro_f1',
            mode='max',
            save_top_k=3,
            save_last=False,  # 不再保存 last-vX.ckpt 文件
            verbose=True
        ),
        EarlyStopping(
            monitor='val_fg_macro_f1',
            patience=10,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 配置 TensorBoard 日志记录器
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir=os.path.join(project_root, 'training_logs'),  # 写到项目根的 training_logs 目录
        name='',                  # 取消额外的子目录（不再创建 'unetpp_training' 层）
        version=None,  # 自动递增版本号
        log_graph=False,  # 关闭图记录，避免与 torch.compile/FX 冲突
        default_hp_metric=False  # 避免默认超参数指标
    )
    
    print("⚙️ 配置训练器...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=2,
        callbacks=callbacks,
        logger=logger,  # 显式指定 TensorBoard 日志记录器
        log_every_n_steps=10,  # 减少日志频率以提高性能
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        enable_checkpointing=True,
        enable_model_summary=True,
        gradient_clip_val=2.0,  # 添加梯度裁剪防止训练不稳定
        deterministic=False,  # 允许非确定性操作以提高性能
        strategy='ddp_find_unused_parameters_true',
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
        ckpt_path=None,  # 这里设置None，Scheduler会从头开始，如果是微调，权重已在前面加载
    )
    
    # 保存最终模型
    print("=" * 60)
    print("🎉 训练完成！模型已保存至checkpoints")

# 需要减少内存和CPU侧的占用，尽量利用GPU