import cv2
import os, re
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, cast
import albumentations as albu

import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['SMP_HUB_MODE'] = "original"

import torch.nn as nn
import torch.nn.functional as F


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
        ignore_index: int | None = None,
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
        y_pred: torch.Tensor,  # (N, C, H, W) probabilities
        y_true: torch.Tensor,  # (N, H, W) int64 labels
        class_weights: torch.Tensor | None = None,  # (C,)
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

        # Apply optional class weights (to reduce background dominance)
        if class_weights is not None:
            # Ensure on same device and proper dtype
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

class UNetLightningModule(pl.LightningModule):
    def __init__(self, encoder_name="resnet50", classes=2, learning_rate=0.0001, weights_path=None, scheduler_type="cosine", class_weights=None, focal_gamma: float = 2.0):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_type = scheduler_type
        self.num_classes = classes
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=1,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_norm="batchnorm",
            decoder_attention_type="scse",
            decoder_interpolation="bilinear",
            classes=classes,
            activation='softmax2d'  # 显式在通道维度做softmax，直接输出概率，避免警告
        )
        
        # 如果有预训练权重，加载它们
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weights_path}")
            if missing:
                print(f"Missing keys: {len(missing)} (e.g., {missing[:5]})")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} (e.g., {unexpected[:5]})")
        
        # 类别权重（用于缓解类别不平衡）
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        # 新的损失函数：Focal Tversky（接受概率输入，类别不平衡友好，抗噪）
        # 可调整 alpha/beta 控制 FP/FN 权衡；gamma>1 聚焦难例；label_smoothing 提升噪声鲁棒
        self.focal_tversky = FocalTverskyLoss(
            num_classes=self.num_classes,
            alpha=0.8,
            beta=0.2,
            gamma=1.5,
            smooth=1e-6,
            reduction='mean',
            ignore_index=None,
            label_smoothing=0.05,
        )

        # 现阶段按你的要求先使用 DiceLoss（多类、概率输入），便于与常见基线对照
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=False,
            ignore_index=None,
            smooth=1e-6,
        )

        self.learning_rate = learning_rate
        
    def joint_loss(self, y_pred, y_true):
        """统一的损失入口：直接使用 Focal Tversky Loss（带类别权重）。"""
        if y_true.dtype != torch.long:
            y_true = y_true.long()
        return self.focal_tversky(y_pred, y_true, class_weights=self.class_weights)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        loss = self.joint_loss(outputs, masks)

        # outputs 已经是概率（由于 activation='softmax'）
        probs = outputs
        preds = torch.argmax(probs, dim=1)
        
        preds_long = cast(torch.LongTensor, preds.long())
        masks_long = cast(torch.LongTensor, masks.long())
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds_long, masks_long, mode='multiclass', num_classes=self.num_classes
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        if torch.any(masks < 0) or torch.any(masks >= self.num_classes):
            print(f"Invalid mask values found: min={masks.min()}, max={masks.max()}")
            masks = torch.clamp(masks, 0, self.num_classes - 1)
        
        outputs = self(images)
        
        loss = self.joint_loss(outputs, masks)

        # outputs 已经是概率（由于 activation='softmax'）
        probs = outputs
        preds = torch.argmax(probs, dim=1)
        
        preds_long = cast(torch.LongTensor, preds.long())
        masks_long = cast(torch.LongTensor, masks.long())
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds_long, masks_long, mode='multiclass', num_classes=self.num_classes
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self) -> Any:  # type: ignore[override]
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
    
    def get_scheduler_config(self, optimizer, scheduler_type="cosine"):
        """
        获取不同类型的学习率调度器配置
        
        Args:
            optimizer: 优化器
            scheduler_type: 调度器类型 ("cosine", "onecycle", "plateau", "step")
            
        Returns:
            scheduler配置字典
        """
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,  # 初始周期长度
                T_mult=2,  # 每次重启后周期长度的倍数
                eta_min=1e-6,  # 最小学习率
            )
            return {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
            
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3,
                verbose="True",
                min_lr=1e-7
            )
            return {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
            
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=15,  # 每15个epoch降低学习率
                gamma=0.5,     # 降低到原来的50%
            )
            return {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
            
        elif scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,  # 每个epoch乘以0.95
            )
            return {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

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

    # CLASSES = ["background", "periodic", "blob", "pulse"]
    CLASSES = ["bkg", "chan_rfi", "point_rfi"]  # Define your classes here

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
        # 获取文件名并按 block 编号排序
        all_files = os.listdir(image_dir)
        def _block_key(x: str) -> int:
            m = re.search(r'block(\d+)', x)
            return int(m.group(1)) if m else 0
        self.ids = sorted(all_files, key=_block_key)
        self.image_list = [os.path.join(image_dir, fid) for fid in self.ids]
        self.mask_list = []
        self.normalization_method = normalization_method  # 保存归一化方法
        
        # 验证文件完整性
        valid_pairs = []
        for fid in self.ids:
            file_id = self.extract_id(fid)
            if file_id:
                mask_path = os.path.join(mask_dir, f"mask_merged_{file_id}.png")
                image_path = os.path.join(image_dir, fid)
                
                # 检查文件是否存在且可读
                if os.path.exists(mask_path) and os.path.exists(image_path):
                    try:
                        # 快速检查FITS文件是否可读
                        with fitsio.FITS(image_path, 'r') as fits:
                            _ = fits[1].read_header()
                        # 检查mask文件是否可读
                        test_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                        if test_mask is not None:
                            valid_pairs.append((image_path, mask_path))
                    except Exception as e:
                        print(f"跳过损坏的文件对: {fid} - {e}")
                        continue
                else:
                    print(f"缺少对应文件: {fid}")
        
        # 更新文件列表为有效的文件对
        self.image_list = [pair[0] for pair in valid_pairs]
        self.mask_list = [pair[1] for pair in valid_pairs]
        self.ids = [os.path.basename(pair[0]) for pair in valid_pairs]
        
        print(f"加载了 {len(valid_pairs)} 个有效的图像-掩码对")

        # Read multi-class mask according to string names
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes else None
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    @staticmethod
    def normalize_image_with_mask(image, mask):
        """
        使用掩码来安全地归一化图像，避免数据泄露。
        只使用背景像素来计算统计量。
        """
        # 只含背景像素的掩码
        background_mask = (mask == 0)
        
        # 没有背景像素，则使用整张图
        if not np.any(background_mask):
            background_pixels = image
        else:
            background_pixels = image[background_mask]
        
        # 在背景像素上计算统计量
        median = np.median(background_pixels)
        std = np.std(background_pixels)
        
        if std > 0:
            lower_bound = median - 5 * std
            upper_bound = median + 5 * std
            # 根据背景统计量进行归一化
            image = np.clip((image - lower_bound) / (upper_bound - lower_bound), 0, 1)
        else:
            # 如果背景没有变化，则将整个图像设为中间值
            image = np.full_like(image, 0.5)
            
        return image.astype(np.float32)

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
    def compute_class_weights(image_dir, mask_dir, classes, num_samples=100):
        """
        计算训练集的类别权重，用于平衡类别不平衡。
        
        Args:
            image_dir: 图像目录路径
            mask_dir: 掩码目录路径
            classes: 类别列表
            num_samples: 采样数量，避免加载所有数据
        
        Returns:
            class_weights: 类别权重列表
        """
        import random
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.fits')]
        if len(image_files) > num_samples:
            image_files = random.sample(image_files, num_samples)
        
        class_counts = np.zeros(len(classes))
        total_pixels = 0
        
        for image_file in image_files:
            file_id = FITSDataset.extract_id(image_file)
            if not file_id:
                continue
                
            mask_path = os.path.join(mask_dir, f"mask_merged_{file_id}.png")
            if not os.path.exists(mask_path):
                continue
            
            # 加载掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            
            # 调试：打印掩码的唯一值
            # unique_vals = np.unique(mask)
            # if len(unique_vals) > 0:
            #     print(f"文件 {image_file} 的掩码唯一值: {unique_vals}")
            
            # 统计每个类别的像素数
            for i, cls in enumerate(classes):
                class_value = FITSDataset.CLASSES.index(cls.lower())
                class_pixels = np.sum(mask == class_value)
                class_counts[i] += class_pixels
            
            total_pixels += mask.size
        
        # 计算权重：权重 = 总像素 / (类别像素 * 类别数)
        # 这样少数类别的权重更高
        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_pixels / (count * len(classes))
            else:
                weight = 1.0  # 避免除零
            class_weights.append(weight)
        
        # 归一化权重，使其和为类别数
        class_weights = np.array(class_weights)
        class_weights = class_weights * len(classes) / np.sum(class_weights)
        
        print(f"类别像素分布: {dict(zip(classes, class_counts.astype(int)))}")
        print(f"计算的类别权重: {class_weights}")
        
        return class_weights.tolist()

    @staticmethod
    def load_fits_image(fits_path):
        """
        从FITS文件加载原始图像数据，不进行归一化。
        """
        # 使用更安全的FITS文件读取方式
        with fitsio.FITS(fits_path, 'r') as fits:
            fits_header = fits[1].read_header()
            fits_data = fits[1].read()
            
        nsamp = fits_header["NBLOCKS"] * fits_header["NSBLK"]
        nchan = fits_header["NCHAN"]
        
        # 直接读取并应用缩放偏移，减少中间变量
        data = fits_data[0]["DATA"].reshape(nsamp, nchan).astype(np.float32)
        dat_scl = fits_data[0]["DAT_SCL"]
        dat_offs = fits_data[0]["DAT_OFFS"]
        
        # 原地操作，减少内存分配
        data += dat_offs[np.newaxis, :]  # 原地加法
        data *= dat_scl[np.newaxis, :]   # 原地乘法
        
        # 合并转置和翻转操作
        image = np.flipud(data.T)
                     
        return image

    def __getitem__(self, i):
        try:
            # 1. 加载原始FITS图像 (未归一化)
            raw_image = FITSDataset.load_fits_image(self.image_list[i])

            # 2. 加载掩码
            masks = cv2.imread(self.mask_list[i], cv2.IMREAD_UNCHANGED)

            # 3. 使用基于全图均值与±5σ的归一化（无标签依赖）
            image = FITSDataset.normalize_image_mean_std(raw_image, k=5.0)

            # 4. 创建用于训练的标签掩码
            mask_labels = np.zeros(masks.shape[:2], dtype=np.uint8)
            if self.class_values:
                for idx, cls_val in enumerate(self.class_values):
                    mask_labels[masks == cls_val] = idx

            # 5. 应用数据增强
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            # 6. 应用预处理 (to_tensor)
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            return image, mask_labels
            
        except Exception as e:
            print(f"Error reading file {self.image_list[i]}: {e}")
            return self.__getitem__((i + 1) % len(self.ids))
    
    def __len__(self):
        return len(self.ids)
    
def get_stable_training_augmentation():
    """
    稳定的射电天文数据增强策略 - 平衡性能和质量的版本
    """
    train_transform = [
        albu.RandomScale(scale_limit=(0.8, 1.2), p=0.3),  # 以30%概率随机缩放，引入不同大小
        albu.Resize(512, 512),  # 回到512x512以提高训练速度

        # 几何变换 - 增加多样性
        # albu.Rotate(limit=15, p=0.3),  # 轻微旋转，±15度
        albu.Affine(
            scale=(0.9, 1.1),
            translate_percent=0,
            rotate=0,
            p=0.25
        ),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.15),

        # 颜色和噪声变换 - 增强鲁棒性
        albu.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.2,
            p=0.5
        ),
        albu.RandomGamma(gamma_limit=(90, 110), p=0.3),  # 轻微gamma调整
        albu.GaussNoise(
            std_range=(0.005, 0.025), # 更保守的标准差范围
            mean_range=(0, 0),        # 均值保持为0
            per_channel=True,
            p=0.3
        ),
        
        # 模糊 - 模拟分辨率变化
        albu.GaussianBlur(blur_limit=(1, 2), p=0.2),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    # 检查数据类型来区分图像和掩码
    if x.dtype in [np.uint8, np.int32, np.int64]:  # 标签掩码
        return torch.from_numpy(x.astype(np.int64))
    else:  # 图像
        if len(x.shape) == 2:  # 灰度图 (H, W) -> (1, H, W)
            x = np.expand_dims(x, axis=0)
        # Note: SMP Unet expects (C, H, W) format
        # If your data is (H, W, C), you might need to transpose it here.
        # However, the current load_fits_image returns (H, W) which is handled above.
        return torch.from_numpy(x.astype(np.float32))

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    _transform = [
        # 由于图像已经在[0,1]范围内，不需要额外的归一化
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform) # type: ignore


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(height, width)
    ]
    return albu.Compose(test_transform) # type: ignore

def visualize_augmentations(dataset, samples=3, cols=3):
    """可视化原始图像、ground truth mask 和叠加效果"""
    rows = samples
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 8))
    
    for i in range(samples):
        print(f"可视化样本 {i}: {dataset.image_list[i]}")
        # 获取原始图像和掩码（经过预处理的）
        image, mask = dataset[i]  # 使用固定索引而非随机
        
        # 如果是 torch 张量，转换为 numpy 数组
        if hasattr(image, 'numpy'):
            image = image.squeeze().numpy()  # 去除通道维度 (C,H,W) -> (H,W)
        else:
            image = np.squeeze(image)  # 去除通道维度 (C,H,W) -> (H,W)
        
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        
        mask = mask.squeeze()  # 移除通道维度以便可视化
        
        # 反归一化图像以便正确显示
        # 因为图像经过了 Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
        # 需要反向操作：img_denorm = img * 255.0
        # image_display = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        # 第一列：原始图像
        # ax[i, 0].imshow(image_display, cmap='gray', vmin=0, vmax=255)
        ax[i, 0].imshow(image, cmap='gist_heat', vmin=0, vmax=1.5)  # 显示灰度图像
        ax[i, 0].set_title("Augmented Image")
        ax[i, 0].axis('off')
        
        # 第二列：Ground Truth Mask（伪彩色）
        ax[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=len(dataset.CLASSES)-1)
        ax[i, 1].set_title("Ground Truth Mask")
        ax[i, 1].axis('off')
        
        # 第三列：图像与mask的叠加
        # 直接在伔彩色图像上叠加mask，不转换为RGB
        
        # 先显示伔彩色的图像作为背景
        ax[i, 2].imshow(image, cmap='gist_heat', vmin=0, vmax=1.5, alpha=1.0)
        
        # 创建mask的叠加层，只在RFI区域显示
        rfi_mask = (mask == 1)
        if np.any(rfi_mask):
            # 创建一个只包含RFI区域的mask数组
            overlay_mask = np.zeros_like(mask, dtype=float)
            overlay_mask[rfi_mask] = 1.0  # RFI区域设为1
            overlay_mask[~rfi_mask] = np.nan  # 非RFI区域设为透明
            
            # 在图像上叠加红色的RFI区域
            ax[i, 2].imshow(overlay_mask, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        ax[i, 2].set_title("Image + Mask Overlay")
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_raw_fits(dataset, indices=None):
    """
    可视化原始 FITS 文件（未预处理），在一个图窗中展示所有图像。
    
    Args:
        dataset: FITSDataset 实例
        indices: 要可视化的索引列表，默认前5个
    """
    if indices is None:
        indices = list(range(min(5, len(dataset))))  # 默认前5个
    
    num_images = len(indices)
    cols = min(5, num_images)  # 每行最多5个
    rows = (num_images + cols - 1) // cols  # 计算行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [axes] if rows == 1 else axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if idx >= len(dataset):
            print(f"索引 {idx} 超出数据集大小 {len(dataset)}，跳过")
            continue
        
        print(f"可视化原始样本 {idx}: {dataset.image_list[idx]}")
        
        # 加载原始 FITS 数据
        raw_image = FITSDataset.load_fits_image(dataset.image_list[idx])
        
        # 调试：打印图像统计信息
        print(f"  图像形状: {raw_image.shape}, 最小值: {raw_image.min():.6f}, 最大值: {raw_image.max():.6f}, 均值: {raw_image.mean():.6f}, 标准差: {raw_image.std():.6f}")
        
        # 计算 vmin, vmax 为 ±5σ
        mean_val = np.mean(raw_image)
        std_val = np.std(raw_image)
        vmin = mean_val - 5 * std_val
        vmax = mean_val + 5 * std_val
        
        print(f"  vmin: {vmin:.6f}, vmax: {vmax:.6f}")
        
        # 可视化
        ax = axes[i]
        im = ax.imshow(raw_image, cmap='gist_heat', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f"Sample {idx}: {os.path.basename(dataset.image_list[idx])}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Time Sample")
        plt.colorbar(im, ax=ax, label="Intensity", shrink=0.8)
    
    # 隐藏多余的子图
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始 U-Net 射电干扰分割训练脚本")
    print("=" * 60)
    
    # 解析命令行参数
    print("📋 解析命令行参数...")
    # ...existing code...
    dataset_top_dir = "/home/cbm/deRFI/Datasets/Dataset_G200.48+2.54_5978_2classes_NoSubMed_FITSCorrected"
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")

    train_image_dir = os.path.join(image_dir, "train")
    train_mask_dir = os.path.join(mask_dir, "train")

    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    # 超参数配置 - 平衡性能和质量的版本
    # ENCODER = "resnet50"
    ENCODER = "mit_b2"  # 使用稍大一点的Transformer-based encoder: Mix Transformer B2
    # CLASSES = ["background", "rfi"]
    CLASSES = ["bkg", "chan_rfi", "point_rfi"]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 1e-4  
    BATCH_SIZE = 8  # 增加batch size提高训练效率
    NUM_WORKERS = 8  # 增加worker数量加速数据加载
    MAX_EPOCHS = 50  # 减少epoch数
    
    # 预训练权重路径
    # weights_path = '/home/cbm/deRFI/pretrained_weights/cent_resnet50.pth'
    
    # 创建数据集 - 使用稳定的增强策略
    print("📊 创建训练和验证数据集...")
    train_dataset = FITSDataset(
        train_image_dir,
        train_mask_dir,
        classes=CLASSES,
        augmentation=get_stable_training_augmentation(),  # 改为稳定版本
        preprocessing=get_preprocessing(None),
        normalization_method="median_sigma",  # 推荐：percentile, median_sigma, zscore, log, minmax
    )

    print(f"✅ 训练数据集: {len(train_dataset)} 个样本")

    # 可视化前几个原始 FITS 文件
    print("🔍 可视化前几个原始 FITS 文件...")
    visualize_raw_fits(train_dataset, indices=[0, 1, 2, 3, 4])

    # 计算类别权重
    print("⚖️ 计算类别权重...")
    class_weights = FITSDataset.compute_class_weights(train_image_dir, train_mask_dir, CLASSES, num_samples=50)

    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(512, 512),  # 指定大小以匹配训练
        preprocessing=get_preprocessing(None),
        normalization_method="median_sigma",  # 验证集使用相同的归一化方法
    )
    
    print(f"✅ 验证数据集: {len(val_dataset)} 个样本")
    
    # 可视化数据增强效果
    print("可视化数据增强效果...")
    visualize_augmentations(train_dataset, samples=3, cols=3)

    # 创建数据加载器 - 优化性能设置
    print("🔄 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        # num_workers=min(8, os.cpu_count()),  # 使用多个worker提高数据加载速度
        num_workers=NUM_WORKERS,
        pin_memory=True,  # 加速GPU数据传输
        persistent_workers=True,  # 保持worker进程，减少启动开销
        drop_last=True  # 确保batch size一致
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4,  # 设置为4以获得更平滑的验证度量
        shuffle=False, 
        # num_workers=min(4, os.cpu_count()),  # 验证集使用较少worker
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"✅ 训练数据加载器: {len(train_loader)} 个批次 (batch_size={BATCH_SIZE})")
    print(f"✅ 验证数据加载器: {len(val_loader)} 个批次 (batch_size=1)")

    # 创建模型
    print("🏗️ 创建 U-Net 模型...")
    model = UNetLightningModule(
        encoder_name=ENCODER,
        classes=len(CLASSES),  # 多类分割，输出通道数等于类别数
        learning_rate=LEARNING_RATE,
        # weights_path=None,  # 随机初始化，不加载预训练权重
        scheduler_type="cosine",  # 可选: "cosine", "plateau", "step", "exponential"
        class_weights=class_weights,  # 自动计算的类别权重
        focal_gamma=2.0
    )
    
    print(f"✅ 模型配置: {ENCODER} 编码器, {len(CLASSES)} 个类别 ({CLASSES})")
    print(f"✅ 学习率: {LEARNING_RATE}, 批次大小: {BATCH_SIZE}, 最大轮次: {MAX_EPOCHS}")
    print(f"✅ 类别权重: {class_weights}")
    print(f"✅ 设备: {DEVICE}")
    
    # 设置示例输入数组以便 TensorBoard 记录计算图
    # 使用与训练数据相同的尺寸: (batch_size, channels, height, width)
    model.example_input_array = torch.randn(1, 1, 512, 512)  # 512x512 单通道图像
    
    # 配置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='best_model-{epoch:02d}-{val_iou:.4f}',
            monitor='val_iou',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_iou',
            patience=10,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 配置 TensorBoard 日志记录器
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='unetpp_training',
        version=None,  # 自动递增版本号
        log_graph=True,  # 记录计算图
        default_hp_metric=False  # 避免默认超参数指标
    )
    
    # 配置训练器 - 性能优化版本
    print("⚙️ 配置训练器...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,  # 显式指定 TensorBoard 日志记录器
        log_every_n_steps=10,  # 减少日志频率以提高性能
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        enable_checkpointing=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,  # 添加梯度裁剪防止训练不稳定
        deterministic=False,  # 允许非确定性操作以提高性能
        strategy='auto',
    )
    
    print("🚀 开始训练...")
    print("=" * 60)
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=None,  # 设置为 None 以从头开始训练，不加载 checkpoint
    )
    
    # 保存最终模型
    trainer.save_checkpoint("final_model.ckpt")
    print("=" * 60)
    print("🎉 训练完成！模型已保存为 final_model.ckpt")


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
        axes[0, 0].imshow(image_np, cmap='gray' if image.shape[0] == 1 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 真实掩码
        im1 = axes[0, 1].imshow(mask_np, cmap='viridis', vmin=0, vmax=model.num_classes-1)
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # 预测掩码
        im2 = axes[0, 2].imshow(pred_np, cmap='viridis', vmin=0, vmax=model.num_classes-1)
        axes[0, 2].set_title('Predicted Mask')
        axes[0, 2].axis('off')
        
        # 叠加：原始图像 + 预测掩码轮廓
        axes[0, 3].imshow(image_np, cmap='gray' if image.shape[0] == 1 else None)
        axes[0, 3].contour(pred_np, colors='red', linewidths=1)
        axes[0, 3].set_title('Overlay (Image + Pred Contour)')
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
        plt.savefig(os.path.join(save_dir, f'inference_sample_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 保存推理可视化结果: {os.path.join(save_dir, f'inference_sample_{i+1}.png')}")
    
    print(f"🎉 推理可视化完成！共处理 {min(num_samples, len(dataset))} 个样本，结果保存至 {save_dir}/")