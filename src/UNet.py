import matplotlib.pyplot as plt
import numpy as np
import os, re
import pytorch_lightning as pl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用的GPU设备
os.environ['SMP_HUB_MODE'] = "original"

# 优化Tensor Cores的利用 - 针对RTX 4060等支持Tensor Cores的GPU
import torch
torch.set_float32_matmul_precision('medium')  # 在性能和精度之间取得平衡

import numpy as np
import matplotlib.pyplot as plt
import cv2
import fitsio
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

class UNetLightningModule(pl.LightningModule):
    def __init__(self, encoder_name="resnet50", classes=2, learning_rate=0.0001, weights_path=None, scheduler_type="cosine"):
        super().__init__()
        self.save_hyperparameters()
        
        # 保存调度器类型
        self.scheduler_type = scheduler_type
        
        # 初始化模型
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=1,  # 单通道射电图像
            decoder_channels=(256, 128, 64, 32, 16),  # ResNet50需要5个解码器通道
            decoder_use_norm="batchnorm",
            decoder_attention_type="scse",  # 添加注意力机制
            decoder_interpolation="bilinear",  # 更好的上采样
            classes=classes,
            activation=None  # 移除激活函数，在损失函数中处理
        )
        
        # 如果有预训练权重，加载它们
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weights_path}")
        
        # 定义损失函数
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.ce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
        
        self.learning_rate = learning_rate
        
    def joint_loss(self, y_pred, y_true):
        """联合损失函数"""
        # 对于多分类，需要将 y_true 转换为正确的格式
        if y_true.dtype != torch.long:
            y_true = y_true.long()
        
        dice_weight = 0.7
        ce_weight = 0.3
        return dice_weight * self.dice_loss(y_pred, y_true) + ce_weight * self.ce_loss(y_pred, y_true)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        loss = self.joint_loss(outputs, masks)
        
        # 将输出转换为概率并计算预测
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # 计算指标 - 转换为二进制格式
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='multiclass', num_classes=2)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        loss = self.joint_loss(outputs, masks)
        
        # 将输出转换为概率并计算预测
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # 计算指标 - 转换为二进制格式
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='multiclass', num_classes=2)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        # 记录指标
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # 使用AdamW优化器，通常比Adam表现更好
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,  # 添加权重衰减正则化
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 选择学习率调度策略：
        # "cosine" - 余弦退火（推荐，适合长时间训练）
        # "plateau" - 基于验证损失的自适应调整（保守但可靠）
        # "step" - 固定步长降低（简单直接）
        # "exponential" - 指数衰减（平滑下降）
        scheduler_config = self.get_scheduler_config(optimizer, scheduler_type=self.scheduler_type)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
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
                verbose=True,
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
    CLASSES = ["background", "rfi"]  # Define your classes here

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
        self.ids = os.listdir(image_dir)
        self.image_list = [os.path.join(image_dir, fid) for fid in self.ids]
        self.mask_list = []
        self.normalization_method = normalization_method  # 保存归一化方法
        
        # 验证文件完整性
        valid_pairs = []
        for fid in self.ids:
            file_id = self.extract_id(fid)
            if file_id:
                mask_path = os.path.join(mask_dir, f"mask_Subst_{file_id}.png")
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
    def load_fits_image(fits_path):
        """
        从FITS文件加载并处理射电天文图像
        
        Args:
            fits_path (str): FITS文件路径
            normalization_method (str): 归一化方法
                - "percentile": 基于百分位数的robust归一化
                - "median_sigma": 基于中值±5σ的归一化（您提出的方法）
                - "zscore": Z-score标准化后sigmoid映射
                - "minmax": 传统最小-最大值归一化
                - "log": 对数变换后归一化
            
        Returns:
            np.ndarray: 处理后的归一化图像 (nchan, nsamp)，值范围[0,1]
        """
        # 使用更安全的FITS文件读取方式
        with fitsio.FITS(fits_path, 'r') as fits:
            fits_header = fits[1].read_header()
            fits_data = fits[1].read()
            
        nsamp = fits_header["NBLOCKS"] * fits_header["NSBLK"]
        nchan = fits_header["NCHAN"]
        
        # 优化1: 直接读取并应用缩放偏移，减少中间变量
        data = fits_data[0]["DATA"].reshape(nsamp, nchan).astype(np.float32)
        dat_scl = fits_data[0]["DAT_SCL"]
        dat_offs = fits_data[0]["DAT_OFFS"]
        
        # 优化2: 原地操作，减少内存分配
        data += dat_offs[np.newaxis, :]  # 原地加法
        data *= dat_scl[np.newaxis, :]   # 原地乘法
        
        # 优化3: 合并转置和翻转操作
        image = np.flipud(data.T)
        
        # 归一化
        median = np.median(image)
        std = np.std(image)
        if std > 0:
            # 定义5-sigma范围
            lower_bound = median - 5 * std
            upper_bound = median + 5 * std
            # 将5-sigma范围映射到[0,1]，其他值clip掉
            image = np.clip((image - lower_bound) / (upper_bound - lower_bound), 0, 1)
        else:
            image = np.full_like(image, 0.5)  # 如果没有变化，设为中间值
                     
        return image

    def __getitem__(self, i):
        try:
            # 使用指定的归一化方法加载FITS图像
            image = self.load_fits_image(self.image_list[i])

            masks = cv2.imread(self.mask_list[i], cv2.IMREAD_UNCHANGED)
            if masks.dtype == np.uint8 and masks.max() > 1:
                masks = masks // 255

            mask_labels = np.zeros(masks.shape[:2], dtype=np.uint8)
            for idx, cls_val in enumerate(self.class_values):
                mask_labels[masks == cls_val] = idx

            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask_labels)
                image, mask_labels = sample['image'], sample['mask']

            return image, mask_labels
            
        except Exception as e:
            print(f"Error reading file {self.image_list[i]}: {e}")
            return self.__getitem__((i + 1) % len(self.ids))
    
    def __len__(self):
        return len(self.ids)
    
def get_training_augmentation():
    """
    针对射电天文时间-频率图像的数据增强策略
    
    设计原则：
    1. 保持时间-频率轴的物理意义
    2. 模拟真实的射电观测条件变化
    3. 增强模型对不同RFI模式的泛化能力
    """
    train_transform = [
        # albu.Resize(512, 512),  # 标准化输入尺寸
        albu.Resize(512, 512),  # 标准化输入尺寸
        
        # 亮度和对比度变化 - 模拟不同观测条件和仪器响应
        albu.RandomBrightnessContrast(
            brightness_limit=0.2,    # 适度的亮度变化
            contrast_limit=0.3,      # 对比度增强有助于突出RFI特征
            p=0.5
        ),
        
        # 添加噪声 - 模拟射电观测中的热噪声和系统噪声
        albu.GaussNoise(
            std_range=(0.01, 0.05),   # 标准差范围，控制噪声强度
            mean_range=(0, 0),        # 均值范围，保持为0
            per_channel=True,         # 每个通道独立添加噪声
            p=0.3
        ),
        
        # 轻微的缩放 - 模拟不同时间分辨率的观测
        albu.Affine(
            scale=(0.85, 1.15),      # 保守的缩放范围
            translate_percent=0,      # 不进行平移，保持时间对齐
            rotate=0,                 # 不旋转，保持轴向意义
            p=0.3
        ),
        
        # 频率轴方向的翻转 - 某些情况下频谱可能倒置
        albu.VerticalFlip(p=0.2),   # 降低概率，仅在确实合理时使用
        
        # # 小幅度的弹性变形 - 模拟色散延迟等物理效应
        # albu.ElasticTransform(
        #     alpha=20,                 # 降低变形强度，提高稳定性
        #     sigma=3,                  # 降低平滑度参数
        #     approximate=True,         # 加速计算
        #     p=0.15                    # 降低概率
        # ),
        
        # # Gamma校正 - 模拟不同的动态范围压缩
        # # 注意：为了避免数值不稳定，使用更保守的gamma范围
        # albu.RandomGamma(
        #     gamma_limit=(80, 120),    # 更保守的gamma变化 (对应0.8-1.2)
        #     p=0.2                     # 降低概率，减少数值问题
        # ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """验证集只进行尺寸标准化，不做其他变换"""
    test_transform = [
        albu.Resize(512, 512),  # 回到512x512以提高验证速度
    ]
    return albu.Compose(test_transform)

def get_advanced_radio_augmentation():
    """
    高级射电天文数据增强策略（可选使用）
    
    专门针对射电天文RFI检测任务设计的增强方法
    """
    train_transform = [
        albu.Resize(512, 512),
        
        # 模拟频率分辨率变化
        albu.OneOf([
            albu.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, p=1.0),  # 模拟低频率分辨率
            albu.MedianBlur(blur_limit=3, p=1.0),                        # 去除尖锐噪声
        ], p=0.2),
        
        # 模拟不同的动态范围和量化噪声
        albu.OneOf([
            albu.RandomGamma(gamma_limit=(85, 115), p=1.0),  # 更保守的gamma范围
            albu.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),    # 局部对比度增强
        ], p=0.2),  # 降低概率
        
        # 射电干扰的强度变化
        albu.RandomBrightnessContrast(
            brightness_limit=0.2,     # 降低亮度变化范围
            contrast_limit=0.3,       # 降低对比度变化范围
            p=0.4
        ),
        
        # 模拟系统噪声 - 使用更稳定的噪声类型
        albu.OneOf([
            albu.GaussNoise(std_range=(0.01, 0.03), mean_range=(0, 0), per_channel=True, p=1.0),
            albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
        ], p=0.3),
        
        # 轻微的几何变形（模拟色散等效应）- 更保守的参数
        albu.OneOf([
            albu.ElasticTransform(alpha=15, sigma=2, approximate=True, p=1.0),
            albu.OpticalDistortion(distort_limit=0.05, shift_limit=0.02, p=1.0),
        ], p=0.1),
        
        # 保守的缩放（模拟不同观测带宽）
        albu.Affine(
            scale=(0.95, 1.05),       # 更保守的缩放范围
            translate_percent=0,
            rotate=0,
            p=0.15
        ),
    ]
    return albu.Compose(train_transform)

def get_stable_training_augmentation():
    """
    稳定的射电天文数据增强策略 - 平衡性能和质量的版本
    """
    train_transform = [
        albu.Resize(512, 512),  # 回到512x512以提高训练速度
        
        # 基础的亮度对比度调整 - 最稳定的增强
        albu.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # 高斯噪声 - 数值稳定
        albu.GaussNoise(
            std_range=(0.005, 0.025), # 更保守的标准差范围
            mean_range=(0, 0),        # 均值保持为0
            per_channel=True,
            p=0.3
        ),
        
        # 轻微缩放 - 保守参数
        albu.Affine(
            scale=(0.9, 1.1),
            translate_percent=0,
            rotate=0,
            p=0.25
        ),
        
        # 频率轴翻转 - 在某些情况下合理
        albu.VerticalFlip(p=0.15),
        
        # 轻微模糊 - 模拟分辨率变化
        albu.GaussianBlur(blur_limit=(1, 2), p=0.2),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    # 对于图像：确保输入是灰度图像
    if len(x.shape) == 2:
        # 灰度图像或标签掩码，添加通道维度
        if x.dtype == np.uint8 and x.max() < 10:  # 假设是标签掩码
            return x.astype('int64')  # 标签掩码不需要额外的通道维度
        else:
            # 灰度图像，添加通道维度
            x = np.expand_dims(x, axis=0)  # Shape is now (1, H, W)
    elif len(x.shape) == 3:
        # 3D数组，从 (H, W, C) 转换为 (C, H, W)
        x = x.transpose((2, 0, 1))  # Shape is now (C, H, W)
    return x.astype('float32')      # Convert to float32

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    _transform = [
        # 由于图像已经在[0,1]范围内，不需要额外的归一化
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def visualize_augmentations(dataset, samples=3, cols=3):
    """可视化原始图像、ground truth mask 和叠加效果"""
    rows = samples
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 8))
    
    for i in range(samples):
        # 获取原始图像和掩码（经过预处理的）
        image, mask = dataset[i]  # 使用固定索引而非随机
        
        # 如果是 torch 张量，转换为 numpy 数组
        if hasattr(image, 'numpy'):
            image = image.squeeze().numpy()  # 去除通道维度 (C,H,W) -> (H,W)
        else:
            image = np.squeeze(image)  # 去除通道维度 (C,H,W) -> (H,W)
        
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        
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
        # 直接在伪彩色图像上叠加mask，不转换为RGB
        
        # 先显示伪彩色的图像作为背景
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

if __name__ == "__main__":
    dataset_top_dir = "/home/cbm/deRFI/dataset"
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")

    train_image_dir = os.path.join(image_dir, "train")
    train_mask_dir = os.path.join(mask_dir, "train")

    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    # 超参数配置 - 平衡性能和质量的版本
    ENCODER = "resnet50"
    CLASSES = ["background", "rfi"]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 3e-4  
    BATCH_SIZE = 3  # 增加batch size提高训练效率
    NUM_WORKERS = 8  # 增加worker数量加速数据加载
    MAX_EPOCHS = 30  # 减少epoch数
    
    # 预训练权重路径
    weights_path = '/home/cbm/deRFI/pretrained_weights/cent_resnet50.pth'
    
    # 创建数据集 - 使用稳定的增强策略
    train_dataset = FITSDataset(
        train_image_dir,
        train_mask_dir,
        classes=CLASSES,
        augmentation=get_stable_training_augmentation(),  # 改为稳定版本
        preprocessing=get_preprocessing(None),
        normalization_method="median_sigma",  # 推荐：percentile, median_sigma, zscore, log, minmax
    )

    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(None),
        normalization_method="median_sigma",  # 验证集使用相同的归一化方法
    )
    
    # 可视化数据增强效果
    print("可视化数据增强效果...")
    visualize_augmentations(train_dataset, samples=3, cols=3)

    # 创建数据加载器 - 优化性能设置
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
        batch_size=1, 
        shuffle=False, 
        # num_workers=min(4, os.cpu_count()),  # 验证集使用较少worker
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # 创建模型
    model = UNetLightningModule(
        encoder_name=ENCODER,
        classes=len(CLASSES),
        learning_rate=LEARNING_RATE,
        weights_path=weights_path,
        scheduler_type="cosine"  # 可选: "cosine", "plateau", "step", "exponential"
    )
    
    # 设置示例输入数组以便 TensorBoard 记录计算图
    # 使用与训练数据相同的尺寸: (batch_size, channels, height, width)
    model.example_input_array = torch.randn(1, 1, 512, 512)  # 512x512 单通道图像
    
    # 配置回调函数
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints',
            filename='best_model-{epoch:02d}-{val_iou:.4f}',
            monitor='val_iou',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
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
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    # 保存最终模型
    trainer.save_checkpoint("final_model.ckpt")
    print("训练完成！模型已保存。")