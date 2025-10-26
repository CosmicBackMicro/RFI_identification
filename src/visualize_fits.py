#!/usr/bin/env python3
"""
简单的FITS文件可视化脚本 - 直接显示load_fits_image函数的输出
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import fitsio

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

def test_load_fits_image():
    """测试load_fits_image函数，直接可视化输出"""
    # 使用找到的FITS文件
    fits_path = "/home/cbm/deRFI/output/G200.14+2.80_20251026_block3.fits"
    
    if not os.path.exists(fits_path):
        print(f"Test FITS file not found: {fits_path}")
        return
    
    print(f"Loading FITS file: {fits_path}")
    
    # 调用load_fits_image函数
    image = load_fits_image(fits_path)
    
    if image is None:
        print("Failed to load image")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: {image.min():.2e} to {image.max():.2e}")
    print(f"Image mean: {image.mean():.2e}")
    print(f"Image std: {image.std():.2e}")
    
    # 直接可视化image数组
    plt.figure(figsize=(12, 8))
    plt.imshow(image, aspect='auto', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('load_fits_image Output')
    plt.xlabel('Time Sample')
    plt.ylabel('Channel')
    plt.show()

if __name__ == "__main__":
    test_load_fits_image()