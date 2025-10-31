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
    data *= dat_scl[np.newaxis, :]   # 原地乘法
    data += dat_offs[np.newaxis, :]  # 原地加法
    
    # 合并转置和翻转操作
    image = np.flipud(data.T)
                    
    return image

def test_load_fits_image():
    """测试load_fits_image函数，直接可视化输出"""
    import glob

    # Get current directory
    current_dir = os.getcwd()
    print(f"Searching for .fits and .fit files in: {current_dir}")

    # Find all .fits and .fit files
    fits_files = glob.glob(os.path.join(current_dir, '**', '*.fits'), recursive=True)
    fits_files += glob.glob(os.path.join(current_dir, '**', '*.fit'), recursive=True)

    if not fits_files:
        print("No FITS files found in the current directory.")
        return

    for fits_path in fits_files:
        if not os.path.exists(fits_path):
            print(f"Test FITS file not found: {fits_path}")
            continue
        
        print(f"Loading FITS file: {fits_path}")
        
        # 调用load_fits_image函数
        image = load_fits_image(fits_path)
        
        if image is None:
            print("Failed to load image")
            continue
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image range: {image.min():.2e} to {image.max():.2e}")
        print(f"Image mean: {image.mean():.2e}")
        print(f"Image std: {image.std():.2e}")
        
        # Calculate vmin and vmax based on 5-sigma
        mean = image.mean()
        std = image.std()
        vmin = mean - 3 * std
        vmax = mean + 3 * std
        
        # 直接可视化image数组
        plt.figure(figsize=(12, 8))
        plt.imshow(image, aspect='auto', cmap='gist_heat', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Intensity')
        plt.title(f'Visualization of {os.path.basename(fits_path)}')
        plt.xlabel('Time Sample')
        plt.ylabel('Channel')
        plt.show()
        
        # Wait for user input to continue
        # input("Press Enter to continue to the next image...")

if __name__ == "__main__":
    test_load_fits_image()