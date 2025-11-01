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

import argparse

def test_load_fits_image(output_dir):
    """测试load_fits_image函数，在同一个图窗中步进显示所有图像"""
    import glob
    import re

    print(f"Searching for .fits and .fit files in: {output_dir}")

    # Find all .fits and .fit files only in the output directory
    fits_files = glob.glob(os.path.join(output_dir, '*.fits'))
    fits_files += glob.glob(os.path.join(output_dir, '*.fit'))

    # Sort files based on the block number in the filename
    def get_block_number(filename):
        match = re.search(r'block(\d+)\.(fits|fit)', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return -1
    
    fits_files.sort(key=get_block_number)

    if not fits_files:
        print(f"No FITS files found in the directory: {output_dir}")
        return

    # 在循环外创建图窗和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))
    # 添加一个占位符图像和颜色条
    image_display = ax.imshow(np.zeros((1,1)), aspect='auto', cmap='gist_heat')
    colorbar = fig.colorbar(image_display, ax=ax)
    colorbar.set_label('Intensity')

    for i, fits_path in enumerate(fits_files):
        if not os.path.exists(fits_path):
            print(f"Test FITS file not found: {fits_path}")
            continue
        
        print(f"Loading FITS file: {fits_path} ({i+1}/{len(fits_files)})")
        
        # 调用load_fits_image函数
        image = load_fits_image(fits_path)
        
        if image is None:
            print("Failed to load image")
            continue
        
        # Calculate vmin and vmax based on 3-sigma
        mean = image.mean()
        std = image.std()
        vmin = mean - 3 * std
        vmax = mean + 3 * std
        
        # 更新图像数据和范围
        image_display.set_data(image)
        image_display.set_clim(vmin, vmax)
        
        # 更新标题和标签
        ax.set_title(f'Visualization of {os.path.basename(fits_path)}')
        ax.set_xlabel('Time Sample')
        ax.set_ylabel('Channel')
        
        # 重绘图窗
        fig.canvas.draw()

        # 如果不是最后一张图，则等待按键
        if i < len(fits_files) - 1:
            plt.waitforbuttonpress()
        else:
            # 显示最后一张图，直到手动关闭
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize FITS files in a directory.')
    parser.add_argument('--dir', type=str, default="/home/cbm/deRFI/output",
                        help='Directory containing FITS files to visualize. Defaults to the hardcoded path.')
    args = parser.parse_args()
    test_load_fits_image(args.dir)