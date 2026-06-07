#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import fitsio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import cv2

CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),       # black
    1: (0.0, 1.0, 1.0),       # cyan
    2: (1.0, 0.0, 1.0),       # magenta
    3: (0.0, 1.0, 0.0),       # bright green
    4: (1.0, 1.0, 0.0),       # yellow
    5: (1.0, 0.0, 0.0),       # red
    6: (0.0, 1.0, 0.0),       # bright green (was blue)
    7: (1.0, 0.6, 0.0),       # orange
    8: (0.8, 0.0, 0.0),       # dark red
}

def mask_to_rgb(mask_path, is_binary=False):
    """将 Mask 文件转换为 RGB 彩色渲染"""
    if not os.path.exists(mask_path):
        return None
    
    # 读取 Mask 图像 (通常是 uint8)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    if is_binary:
        # 对于 AOFlagger，非 0 即为 RFI (白色显示)
        rgb[mask > 0] = (1.0, 1.0, 1.0)
    else:
        # 对于多分类模型和 GT
        for cid, color in CLASS_COLORS.items():
            if cid == 0: continue
            rgb[mask == cid] = color
            
        # 兼容性：如果 mask 是二值化的 (0, 255)，且不是 is_binary，则全转为红色辅助显示
        if mask.max() == 255 and not np.any((mask > 0) & (mask < 255)):
             rgb[mask == 255] = (1.0, 0.0, 0.0)

    return rgb

def load_fits_subint(fits_path, subint_idx):
    """
    Load raw image data from a specific subint of a FITS file and return as (nchan, nsamp) array.
    """
    with fitsio.FITS(fits_path, 'r') as fits:
        fits_header = fits[1].read_header()
        # Read only the specific row (subint)
        fits_data = fits[1][subint_idx:subint_idx+1]
        
    nchan = int(fits_header["NCHAN"])
    
    raw = np.asarray(fits_data[0]["DATA"])
    nsamp_row = raw.size // nchan
    arr = raw.reshape(nsamp_row, nchan).astype(np.float32)

    dat_scl = np.asarray(fits_data[0]["DAT_SCL"], dtype=np.float32)
    dat_offs = np.asarray(fits_data[0]["DAT_OFFS"], dtype=np.float32)
    
    # Apply scale offset
    arr *= dat_scl[np.newaxis, :]
    arr += dat_offs[np.newaxis, :]
    
    # Convert to (nchan, nsamp) and flip up/down (low freq bottom)
    image = np.flipud(arr.T)
    return image

def main():
    parser = argparse.ArgumentParser(description='Plot triple comparison: before, mask, after for a specific subint')
    parser.add_argument('--before', type=str, required=True, help='Path to "before" PSRFITS file')
    parser.add_argument('--after', type=str, required=True, help='Path to "after" PSRFITS file')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask PNG files')
    parser.add_argument('--subint', type=int, required=True, help='Subint index (0-based) to plot')
    parser.add_argument('--output', type=str, default='triple_comparison.pdf', help='Output filename')
    
    args = parser.parse_args()

    # Validate existence of FITS files
    for f in [args.before, args.after]:
        if not os.path.exists(f):
            print(f"Error: File {f} not found.")
            sys.exit(1)

    # Find mask file
    mask_pattern1 = os.path.join(args.mask_dir, f"*block{args.subint:04d}.png")
    mask_pattern2 = os.path.join(args.mask_dir, f"*block{args.subint}.png")
    mask_pattern3 = os.path.join(args.mask_dir, f"*_{args.subint:04d}.png")
    mask_pattern4 = os.path.join(args.mask_dir, f"*_{args.subint}.png")
    
    mask_files = glob.glob(mask_pattern1)
    if not mask_files:
        mask_files = glob.glob(mask_pattern2)
    if not mask_files:
        mask_files = glob.glob(mask_pattern3)
    if not mask_files:
        mask_files = glob.glob(mask_pattern4)
        
    if not mask_files:
        print(f"Error: Could not find mask file for subint {args.subint} in {args.mask_dir}")
        print(f"Tried patterns: {mask_pattern1}, {mask_pattern2}, {mask_pattern3}, {mask_pattern4}")
        sys.exit(1)
        
    mask_path = mask_files[0]

    print(f"Loading files for subint {args.subint}:")
    print(f"- Before: {args.before}")
    print(f"- Mask:   {mask_path}")
    print(f"- After:  {args.after}")

    # Load data
    img_before = load_fits_subint(args.before, args.subint)
    img_after = load_fits_subint(args.after, args.subint)
    img_mask = mask_to_rgb(mask_path)
    
    if img_mask is None:
        print(f"Error: Failed to load mask image from {mask_path}")
        sys.exit(1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Normalize FITS data for better visualization (5% to 95% percentile)
    def normalize(data):
        vmin, vmax = np.percentile(data, [5, 95])
        return vmin, vmax

    vmin_b, vmax_b = normalize(img_before)
    vmin_a, vmax_a = normalize(img_after)
    
    # Use the same scale for before/after for fair comparison
    vmin = min(vmin_b, vmin_a)
    vmax = max(vmax_b, vmax_a)

    # Set extent to correctly label the y-axis (Frequency Channels)
    # Assuming nchan is the height of the image
    nchan = img_before.shape[0]
    nsamp = img_before.shape[1]
    # extent = [left, right, bottom, top]
    # To keep the image as is but flip the y-axis labels, we set bottom=0 and top=nchan
    extent = [0, nsamp, 0, nchan]

    im0 = axes[0].imshow(img_before, aspect='auto', cmap='gist_heat', vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title(f'Before AI-RFI Excision')
    axes[0].set_ylabel('Frequency Channels')
    axes[0].set_xlabel('Time Samples')

    axes[1].imshow(img_mask, aspect='auto', extent=extent)
    axes[1].set_title(f'RFI Mask')
    axes[1].set_xlabel('Time Samples')

    im2 = axes[2].imshow(img_after, aspect='auto', cmap='gist_heat', vmin=vmin, vmax=vmax, extent=extent)
    axes[2].set_title(f'After AI-RFI Excision')
    axes[2].set_xlabel('Time Samples')

    # Adjust layout before adding colorbar to avoid overlap
    plt.tight_layout()
    
    # Add a colorbar for the FITS images
    fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.03, pad=0.04, label='Intensity')

    plt.savefig(args.output, dpi=300)
    print(f"Triple plot saved to {args.output}")

if __name__ == "__main__":
    main()
