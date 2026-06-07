#!/usr/bin/env python3
"""PointRFIEnhancer: Specialized script for generating reinforced Point-RFI datasets.

Focus:
- Maximizing variety and quantity of isolated, non-periodic point RFI.
- Dimensions: nsamp (Time) = 1792, nchan (Freq) = 1024.
"""
import argparse
import multiprocessing
import os
import time
from datetime import datetime, timezone
from typing import Tuple

import cv2
import fitsio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

BKG = 0
POINT_VERTICAL = 2
POINT_POINT = 6
POINT_BLOCK = 7

def generate_background(nsamp, nchan, bg_mu, bg_sigma, seed):
    rng = np.random.default_rng(seed if seed != 0 else None)
    data = rng.normal(loc=bg_mu, scale=bg_sigma, size=(nsamp, nchan)).astype(np.float32)
    mask = np.zeros_like(data, dtype=np.uint8)
    return data, mask

def dilate_mask(mask_region, radius=3, iterations=1):
    """Dilate boolean 2D array using OpenCV."""
    if radius < 3:
        return mask_region
    kernel = np.ones((radius, radius), np.uint8)
    dilated = cv2.dilate(mask_region.astype(np.uint8), kernel, iterations=iterations)
    return dilated.astype(bool)

def add_block_rfi(data, mask, count, bg_sigma, seed):
    """Add rectangular area contains dense short-time/frequency-extended RFI (from RFISimulator).
    Constrained to the bottom half of the frequency range (L-band-like).
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    for _ in range(count):
        block_w = rng.integers(40, 150) # Time duration
        block_h = rng.integers(100, 400) # Frequency bandwidth
        t0 = rng.integers(0, max(1, nsamp - block_w))
        
        # Constrain to frequencies 0 to nchan//2 (Visual bottom half in origin='lower' + flipud(T) layout)
        f0 = rng.integers(0, max(1, nchan // 2 - block_h))
        
        num_inner_bursts = rng.integers(100, 300)
        for _ in range(num_inner_bursts):
            bt_rel = rng.integers(0, block_w)
            bf_rel = rng.integers(0, block_h)
            burst_dt = rng.integers(1, 4) 
            burst_df = rng.integers(10, 40)
            
            t_start_global = t0 + bt_rel
            f_start_global = f0 + bf_rel
            t_end_global = min(nsamp, t_start_global + burst_dt)
            f_end_global = min(nchan, f_start_global + burst_df)
            
            # Mixture of features: overall lower range (1.5-7.0 sigma)
            # 30%% chance of a "dim" block (SNR 1.5-3.0)
            if rng.random() < 0.30:
                amp = rng.uniform(1.5, 3.0) * bg_sigma
            else:
                amp = rng.uniform(3.0, 7.0) * bg_sigma
                
            data[t_start_global:t_end_global, f_start_global:f_end_global] += amp
            mask[t_start_global:t_end_global, f_start_global:f_end_global] = POINT_BLOCK
    return data, mask

def add_isolated_points_varied(data, mask, bg_sigma, seed):
    rng = np.random.default_rng(seed)
    nsamp, nchan = data.shape
    roll = rng.random()
    if roll < 0.6: point_count = rng.integers(80, 150)
    elif roll < 0.9: point_count = rng.integers(150, 300)
    else: point_count = rng.integers(300, 600)
    for _ in range(point_count):
        t, f = rng.integers(0, nsamp), rng.integers(0, nchan)
        
        # 2x2+ shapes only (no 1x3 or 3x1)
        dt, df = rng.integers(2, 5), rng.integers(2, 5)
            
        t_end = min(nsamp, t + dt)
        f_end = min(nchan, f + df)
        
        amp_roll = rng.random()
        # Lowered amplitude ranges to include more subtle features (3.0-15.0 max)
        if amp_roll < 0.4: amp = rng.uniform(2.5, 5.0) * bg_sigma
        elif amp_roll < 0.8: amp = rng.uniform(5.0, 9.0) * bg_sigma
        else: amp = rng.uniform(9.0, 15.0) * bg_sigma
        
        data[t:t_end, f:f_end] += amp
        mask[t:t_end, f:f_end] = POINT_POINT
    return data, mask

def add_freq_extended_points(data, mask, bg_sigma, seed):
    """Adds larger points with Gaussian frequency extension (extreme aspect ratio 10-30x)."""
    rng = np.random.default_rng(seed)
    nsamp, nchan = data.shape
    # Reduced count from 15-30 to 5-12 to make it more realistic
    count = rng.integers(5, 13)
    for _ in range(count):
        t0 = rng.integers(0, nsamp)
        f0 = rng.integers(0, nchan)
        
        # extreme Frequency scale (sy) is 10-30x larger than Time scale (sx)
        sx = rng.uniform(0.5, 1.0)
        sy = sx * rng.uniform(10.0, 30.0)
        
        # Lowered SNR range (3.0-10.0 sigma)
        amp = rng.uniform(3.0, 10.0) * bg_sigma
        
        # Range to apply (3 sigma coverage)
        wx, wy = int(np.ceil(3*sx)), int(np.ceil(3*sy))
        s_t, e_t = max(0, t0-wx), min(nsamp, t0+wx+1)
        s_f, e_f = max(0, f0-wy), min(nchan, f0+wy+1)
        
        if e_t <= s_t or e_f <= s_f: continue
        
        tt, ff = np.meshgrid(np.arange(s_t, e_t), np.arange(s_f, e_f), indexing='ij')
        gv = amp * np.exp(-0.5 * (((tt-t0)/sx)**2 + ((ff-f0)/sy)**2))
        
        data[s_t:e_t, s_f:e_f] += gv.astype(data.dtype)
        mask[s_t:e_t, s_f:e_f][gv > amp * 0.1] = POINT_POINT
    return data, mask

def add_many_blob_points(data, mask, count, bg_sigma, seed):
    rng = np.random.default_rng(seed)
    nsamp, nchan = data.shape
    for _ in range(count):
        x, y = rng.integers(0, nsamp), rng.integers(0, nchan)
        amp = rng.uniform(8, 16) * bg_sigma
        sx, sy = rng.uniform(0.8, 1.5), rng.uniform(0.8, 1.5)
        wx, wy = int(np.ceil(3*sx)), int(np.ceil(3*sy))
        s_x, e_x = max(0, x-wx), min(nsamp, x+wx+1)
        s_y, e_y = max(0, y-wy), min(nchan, y+wy+1)
        xx, yy = np.meshgrid(np.arange(s_y, e_y), np.arange(s_x, e_x))
        gv = amp * np.exp(-0.5 * (((yy-y)/sy)**2 + ((xx-x)/sx)**2))
        data[s_x:e_x, s_y:e_y] += gv.astype(data.dtype)
        mask[s_x:e_x, s_y:e_y][gv > amp*0.01] = POINT_POINT
    return data, mask

def generate_sample(nsamp, nchan, bg_mu, bg_sigma, seed):
    data, mask = generate_background(nsamp, nchan, bg_mu, bg_sigma, seed)
    rng = np.random.default_rng(seed)
    
    # 1. Block (Limit to 1 instance max, with 20%% chance of none)
    if rng.random() > 0.2:
        data, mask = add_block_rfi(data, mask, 1, bg_sigma, seed + 20)

    # 2. Points
    # 2a. Varied shape points (min 1x3, 3x1, or 2x2)
    data, mask = add_isolated_points_varied(data, mask, bg_sigma, seed)
    # 2b. Large extension points (Frequency extended 10-35 channels)
    data, mask = add_freq_extended_points(data, mask, bg_sigma, seed + 50)
    # 2c. Blob points (Gaussian profile)
    data, mask = add_many_blob_points(data, mask, rng.integers(2, 9), bg_sigma, seed + 1)
    
    # 3. Vertical (Broadband impulses: Single time, all channels)
    # Reduced frequency: 30%% chance of appearing per image, and lower count (1-3)
    if rng.random() > 0.7:
        for _ in range(rng.integers(1, 4)):
            v_t = rng.integers(0, nsamp)
            v_width = rng.integers(1, 3) 
            t_end = min(nsamp, v_t + v_width)
            
            # Substantially lower intensity for impulses (1.0 - 4.0 sigma)
            amp = rng.uniform(1.0, 4.0) * bg_sigma
            data[v_t:t_end, :] += amp
            mask[v_t:t_end, :] = POINT_VERTICAL
    
    return data, mask

def plot_spectrogram_and_mask(data, mask, plot_path, title):
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.flipud(data.T), aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title(f"Data: {title}")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.flipud(mask.T), aspect="auto", origin='lower', cmap="tab10", vmin=0, vmax=9)
    ax2.set_title(f"Mask: {title}")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--nsamp", type=int, default=1024)
    parser.add_argument("--nchan", type=int, default=1792)
    parser.add_argument("--outdir", type=str, default="Datasets/PointReinforced")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.join(args.outdir, "fits"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    
    print(f"Generating {args.num_samples} samples with Dimensions {args.nsamp}x{args.nchan} (Time x Freq)...")
    
    for i in range(args.num_samples):
        # Use a deterministic-ish seed based on sample index
        seed = 42 + i
        data, mask = generate_sample(args.nsamp, args.nchan, 0, 1.0, seed)
        
        # Naming format: PointEnhance_block1, PointEnhance_block2, etc.
        fname = f"PointEnhance_block{i+1}"
        
        fits_path = os.path.join(args.outdir, "fits", f"{fname}.fits")
        mask_path = os.path.join(args.outdir, "masks", f"{fname}.png")
        
        # Write FITS (Time x Freq flattened)
        # RFISimulator uses NCHAN=1792, NSBLK=1024 in SUBINT table.
        with fitsio.FITS(fits_path, "rw", clobber=True) as f:
            # Primary HDU
            f.write(None, header={"NCHAN": args.nchan, "NSAMP": args.nsamp, "TBIN": 0.0002})
            # SUBINT HDU
            # Note: data is (nsamp, nchan). flatten() preserves Time-major order.
            row_data = np.zeros(1, dtype=[
                ("DAT_OFFS", "f4", (args.nchan,)),
                ("DAT_SCL", "f4", (args.nchan,)), 
                ("DATA", "f4", (args.nsamp * args.nchan,))
            ])
            row_data["DATA"][0] = data.flatten()
            row_data["DAT_SCL"][0] = np.ones(args.nchan, dtype="f4")
            row_data["DAT_OFFS"][0] = np.zeros(args.nchan, dtype="f4")
            f.write(row_data, header={"NCHAN": args.nchan, "NSBLK": args.nsamp, "TBIN": 0.0002}, extname="SUBINT")
        
        # Write Mask PNG
        # IMPORTANT: To match RFISimulator layout:
        # 1. Transpose: (Time, Freq) -> (Freq, Time)
        # 2. Flipud: Put Freq 0 at the bottom (image last row)
        # Resulting shape: (1792, 1024)
        mask_img = np.flipud(mask.T).astype(np.uint8)
        cv2.imwrite(mask_path, mask_img)
        
        if i % 500 == 0:
            plot_path = os.path.join(args.outdir, "plots", f"{fname}.png")
            plot_spectrogram_and_mask(data, mask, plot_path, fname)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {i}/{args.num_samples}")

    print("Generation complete.")

if __name__ == "__main__":
    main()
