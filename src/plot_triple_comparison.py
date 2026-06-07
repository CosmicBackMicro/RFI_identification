#!/usr/bin/env python3
"""
Compare Original, Mask, and AI-Cleaned data from experimental results.
- Original: results/PixelSubstPlot/G30.00+6.44_20240120_snapshot-M09-P4-c2048b1/subint_XXXX.npy
- Mask: results/PixelSubstPlot/G30.00_mask/subint_XXXX.png
- Cleaned: results/PixelSubstPlot/G30.00+6.44_20240120_snapshot-M09-P4-c2048b1_AIRFI/subint_XXXX.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

def _setup_mpl_backend() -> None:
    try:
        import tkinter
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)

_setup_mpl_backend()

# Frequency coverage (matching earlier scripts)
FREQ_MIN_MHZ = 1031.25
FREQ_MAX_MHZ = 1500.0 - 31.25 # 1468.75 MHz
TBIN = 0.000131072 # Typical FAST tbin, adjust if known precisely

def load_npy_data(path: Path) -> np.ndarray:
    data = np.load(path)
    # The extraction script saves as Time x Frequency.
    # To match plot_simdata_looks (Frequency on Y, Time on X), we transpose and flip.
    return np.flipud(data.T)

def load_mask_png(path: Path) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    mask = np.array(im)
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Mask is likely already 1792x1024 (Freq x Time)
    return np.flipud(mask)

def plot_triple_comparison(
    subint_idx: int,
    base_dir: Path,
    output_path: Path,
    dpi: int = 150
):
    orig_dir = base_dir / "G30.00+6.44_20240120_snapshot-M09-P4-c2048b1"
    mask_dir = base_dir / "G30.00_mask"
    clean_dir = base_dir / "G30.00+6.44_20240120_snapshot-M09-P4-c2048b1_AIRFI"

    orig_file = orig_dir / f"subint_{subint_idx:04d}.npy"
    # Find mask file with glob due to differing naming conventions
    mask_pattern = f"*_block{subint_idx:04d}.png"
    mask_matches = list(mask_dir.glob(mask_pattern))
    if not mask_matches:
        mask_pattern = f"subint_{subint_idx:04d}.png"
        mask_matches = list(mask_dir.glob(mask_pattern))
    
    if not mask_matches:
        print(f"[Error] Mask file not found for subint {subint_idx} using pattern {mask_pattern}")
        return
    mask_file = mask_matches[0]
    
    clean_file = clean_dir / f"subint_{subint_idx:04d}.npy"

    if not all([orig_file.exists(), mask_file.exists(), clean_file.exists()]):
        print(f"[Error] Missing files for subint {subint_idx}")
        return

    data_orig = load_npy_data(orig_file)
    data_mask = load_mask_png(mask_file)
    data_clean = load_npy_data(clean_file)

    # Offset for visualization (matching plot_simdata_looks)
    data_orig += 30.0
    data_clean += 30.0

    # Stats for color scaling (use original data)
    mean, std = data_orig.mean(), data_orig.std()
    vmin, vmax = max(0, mean - 3*std), mean + 5*std

    nchan, nsamp = data_orig.shape
    extent = (0, nsamp * TBIN, FREQ_MIN_MHZ, FREQ_MAX_MHZ)

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 3, wspace=0.15, left=0.08, right=0.92, top=0.85, bottom=0.15)
    
    titles = ["(a) Original Data", "(b) Identified RFI Mask", "(c) Cleaned Data"]
    images = [data_orig, data_mask, data_clean]
    cmaps = ["gist_heat", "gray", "gist_heat"]
    
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        curr_vmin = 0 if i == 1 else vmin
        curr_vmax = 1 if i == 1 else vmax
        
        im = ax.imshow(
            images[i], 
            aspect='auto', 
            extent=extent, 
            origin='lower', 
            cmap=cmaps[i],
            vmin=curr_vmin, 
            vmax=curr_vmax,
            interpolation='nearest'
        )
        ax.set_title(titles[i], fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Time (s)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Frequency (MHz)", fontsize=12)
        else:
            ax.set_yticklabels([])

        if i != 1: # Colorbar for data panels
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)

    plt.suptitle(f"RFI Mitigation Comparison - G30.00 Subint {subint_idx}", fontsize=16, fontweight='bold', y=0.96)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"[Saved] {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subint", type=int, default=0)
    parser.add_argument("--base-dir", type=str, default="/home/cbm/deRFI/results/PixelSubstPlot")
    parser.add_argument("--outdir", type=str, default="results/comparison_plots")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_triple_comparison(args.subint, base_dir, outdir / f"triple_comp_subint_{args.subint:04d}.png")

if __name__ == "__main__":
    main()
