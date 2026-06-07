#!/usr/bin/env python3
"""
Real-data triple comparison (Original, Mask, Cleaned) based on plot_pixel_subst.py style.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib
import matplotlib.patches as mpatches


def _setup_mpl_backend() -> None:
    """Prefer interactive backend; fallback to Agg in headless environments."""
    try:
        import tkinter
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)


_setup_mpl_backend()

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

CLASS_NAMES: Dict[int, str] = {
    1: "Horizontal",
    2: "Vertical",
    3: "Point",
    4: "Block",
    5: "Pulsar",
}

CLASS_COLOR_MAP: Dict[int, Tuple[float, float, float]] = {
    1: (0.0, 1.0, 1.0),
    2: (1.0, 0.0, 1.0),
    3: (0.0, 0.5, 1.0),
    4: (1.0, 1.0, 0.0),
    5: (0.0, 1.0, 0.0),
}

FREQ_MIN_MHZ = 1031.25
FREQ_MAX_MHZ = 1500.0 - 31.25  # 1468.75 MHz

# 手动指定要绘制的样本名
MANUAL_SAMPLE_BASENAMES: List[str] = [
    "subint_0000",
]

SAMPLE_DISPLAY_NAMES: Dict[str, str] = {
    "subint_0000": "G30.00 Subint 0",
}


def load_npy_data(path: Path) -> Tuple[np.ndarray, float]:
    """Load subint data from .npy and transpose to match plot format."""
    data = np.load(path)
    # npy is (Time x Frequency) -> we want (Frequency x Time)
    # The previous flipud was causing misalignment with the mask.
    image = data.T
    return image, 0.000131072


def read_mask_png(mask_path: Path) -> Optional[np.ndarray]:
    """Read mask PNG and return 2D uint8 class-id array."""
    try:
        from PIL import Image

        im = Image.open(mask_path)
        mask_idx = np.array(im.convert("L"), dtype=np.uint8)
        if mask_idx.max() == 255:
            # Simple heuristic for binary masks if they aren't class-coded indexed
            unique_vals = np.unique(mask_idx)
            if len(unique_vals) == 2 and 0 in unique_vals and 255 in unique_vals:
                mask_idx = (mask_idx // 255).astype(np.uint8)
        return mask_idx
    except Exception:
        return None


def to_overlay_rgba(
    mask_aligned: np.ndarray, class_color_map: Dict[int, Tuple[float, float, float]], alpha: float
) -> np.ndarray:
    ih, iw = mask_aligned.shape
    rgba = np.zeros((ih, iw, 4), dtype=float)
    cls_ids = np.unique(mask_aligned)
    cls_ids = cls_ids[cls_ids != 0]
    for cid in cls_ids:
        color = class_color_map.get(int(cid), (0.5, 0.5, 0.5))
        sel = mask_aligned == cid
        rgba[..., 0][sel], rgba[..., 1][sel], rgba[..., 2][sel], rgba[..., 3][sel] = (
            color[0],
            color[1],
            color[2],
            float(alpha),
        )
    return rgba


def discover_triplets(base_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
    orig_dir = base_dir / "G30.00+6.44_20240120_snapshot-M09-P4-c2048b1"
    mask_dir = base_dir / "G30.00_mask"
    clean_dir = base_dir / "G30.00+6.44_20240120_snapshot-M09-P4-c2048b1_AIRFI"
    triplets = []
    if not orig_dir.exists():
        return []
    for npy_file in sorted(orig_dir.glob("*.npy")):
        stem = npy_file.stem
        idx_str = stem.split("_")[-1]
        mask_matches = list(mask_dir.glob(f"*_block{idx_str}.png"))
        if not mask_matches:
            mask_matches = list(mask_dir.glob(f"{stem}.png"))
        if not mask_matches:
            continue
        clean_npy = clean_dir / f"{stem}.npy"
        if clean_npy.exists():
            triplets.append((stem, npy_file, mask_matches[0], clean_npy))
    return triplets


def plot_samples_grid(
    selected: Sequence[Tuple[str, Path, Path, Path]], output_dir: Optional[Path], mask_alpha: float, dpi: int, plot_profile: bool = False
) -> None:
    n = len(selected)
    # Changed from n x 3 to (n*3) x 1 to stack vertically
    rows, cols = n * 3, 1
    
    # Adjust figure size: stretched horizontally and more vertical space
    fig_w = 9.0 if not plot_profile else 12.0
    # Increase height multiplier to accommodate larger hspace
    fig_h = rows * (4.5 if plot_profile else 3.2) + 1.5
    
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Increased hspace for breathing room
    outer_gs = GridSpec(rows, cols, figure=fig, hspace=0.5 if plot_profile else 0.4, 
                        left=0.12, right=0.88, top=0.90, bottom=0.1)

    legend_elements = [mpatches.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOR_MAP[cid], label=CLASS_NAMES[cid]) for cid in [1, 3, 4, 2, 5] if cid in CLASS_NAMES]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.96))

    for idx, (sample_name, orig_p, mask_p, clean_p) in enumerate(selected):
        # Calculate vmin/vmax from original
        ref_img, _ = load_npy_data(orig_p)
        ref_img += 30.0
        mean, std = ref_img.mean(), ref_img.std()
        vmin, vmax = max(0, mean - 5*std), mean + 5*std

        for row_in_sample in range(3):
            # idx_outer is the absolute row index in the GridSpec
            idx_outer = idx * 3 + row_in_sample
            
            data_path = clean_p if row_in_sample == 2 else orig_p
            show_mask = (row_in_sample == 1)
            
            if plot_profile:
                gs_curr = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[idx_outer, 0], width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.03, hspace=0.03)
                ax_main = fig.add_subplot(gs_curr[1, 0])
                ax_top = fig.add_subplot(gs_curr[0, 0], sharex=ax_main)
                ax_right = fig.add_subplot(gs_curr[1, 1], sharey=ax_main)
            else:
                ax_main = fig.add_subplot(outer_gs[idx_outer, 0])
                ax_top = ax_right = None

            image, tbin = load_npy_data(data_path)
            image += 30.0
            nchan, nsamp = image.shape
            extent = (0.0, nsamp * tbin, FREQ_MIN_MHZ, FREQ_MAX_MHZ)

            im = ax_main.imshow(image, cmap="gist_heat", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, extent=extent, origin="lower")
            
            if plot_profile:
                from scipy.ndimage import gaussian_filter1d
                ax_top.plot(np.linspace(extent[0], extent[1], nsamp), gaussian_filter1d(image.mean(axis=0), sigma=2.0), color='tab:blue')
                ax_right.plot(gaussian_filter1d(image.mean(axis=1), sigma=2.0), np.linspace(extent[2], extent[3], nchan), color='tab:blue')
                ax_top.tick_params(labelbottom=False, labelsize=10)
                ax_right.tick_params(labelleft=False, labelsize=10)

            if show_mask:
                m_idx = read_mask_png(mask_p)
                if m_idx is not None:
                    overlay = to_overlay_rgba(np.flipud(m_idx), CLASS_COLOR_MAP, mask_alpha)
                    ax_main.imshow(overlay, aspect="auto", interpolation="nearest", extent=extent, origin="lower")

            ax_main.set_xlabel("Time (s)", fontsize=12, labelpad=5)
            ax_main.set_ylabel("Frequency (MHz)", fontsize=12)
            ax_main.tick_params(labelsize=10)
            
            # Title for each vertical panel
            if plot_profile and ax_top is not None:
                ax_t = ax_top
            else:
                ax_t = ax_main
            
            state_titles = ["(a) Original Data", "(b) Identified RFI Mask", "(c) Cleaned Data"]
            ax_t.set_title(state_titles[row_in_sample], fontsize=14, fontweight='bold', pad=8)

            # Colorbar for each panel
            if plot_profile and ax_top is not None and ax_right is not None:
                cb_ax = [ax_main, ax_top, ax_right]
            else:
                cb_ax = ax_main
            
            cbar = fig.colorbar(im, ax=cb_ax, 
                               pad=0.08 if plot_profile else 0.02, fraction=0.046, 
                               shrink=0.8 if plot_profile else 1.0)
            cbar.set_label("Intensity", fontsize=10, fontweight='bold')


    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_p = output_dir / f"triple_comparison_real.pdf"
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {out_p}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="results/comparison_plots")
    parser.add_argument("--mask-alpha", type=float, default=0.7)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--plot-profile", action="store_true")
    args = parser.parse_args()

    base_dir = Path("/home/cbm/deRFI/results/PixelSubstPlot")
    triplets = discover_triplets(base_dir)
    selected = [t for t in triplets if t[0] in MANUAL_SAMPLE_BASENAMES]
    if not selected:
        selected = triplets[:2]

    plot_samples_grid(selected, Path(args.save_dir), args.mask_alpha, args.dpi, args.plot_profile)


if __name__ == "__main__":
    main()
