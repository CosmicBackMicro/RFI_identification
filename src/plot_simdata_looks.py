#!/usr/bin/env python3
"""
Plot class appearance examples from SynthesizedDataset.

- Reads FITS from: <dataset_root>/image/<split>/*.fits
- Reads mask PNG from: <dataset_root>/mask/<split>/*.png
- Pairs by basename intersection (robust to extra/unmatched files)
- Displays/saves one figure per sample:
  - base FITS image in gist_heat
  - optional class-colored mask overlay
  - right-side colorbar for intensity

No marginal/edge distribution plots are produced.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import fitsio
import numpy as np

import matplotlib
import matplotlib.patches as mpatches


def _setup_mpl_backend() -> None:
    """Prefer interactive backend; fallback to Agg in headless environments."""
    env_backend = __import__("os").environ.get("DERFI_MPL_BACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            return
        except Exception:
            pass

    try:
        import tkinter  # noqa: F401

        test_root = tkinter.Tk()
        test_root.withdraw()
        test_root.destroy()
        matplotlib.use("TkAgg", force=True)
    except Exception:
        print("[Info] No graphical display detected, using Agg backend.")
        matplotlib.use("Agg", force=True)


_setup_mpl_backend()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402


CLASS_NAMES: Dict[int, str] = {
    1: "Horizontal",
    2: "Vertical",
    3: "Point",
    4: "Block",
    5: "Pulsar",
    6: "Point",
    7: "Block",
    8: "Pulsar",
}

CLASS_COLOR_MAP: Dict[int, Tuple[float, float, float]] = {
    1: (0.0, 1.0, 1.0),
    2: (1.0, 0.0, 1.0),
    3: (0.0, 0.5, 1.0),
    4: (1.0, 1.0, 0.0),
    5: (0.0, 1.0, 0.0),  # Pulsar -> Green (was red)
    6: (0.0, 0.0, 1.0),
    7: (1.0, 0.6, 0.0),
    8: (0.0, 0.8, 0.0),  # Legacy Pulsar -> Darker Green
}

# Frequency coverage for both 896/1792 channel modes
FREQ_MIN_MHZ = 1031.25
FREQ_MAX_MHZ = 1500.0 - 31.25  # 1468.75 MHz

# 手动指定要绘制的样本名（不带扩展名）。
MANUAL_SAMPLE_BASENAMES: List[str] = [
    # 请在这里填入你想展示的模拟数据的FITS文件名（不带扩展名）
    # /home/cbm/deRFI/Datasets/SynthesizedDataset/image/train/simulation_psrfits_block0001.fits
    "simulation_psrfits_block0002",
]

SAMPLE_DISPLAY_NAMES: Dict[str, str] = {
    "simulation_psrfits_block0002": "Simulated Example",
}


def load_fits_image(fits_path: Path) -> Tuple[np.ndarray, float]:
    """Load FITS image using the same core logic as visualize_fits.py."""
    with fitsio.FITS(str(fits_path), "r") as fits:
        fits_header = fits[1].read_header()
        fits_data = fits[1].read()

    nchan = int(fits_header["NCHAN"])

    rows = fits_data.shape[0] if hasattr(fits_data, "shape") else len(fits_data)
    pieces = []
    for r in range(rows):
        raw = np.asarray(fits_data[r]["DATA"])
        if raw.size % nchan != 0:
            raise ValueError(
                f"DATA length {raw.size} is not divisible by NCHAN {nchan} at row {r}"
            )
        nsamp_row = raw.size // nchan
        arr = raw.reshape(nsamp_row, nchan).astype(np.float32, copy=False)

        dat_scl = np.asarray(fits_data[r]["DAT_SCL"], dtype=np.float32)
        dat_offs = np.asarray(fits_data[r]["DAT_OFFS"], dtype=np.float32)
        if dat_scl.size != nchan or dat_offs.size != nchan:
            raise ValueError(
                f"DAT_SCL/DAT_OFFS length mismatch with NCHAN={nchan} at row {r}"
            )

        arr *= dat_scl[np.newaxis, :]
        arr += dat_offs[np.newaxis, :]
        pieces.append(arr)

    if not pieces:
        raise ValueError("No rows found in SUBINT table.")

    data = np.vstack(pieces)
    image = np.flipud(data.T)

    if "TBIN" in fits_header:
        tbin = float(fits_header["TBIN"])
    elif "TSAMP" in fits_header:
        tbin = float(fits_header["TSAMP"])
    else:
        tbin = 1.0

    return image, tbin


def read_mask_png(mask_path: Path) -> Optional[np.ndarray]:
    """Read mask PNG and return 2D uint8 class-id array."""
    mask_idx = None

    try:
        from PIL import Image

        im = Image.open(mask_path)
        if im.mode == "P":
            mask_idx = np.array(im, dtype=np.uint16)
        elif im.mode == "L":
            mask_idx = np.array(im, dtype=np.uint8)
        elif im.mode.startswith("I"):
            mask_idx = np.array(im, dtype=np.int32)
        else:
            mask_idx = np.array(im.convert("L"), dtype=np.uint8)
    except Exception:
        try:
            import matplotlib.image as mpimg

            mask_img = mpimg.imread(mask_path)
            if mask_img.ndim == 2:
                mask_idx = (mask_img * 255.0 + 0.5).astype(np.uint8)
            elif mask_img.ndim == 3:
                mask_idx = (mask_img[..., 0] * 255.0 + 0.5).astype(np.uint8)
        except Exception:
            return None

    if mask_idx is None or mask_idx.ndim != 2:
        return None

    try:
        mx = int(np.max(mask_idx))
        if mx == 255:
            unique_vals = np.unique(mask_idx)
            if len(unique_vals) == 2 and 0 in unique_vals and 255 in unique_vals:
                mask_idx = (mask_idx // 255).astype(np.uint8, copy=False)
            elif mx > 20:
                mask_idx = (mask_idx // 255).astype(np.uint8, copy=False)
    except Exception:
        pass

    return mask_idx.astype(np.uint8, copy=False)


def to_overlay_rgba(
    mask_aligned: np.ndarray,
    class_color_map: Dict[int, Tuple[float, float, float]],
    alpha: float,
) -> np.ndarray:
    """Convert class-id mask to RGBA overlay."""
    ih, iw = mask_aligned.shape
    rgba = np.zeros((ih, iw, 4), dtype=float)
    cls_ids = np.unique(mask_aligned)
    cls_ids = cls_ids[cls_ids != 0]

    for cid in cls_ids:
        cid_int = int(cid)
        color = class_color_map.get(cid_int, (0.5, 0.5, 0.5))
        sel = mask_aligned == cid_int
        rgba[..., 0][sel] = float(color[0])
        rgba[..., 1][sel] = float(color[1])
        rgba[..., 2][sel] = float(color[2])
        rgba[..., 3][sel] = float(alpha)

    return rgba


def parse_show_class(show_class: Optional[str]) -> Optional[Set[int]]:
    if not show_class:
        return None
    classes = set()
    for token in show_class.replace(" ", "").split(","):
        if token == "":
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid class id token: '{token}'")
        classes.add(int(token))
    return classes if classes else None


def discover_pairs(dataset_root: Path, split: str) -> List[Tuple[str, Path, Path]]:
    """Discover FITS/PNG pairs by basename intersection."""
    all_pairs = []
    # Force check both 'train' and 'val' regardless of args.split for manual names
    for s in ["train", "val"]:
        image_dir = dataset_root / "image" / s
        mask_dir = dataset_root / "mask" / s

        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue

        fits_map = {
            p.stem: p
            for p in sorted(image_dir.glob("*.fits"))
            if p.is_file()
        }
        png_map = {
            p.stem: p
            for p in sorted(mask_dir.glob("*.png"))
            if p.is_file()
        }

        common = sorted(set(fits_map.keys()) & set(png_map.keys()))
        for name in common:
            # We store (name, fits, mask, split) but to keep compatibility with existing code, 
            # we'll return list of (name, fits, mask)
            all_pairs.append((name, fits_map[name], png_map[name]))
    
    # Remove duplicates if any (though usually unique per split)
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        if p[0] not in seen:
            unique_pairs.append(p)
            seen.add(p[0])

    return unique_pairs


def select_pairs(
    pairs: Sequence[Tuple[str, Path, Path]],
    names: Optional[Sequence[str]],
    num_samples: int,
    seed: int,
) -> List[Tuple[str, Path, Path]]:
    if names:
        req = [n.strip() for n in names if n.strip()]
        pair_map = {name: (name, f, m) for name, f, m in pairs}
        selected = []
        missing = []
        for n in req:
            if n in pair_map:
                selected.append(pair_map[n])
            else:
                missing.append(n)
        if missing:
            print(f"[Warn] {len(missing)} requested names not found: {', '.join(missing[:10])}")
        return selected

    if num_samples <= 0 or num_samples >= len(pairs):
        return list(pairs)

    rng = random.Random(seed)
    return rng.sample(list(pairs), k=num_samples)


def _draw_sample_on_axis(
    ax: Axes,
    sample_name: str,
    fits_path: Path,
    mask_path: Path,
    mask_alpha: float,
    allowed_classes: Optional[Set[int]],
    disable_mask: bool,
    zoom_factor: float = 1.0,
    ax_top: Optional[Axes] = None,
    ax_right: Optional[Axes] = None,
):
    image, tbin = load_fits_image(fits_path)
    image = np.flipud(image)
    
    # Add 30 to all pixels
    image += 30.0

    mean = float(image.mean())
    std = float(image.std())
    
    # Debug print for data distribution
    print(f"[Debug] {sample_name} - Original data -> mean: {mean:.4f}, std: {std:.4f}, min: {float(image.min()):.4f}, max: {float(image.max()):.4f}")
    
    if std <= 0:
        vmin, vmax = mean - 1e-6, mean + 1e-6
    else:
        # Use +- 5 sigma range as requested
        vmin, vmax = mean - 5.0 * std, mean + 5.0 * std

    # Bottom limit to 0
    vmin = max(0.0, vmin)

    print(f"[Debug] {sample_name} - Plot bounds -> vmin (-5sig): {vmin:.4f}, vmax (+5sig): {vmax:.4f}")

    nchan, nsamp = image.shape
    extent = (0.0, nsamp * tbin, FREQ_MIN_MHZ, FREQ_MAX_MHZ)

    if nchan not in (896, 1792):
        print(
            f"[Warn] Unexpected channel count {nchan}; still mapping to "
            f"[{FREQ_MIN_MHZ}, {FREQ_MAX_MHZ}] MHz"
        )

    # Handle zooming if needed
    orig_nchan, orig_nsamp = nchan, nsamp
    x0, x1, y0, y1 = 0, nsamp, 0, nchan
    if zoom_factor > 1.0:
        # Calculate central crop in pixel coordinates
        ch, cw = nchan // 2, nsamp // 2
        rh, rw = int(nchan / (2 * zoom_factor)), int(nsamp / (2 * zoom_factor))
        y0, y1 = max(0, ch - rh), min(nchan, ch + rh)
        x0, x1 = max(0, cw - rw), min(nsamp, cw + rw)
        
        image = image[y0:y1, x0:x1]
        # Update extent for the zoomed region
        f_step = (FREQ_MAX_MHZ - FREQ_MIN_MHZ) / nchan
        new_fmin = FREQ_MIN_MHZ + y0 * f_step
        new_fmax = FREQ_MIN_MHZ + y1 * f_step
        new_tmin = x0 * tbin
        new_tmax = x1 * tbin
        extent = (new_tmin, new_tmax, new_fmin, new_fmax)
        nchan, nsamp = image.shape

    im_data = ax.imshow(
        image,
        cmap="gist_heat",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
    )

    if ax_top is not None:
        from scipy.ndimage import gaussian_filter1d
        # Reduced sigma from 3.0 to 1.0 for less aggressive smoothing
        time_mean = gaussian_filter1d(image.mean(axis=0), sigma=2.0)
        t_axis = np.linspace(extent[0], extent[1], len(time_mean))
        ax_top.plot(t_axis, time_mean, color='tab:blue', lw=1.5)
        
    if ax_right is not None:
        from scipy.ndimage import gaussian_filter1d
        # Reduced sigma from 3.0 to 1.0
        freq_mean = gaussian_filter1d(image.mean(axis=1), sigma=2.0)
        f_axis = np.linspace(extent[2], extent[3], len(freq_mean))
        ax_right.plot(freq_mean, f_axis, color='tab:blue', lw=1.5)

    if not disable_mask:
        mask_idx = read_mask_png(mask_path)
        if mask_idx is not None:
            mh, mw = mask_idx.shape
            
            # Simple alignment logic against original full resolution
            if (mh, mw) == (orig_nchan, orig_nsamp):
                mask_aligned = mask_idx
            elif (mw, mh) == (orig_nchan, orig_nsamp):
                mask_aligned = mask_idx.T
            else:
                mask_aligned = None
                print(
                    f"[Warn] Skip mask overlay for {sample_name}: "
                    f"mask shape {mask_idx.shape} mismatches original dimensions ({orig_nchan}, {orig_nsamp})"
                )

            if mask_aligned is not None:
                mask_aligned = np.flipud(mask_aligned)

                if zoom_factor > 1.0:
                    # Use same crop coordinates as image
                    mask_aligned = mask_aligned[y0:y1, x0:x1]

                if allowed_classes is not None:
                    mask_aligned = np.where(
                        np.isin(mask_aligned, list(allowed_classes)),
                        mask_aligned,
                        0,
                    ).astype(np.uint8, copy=False)

                overlay = to_overlay_rgba(mask_aligned, CLASS_COLOR_MAP, mask_alpha)
                ax.imshow(
                    overlay,
                    aspect="auto",
                    interpolation="nearest",
                    extent=extent,
                    origin="lower",
                )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    return im_data


def _compute_grid(n: int) -> Tuple[int, int]:
    """Return (rows, cols) for the grid. For vertical comparison mode, we use n rows and 2 cols."""
    return n, 2


def plot_samples_grid(
    selected: Sequence[Tuple[str, Path, Path]],
    output_dir: Optional[Path],
    split: str,
    mask_alpha: float,
    allowed_classes: Optional[Set[int]],
    disable_mask: bool,
    dpi: int,
    plot_profile: bool = False,
) -> None:
    n = len(selected)
    # Nx2 Layout: Left col (no mask), Right col (with mask)
    rows, cols = n, 2

    # Vertical comparison: width is slightly wider to accommodate colorbar and larger font
    fig_w = max(12.0, cols * 6.0)
    if plot_profile: 
        fig_w += 2.0
        # Increased multiplier from 4.0 to 5.0 to stretch vertically even more
        fig_h = max(5.0, rows * 5.0 + 1.2)
    else:
        # Increased multiplier from 3.8 to 4.2
        fig_h = max(5.0, rows * 4.2 + 1.2)  # Added 1.2 to the total height for the legend at the top
    
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Nx2 Layout: wspace=0.1 to keep columns relatively close
    outer_gs = GridSpec(rows, cols, figure=fig, 
                        wspace=0.1, 
                        hspace=0.3 if plot_profile else 0.2)

    # Dictionary for global legend (consolidate 5/8 and 4/7)
    legend_elements = []
    # Order: Horizontal(1), Point(3), Block(4), Vertical(2), Pulsar(5)
    display_order = [1, 3, 4, 2, 5]
    for cid in display_order:
        if cid in CLASS_COLOR_MAP and cid in CLASS_NAMES:
            color = CLASS_COLOR_MAP[cid]
            name = CLASS_NAMES[cid]
            legend_elements.append(
                mpatches.Rectangle((0, 0), 1, 1, facecolor=color, label=name, edgecolor='none')
            )

    # Add global horizontal legend at the top
    # non-profile mode: legend at 0.95, subplots top at 0.85 to avoid overlap
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(display_order),
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.45, 0.94 if not plot_profile else 0.87)
    )

    for idx, (sample_name, fits_path, mask_path) in enumerate(selected):
        # 默认不缩放
        z_factor = 1.0

        if plot_profile:
            gs_l = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[idx, 0], width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.03, hspace=0.03)
            ax_main_l = fig.add_subplot(gs_l[1, 0])
            ax_top_l = fig.add_subplot(gs_l[0, 0], sharex=ax_main_l)
            ax_right_l = fig.add_subplot(gs_l[1, 1], sharey=ax_main_l)
            
            gs_r = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[idx, 1], width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.03, hspace=0.03)
            ax_main_r = fig.add_subplot(gs_r[1, 0])
            ax_top_r = fig.add_subplot(gs_r[0, 0], sharex=ax_main_r)
            ax_right_r = fig.add_subplot(gs_r[1, 1], sharey=ax_main_r)
        else:
            ax_main_l = fig.add_subplot(outer_gs[idx, 0])
            ax_top_l = None
            ax_right_l = None
            
            ax_main_r = fig.add_subplot(outer_gs[idx, 1])
            ax_top_r = None
            ax_right_r = None

        # Left Column: No Mask
        im_left = _draw_sample_on_axis(
            ax=ax_main_l,
            sample_name=sample_name,
            fits_path=fits_path,
            mask_path=mask_path,
            mask_alpha=mask_alpha,
            allowed_classes=allowed_classes,
            disable_mask=True,
            zoom_factor=z_factor,
            ax_top=ax_top_l,
            ax_right=ax_right_l,
        )
        
        # INCREASE PAD further for titles to avoid overlap with legend
        title_pad = 45 if plot_profile else 35
        
        title_ax_l = ax_top_l if plot_profile else ax_main_l
        if idx == 0:
            title_ax_l.set_title("Original Image", fontsize=18, pad=title_pad, fontweight='bold')
        
        display_name = SAMPLE_DISPLAY_NAMES.get(sample_name, f"Sample {idx+1}")
        ax_main_l.set_ylabel(f"{display_name}\nFreq (MHz)", fontsize=14, fontweight='bold')
        
        ax_main_l.tick_params(axis='both', labelsize=12)
        ax_main_l.xaxis.label.set_size(14)
        
        # Right Column: With Mask Overlay
        im_right = _draw_sample_on_axis(
            ax=ax_main_r,
            sample_name=sample_name,
            fits_path=fits_path,
            mask_path=mask_path,
            mask_alpha=mask_alpha,
            allowed_classes=allowed_classes,
            disable_mask=False,
            zoom_factor=z_factor,
            ax_top=ax_top_r,
            ax_right=ax_right_r,
        )
        
        title_ax_r = ax_top_r if plot_profile else ax_main_r
        if idx == 0:
            title_ax_r.set_title("With Mask Overlay", fontsize=18, pad=title_pad, fontweight='bold')
        
        # IMPORTANT: Hide Y ticks and labels on the right col as it's the same image
        ax_main_r.set_ylabel("")
        ax_main_r.set_yticklabels([])
        ax_main_r.tick_params(axis='y', which='both', left=False, right=False)
        ax_main_r.tick_params(axis='x', labelsize=12)
        ax_main_r.xaxis.label.set_size(14)

        if plot_profile:
            for ax_m in [ax_top_l, ax_top_r]:
                ax_m.tick_params(axis='x', labelbottom=False)
                ax_m.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True, labelsize=10)
            for ax_m in [ax_right_l, ax_right_r]:
                ax_m.tick_params(axis='y', labelleft=False)
                ax_m.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=10)
                ax_m.xaxis.set_tick_params(rotation=45)

        # Separate colorbar parameters for full decoupling
        if plot_profile:
            cbar_ax = [ax_main_r, ax_top_r, ax_right_r]
            cbar_pad = 0.08
            cbar_shrink = 0.75
            cbar_anchor = (0.0, -0.06)
        else:
            cbar_ax = ax_main_r
            cbar_pad = 0.01
            cbar_shrink = 1.0  # Full height when no profiles
            cbar_anchor = (0.0, 0.0)

        # Print colorbar with +/- 5 sigma range naturally inherited from im_left
        cbar = fig.colorbar(
            im_left, 
            ax=cbar_ax, 
            pad=cbar_pad, 
            fraction=0.046, 
            shrink=cbar_shrink, 
            anchor=cbar_anchor
        )
        
        cbar.set_label("Intensity", fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        print(f"[Sample {idx:02d}] {sample_name}")

    # Remove tight_layout and use manual subplots_adjust for total control
    # Set right=0.85 to push the entire grid left, leaving more room for colorbar
    # non-profile mode: top=0.85 to give space for legend at 0.95
    fig.subplots_adjust(
        top=0.85 if not plot_profile else 0.80, 
        bottom=0.1 if not plot_profile else 0.1, 
        left=0.1, 
        right=0.85, 
        hspace=0.4 if plot_profile else 0.2, 
        wspace=0.25
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_profile" if plot_profile else "_noprofile"
        out_path = output_dir / f"simdata_looks_{split}{suffix}.pdf"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {out_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot class appearance samples from SynthesizedDataset (no marginal plots)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/cbm/deRFI/Datasets/SynthesizedDataset",
        help="Dataset root containing image/<split> and mask/<split>.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of random samples when --names is not provided. <=0 means all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="",
        help="Comma-separated basenames (without extension), e.g. a,b,c",
    )
    parser.add_argument(
        "--showclass",
        type=str,
        default=None,
        help="Only show specified classes, comma-separated, e.g. 1,2,5",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.7,
        help="Mask overlay alpha in [0,1].",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable mask overlay.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="If set, save all figures to this directory instead of interactive show.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output DPI when saving figures.",
    )
    parser.add_argument(
        "--plot-profile",
        action="store_true",
        help="Enable drawing 1D marginal profile plots (top and right) using gridspec.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not (0.0 <= args.mask_alpha <= 1.0):
        raise ValueError("--mask-alpha must be in [0,1]")

    dataset_root = Path(args.dataset_root)
    pairs = discover_pairs(dataset_root, args.split)
    if not pairs:
        raise RuntimeError(
            f"No valid FITS/PNG pairs found under {dataset_root} for split '{args.split}'"
        )

    names = [n.strip() for n in args.names.split(",")] if args.names else None
    if not names and MANUAL_SAMPLE_BASENAMES:
        names = MANUAL_SAMPLE_BASENAMES
        print(f"[Info] Using MANUAL_SAMPLE_BASENAMES with {len(names)} entries.")

    selected = select_pairs(
        pairs=pairs,
        names=names,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    if not selected:
        raise RuntimeError("No sample selected. Check --names or --split.")

    allowed_classes = parse_show_class(args.showclass)
    if allowed_classes is not None:
        allowed_classes = {c for c in allowed_classes if c in CLASS_COLOR_MAP}

    output_dir = Path(args.save_dir) if args.save_dir else None

    print(
        f"[Info] split={args.split}, total_pairs={len(pairs)}, selected={len(selected)}, "
        f"mask={'off' if args.no_mask else 'on'}"
    )

    plot_samples_grid(
        selected=selected,
        output_dir=output_dir,
        split=args.split,
        mask_alpha=args.mask_alpha,
        allowed_classes=allowed_classes,
        disable_mask=args.no_mask,
        dpi=args.dpi,
        plot_profile=args.plot_profile,
    )


if __name__ == "__main__":
    main()
