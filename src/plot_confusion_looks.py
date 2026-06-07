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
    "simulation_psrfits_block0002"
]

SAMPLE_DISPLAY_NAMES: Dict[str, str] = {
    "simulation_psrfits_block0002": "SegFormer Confusion Sample"
}


def load_fits_image(fits_path: Path) -> Tuple[np.ndarray, float]:
    """Load specific block0002 from the simulation_psrfits.fits file."""
    # We ignore fits_path and hardcode the row 1 (which is block 0002 if 0-indexed is block0001)
    # The subint rows in simulation_psrfits are typically 256 samples per row.
    # User specifically wants block0002.
    with fitsio.FITS(str(fits_path), "r") as fits:
        fits_header = fits[1].read_header()
        # Only read the specific row (r=1 for block0002) instead of the whole file
        r = 2 
        fits_data_row = fits[1][r:r+1]

    nchan = int(fits_header["NCHAN"])
    raw = np.asarray(fits_data_row["DATA"][0])
    if raw.size % nchan != 0:
        raise ValueError(f"DATA length {raw.size} is not divisible by NCHAN {nchan}")
    nsamp_row = raw.size // nchan
    arr = raw.reshape(nsamp_row, nchan).astype(np.float32, copy=False)

    dat_scl = np.asarray(fits_data_row["DAT_SCL"][0], dtype=np.float32)
    dat_offs = np.asarray(fits_data_row["DAT_OFFS"][0], dtype=np.float32)
    
    arr *= dat_scl[np.newaxis, :]
    arr += dat_offs[np.newaxis, :]
    arr += 30.0  # Add 30 to all pixels as requested
    
    data = arr
    # image = np.flipud(data.T)  # We remove flipud because we want raw Frequency-major data
    image = data.T # (NCHAN, NSAMP)

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
    """Discover FITS/PNG pairs specifically for the SegFormer confusion test."""
    all_pairs = []
    
    # Force set paths to user specified directories
    base = Path("/home/cbm/deRFI/PaperExperiments/simulation_v4")
    image_dir = base
    mask_dir = base / "mask_SegFormer"

    if image_dir.is_dir() and mask_dir.is_dir():
        fits_map = {
            p.stem: p
            for p in sorted(image_dir.glob("*.fits"))
            if p.is_file()
        }
        # In this folder, its simulation_psrfits.fits, which contains many blocks.
        # But our loading logic handles reading the underlying FITS.
        # However, the user wants "simulation_psrfits.fits" block0002.
        # The existing discover_pairs expects a 1:1 file mapping.
        # Let's adjust to create a virtual mapping for block0002.
        
        target_stem = "simulation_psrfits_block0002"
        fits_path = image_dir / "simulation_psrfits.fits"
        mask_path = mask_dir / "simulation_psrfits_block0002.png"
        
        if fits_path.exists() and mask_path.exists():
            all_pairs.append((target_stem, fits_path, mask_path))

    return all_pairs


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
):
    image, tbin = load_fits_image(fits_path)
    # image = np.flipud(image) # Removed to keep natural orientation compatible with standard imshow extent

    mean = float(image.mean())
    std = float(image.std())
    
    # Debug print for data distribution
    print(f"[Debug] {sample_name} - Original data -> mean: {mean:.4f}, std: {std:.4f}, min: {float(image.min()):.4f}, max: {float(image.max()):.4f}")
    
    if std <= 0:
        vmin, vmax = mean - 1e-6, mean + 1e-6
    else:
        # Use +- 5 sigma range as requested
        vmin, vmax = mean - 5.0 * std, mean + 5.0 * std

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
                # Re-adding flipud for mask as requested to match image orientation
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
) -> None:
    n = len(selected)
    # Nx2 Layout: Left col (no mask), Right col (with mask)
    rows, cols = n, 2

    # Vertical comparison: width is slightly wider to accommodate colorbar and larger font
    fig_w = max(12.0, cols * 6.0)
    fig_h = max(4.0, rows * 5.0 + 1.2)  # Adjusted for better aspect ratio
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

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
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(display_order),
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98)
    )

    for idx, (sample_name, fits_path, mask_path) in enumerate(selected):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Set zoom factor for Point sample to 2.0, else 1.0
        z_factor = 2.0 if sample_name == "G72.95-1.53_20251108_block1" else 1.0

        # Left Column: No Mask
        ax_left = axes[idx, 0]
        im_left = _draw_sample_on_axis(
            ax=ax_left,
            sample_name=sample_name,
            fits_path=fits_path,
            mask_path=mask_path,
            mask_alpha=mask_alpha,
            allowed_classes=allowed_classes,
            disable_mask=True,
            zoom_factor=z_factor,
        )
        if idx == 0:
            ax_left.set_title("Original Image", fontsize=18, pad=10, fontweight='bold')
        
        display_name = SAMPLE_DISPLAY_NAMES.get(sample_name, f"Sample {idx+1}")
        ax_left.set_ylabel(f"{display_name}\nFreq (MHz)", fontsize=14, fontweight='bold')
        
        ax_left.tick_params(axis='both', labelsize=12)
        ax_left.xaxis.label.set_size(14)
        
        # Right Column: With Mask Overlay
        ax_right = axes[idx, 1]
        im_right = _draw_sample_on_axis(
            ax=ax_right,
            sample_name=sample_name,
            fits_path=fits_path,
            mask_path=mask_path,
            mask_alpha=mask_alpha,
            allowed_classes=allowed_classes,
            disable_mask=False,
            zoom_factor=z_factor,
        )
        if idx == 0:
            ax_right.set_title("With Model B Mask Overlay", fontsize=18, pad=10, fontweight='bold')
        
        # IMPORTANT: Hide Y ticks and labels on the right col as it's the same image
        ax_right.set_ylabel("")
        ax_right.set_yticklabels([])
        ax_right.tick_params(axis='y', which='both', left=False, right=False)
        ax_right.tick_params(axis='x', labelsize=12)
        ax_right.xaxis.label.set_size(14)

        # Use make_axes_locatable to create a colorbar axis that matches the image axis height
        divider = make_axes_locatable(ax_right)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Print colorbar with +/- 5 sigma range naturally inherited from im_left
        # --- 针对 Sample 3 ("G200.14+2.80_20250409_block79") 特判 colorbar 数值 ---
        if sample_name == "G200.14+2.80_20250409_block79":
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            # 假定你想要的真实数值范围是 10.0 到 35.0，这只改变色带标尺刻度
            custom_vmin = 0.0  # <--- 请修改为真实的最小值
            custom_vmax = 47.0  # <--- 请修改为真实的最大值
            sm = cm.ScalarMappable(cmap=im_left.cmap, norm=mcolors.Normalize(vmin=custom_vmin, vmax=custom_vmax))
            cbar = fig.colorbar(sm, cax=cax)
        else:
            cbar = fig.colorbar(im_left, cax=cax)
        # -------------------------------------------------------------------------
        
        cbar.set_label("Intensity", fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        print(f"[Sample {idx:02d}] {sample_name}")

    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Leaving space for the global legend at top

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"class_looks_{split}_{n}samples.pdf"
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
    )


if __name__ == "__main__":
    main()
