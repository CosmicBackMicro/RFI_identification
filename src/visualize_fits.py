#!/usr/bin/env python3
"""
简单的FITS文件可视化脚本 - 直接显示load_fits_image函数的输出
"""

import os
import numpy as np
import fitsio
import matplotlib


def _setup_mpl_backend():
    """Prefer a Tk backend to avoid mixing Qt (matplotlib) with Tk (file dialog),
    which can cause freezes on some Linux/remote/X11 setups. Fallback to Agg if Tk is unavailable.
    You can override by setting environment variable DERFI_MPL_BACKEND.
    """
    env_backend = os.environ.get("DERFI_MPL_BACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            return
        except Exception:
            pass  # fall through to auto selection
    # Try TkAgg first
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg", force=True)
        return
    except Exception:
        # Headless or Tk unavailable -> safe non-interactive backend
        matplotlib.use("Agg", force=True)


_setup_mpl_backend()

# Disable matplotlib's default key bindings that conflict with our shortcuts
try:
    # Avoid grid toggling when pressing 'g'/'G'
    matplotlib.rcParams['keymap.grid'] = []
    matplotlib.rcParams['keymap.grid_minor'] = []
    # Avoid built-in nav on left/right/home that could interfere with our handler
    matplotlib.rcParams['keymap.back'] = []
    matplotlib.rcParams['keymap.forward'] = []
    matplotlib.rcParams['keymap.home'] = []
    # Disable axis offset formatting like +1.37e+2 on x-axis
    matplotlib.rcParams['axes.formatter.useoffset'] = False
except Exception:
    pass

import matplotlib.pyplot as plt

def load_fits_image(fits_path):
    """
    从FITS文件加载原始图像数据，不进行归一化。
    返回 (image, tbin)，其中 tbin 为每个时间样本的时间宽度（秒）。
    若头信息中无 TBIN/TSAMP，则 tbin 默认 1.0。
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
    
    # 合并转置和翻转操作（保持原先显示方向）
    image = np.flipud(data.T)

    # 提取时间分辨率
    tbin = None
    # 常见键名：TBIN（PSRFITS），或 TSAMP 等
    if 'TBIN' in fits_header:
        tbin = float(fits_header['TBIN'])
    elif 'TSAMP' in fits_header:
        tbin = float(fits_header['TSAMP'])
    else:
        tbin = 1.0
        print("[Info] Header 未找到 TBIN/TSAMP，tbin 采用默认 1.0 秒")

    return image, tbin

import argparse
from typing import Optional

def test_load_fits_image(output_dir: str, verbose: bool=False, mask_dir: Optional[str]=None, mask_alpha: float=0.35):
    """使用键盘左右方向键进行双向浏览（←/→），按 J 跳转编号，Q/Esc 退出。
    verbose=True 时会打印每帧加载日志，默认关闭以减少 I/O。
    """
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

    print("Right/Left Arrow: Step Forward/Back; J: Jump to index; Q/Esc: Quit")

    # Optional: prepare mask file mapping by block index
    mask_map = {}
    if mask_dir:
        import glob as _glob
        import re as _re
        if not os.path.isdir(mask_dir):
            print(f"[Warn] Mask dir not found: {mask_dir}, overlay disabled.")
            mask_dir = None
        else:
            # Accept common case variants
            pngs = []
            for pat in ('*.png', '*.PNG', '*.Png'):
                pngs.extend(_glob.glob(os.path.join(mask_dir, pat)))
            def _idx_from_mask_name(name):
                # Prefer 'blockNNN' pattern; fallback to last integer in filename
                m = _re.search(r'block(\d+)', name)
                if m:
                    return int(m.group(1))
                m2 = _re.findall(r'(\d+)', name)
                if m2:
                    return int(m2[-1])
                return None
            for p in pngs:
                idx = _idx_from_mask_name(os.path.basename(p))
                if idx is not None:
                    mask_map[idx] = p
            print(f"[Info] Found {len(mask_map)} mask PNG(s) in: {mask_dir}")

    # Create figure and axis once
    fig, ax = plt.subplots(figsize=(12, 8))
    image_display = ax.imshow(np.zeros((1, 1)), aspect='auto', cmap='gist_heat')
    # Create an empty RGBA overlay for mask; updated per frame when available
    mask_display = ax.imshow(np.zeros((1, 1, 4), dtype=float), aspect='auto', interpolation='nearest')
    colorbar = fig.colorbar(image_display, ax=ax)
    colorbar.set_label('Intensity')

    # idx: current index; busy: rendering in progress; pending: pending direction (-1/0/+1)
    state = {'idx': 0, 'busy': False, 'pending': 0}

    def show(index):
        # Clamp index
        index = max(0, min(index, len(fits_files) - 1))
        path = fits_files[index]
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            return

        if verbose:
            print(f"Loading FITS file: {path} ({index+1}/{len(fits_files)})")
        image, tbin = load_fits_image(path)
        if image is None:
            print("Failed to load image")
            return

        # Calculate vmin and vmax based on 3-sigma (guard zero-std)
        mean = float(image.mean())
        std = float(image.std())
        if std <= 0:
            vmin, vmax = mean - 1e-6, mean + 1e-6
        else:
            vmin, vmax = mean - 3 * std, mean + 3 * std

        # Update image, color limits and coordinate extent
        image_display.set_data(image)
        image_display.set_clim(vmin, vmax)

    # 设置坐标范围：
        #  x 轴：相对第 1 个样本的时间，(block_index-1)*nsamp*tbin + i*tbin
        #  y 轴：频道 0..nchan
        nchan, nsamp = image.shape
        import re as _re
        m = _re.search(r'block(\d+)\.(fits|fit)$', os.path.basename(path))
        block_idx = int(m.group(1)) if m else 1
        x_left = ((block_idx - 1) * nsamp) * tbin
        x_right = x_left + nsamp * tbin
        y_bottom, y_top = 0, nchan
        try:
            image_display.set_extent((x_left, x_right, y_bottom, y_top))
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_bottom, y_top)
        except Exception:
            pass

        # If mask overlay enabled, try to overlay corresponding PNG by block index
        if mask_dir:
            # Derive block index from FITS filename
            m2 = _re.search(r'block(\d+)\.(fits|fit)$', os.path.basename(path))
            block_idx = int(m2.group(1)) if m2 else None
            overlay = None
            # Try exact match first; then try off-by-one (0-based masks vs 1-based blocks)
            candidate_path = None
            if block_idx is not None:
                if block_idx in mask_map:
                    candidate_path = mask_map[block_idx]
                elif (block_idx - 1) in mask_map:
                    candidate_path = mask_map[block_idx - 1]
                elif (block_idx + 1) in mask_map:
                    candidate_path = mask_map[block_idx + 1]
            if candidate_path:
                try:
                    # Read class-index mask preserving integer labels
                    mask_idx = None
                    try:
                        from PIL import Image as _PIL_Image
                        _im = _PIL_Image.open(candidate_path)
                        # Preserve palette indices if present; else convert to 8-bit or 32-bit integer
                        if _im.mode == 'P':
                            mask_idx = np.array(_im, dtype=np.uint16)
                        elif _im.mode in ('L',):
                            mask_idx = np.array(_im, dtype=np.uint8)
                        elif _im.mode.startswith('I'):
                            mask_idx = np.array(_im, dtype=np.int32)
                        else:
                            # Fallback: convert to 'L' (8-bit) which holds label indices up to 255
                            mask_idx = np.array(_im.convert('L'), dtype=np.uint8)
                    except Exception:
                        # Fallback to matplotlib if PIL unavailable; will likely return floats in [0,1]
                        import matplotlib.image as mpimg
                        mask_img = mpimg.imread(candidate_path)
                        if mask_img.ndim == 2:
                            mask_idx = (mask_img * 255.0 + 0.5).astype(np.uint8)
                        elif mask_img.ndim == 3:
                            mask_idx = (mask_img[..., 0] * 255.0 + 0.5).astype(np.uint8)
                        else:
                            mask_idx = None

                    if mask_idx is not None and mask_idx.ndim == 2:
                        # Try to match orientation to displayed image (nchan x nsamp)
                        mh, mw = mask_idx.shape
                        ih, iw = image.shape
                        if (mh, mw) == (ih, iw):
                            mask_aligned = mask_idx
                        elif (mh, mw) == (iw, ih):
                            mask_aligned = mask_idx.T
                        else:
                            # Shapes differ; cannot safely rescale without extra deps; disable overlay for this frame
                            print(f"[Warn] Mask shape {mh}x{mw} mismatches image {ih}x{iw}; skip overlay for {block_idx}")
                            mask_aligned = None

                        if mask_aligned is not None:
                            # Build categorical color overlay: 0=background (transparent), >0 are classes
                            rgba = np.zeros((ih, iw, 4), dtype=float)
                            # A small color palette for classes 1..N (cycled)
                            # Prefer GB-dominant, low-R colors to contrast gi st_heat (red-toned)
                            palette = np.array([
                                [0.00, 1.00, 0.00],  # green
                                [0.00, 1.00, 1.00],  # cyan
                                [0.00, 0.85, 0.70],  # teal
                                [0.00, 0.70, 1.00],  # sky blue
                                [0.10, 0.90, 0.90],  # light cyan (low R)
                                [0.10, 1.00, 0.40],  # spring green (low R)
                                [0.00, 0.50, 1.00],  # blue
                                [0.15, 0.85, 0.55],  # sea green
                                [0.20, 0.70, 1.00],  # azure
                                [0.25, 1.00, 0.75],  # aquamarine
                            ], dtype=float)
                            cls_ids = np.unique(mask_aligned)
                            cls_ids = cls_ids[cls_ids != 0]
                            for cid in cls_ids:
                                sel = (mask_aligned == cid)
                                color = palette[(int(cid) - 1) % len(palette)]
                                rgba[..., 0][sel] = color[0]
                                rgba[..., 1][sel] = color[1]
                                rgba[..., 2][sel] = color[2]
                                rgba[..., 3][sel] = float(mask_alpha)
                            overlay = rgba
                except Exception as e:
                    print(f"[Warn] Failed to overlay mask for block {block_idx}: {e}")

            if overlay is not None:
                mask_display.set_data(overlay)
                try:
                    mask_display.set_extent((x_left, x_right, y_bottom, y_top))
                except Exception:
                    pass
                # Ensure overlay is on top
                try:
                    mask_display.set_zorder(image_display.get_zorder() + 1)
                except Exception:
                    pass
                mask_display.set_visible(True)
            else:
                mask_display.set_visible(False)
        else:
            mask_display.set_visible(False)

        # Force plain tick labels on x-axis: no scientific, no offset string
        try:
            from matplotlib.ticker import ScalarFormatter
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
            # Alternatively, ensure style plain
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        except Exception:
            pass

        # Update title and labels
        ax.set_title(f'[{index+1}/{len(fits_files)}] {os.path.basename(path)}  (←/→:step, J:jump-to, Q/Esc:quit)')
        ax.set_xlabel('Time since first sample (s)')
        ax.set_ylabel('Channel (index)')

        # Do not draw here; draw synchronously in navigation to avoid frame skipping
        state['idx'] = index

    def _do_step(direction):
        """Perform one navigation step synchronously (direction in {-1, +1}).
        Ensures the frame is drawn before accepting another step.
        """
        if direction == 0:
            return
        cur = state['idx']
        target = cur + (1 if direction > 0 else -1)
        target = max(0, min(target, len(fits_files) - 1))
        if target == cur:
            state['pending'] = 0
            return
        state['busy'] = True
        show(target)
        # Force a synchronous redraw to avoid event pile-up when holding keys
        fig.canvas.draw()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        # Yield briefly to the UI loop to ensure the image appears
        plt.pause(0.001)
        state['busy'] = False
        # If a new request came in while drawing, coalesce and process once
        if state['pending'] != 0:
            pending_dir = state['pending']
            state['pending'] = 0
            _do_step(pending_dir)

    def _navigate(direction):
        """Handle navigation requests. If drawing is in progress, remember last direction."""
        if state['busy']:
            state['pending'] = 1 if direction > 0 else -1
            return
        _do_step(direction)

    def on_key(event):
        def _prompt_goto_index():
            """Open a small dialog to ask for an index (1-based). Returns int or None on cancel/error."""
            try:
                import tkinter as tk
                from tkinter import simpledialog
                root = tk.Tk()
                root.withdraw()
                value = simpledialog.askinteger(
                    title="跳转到编号",
                    prompt=f"Jump to (1-{len(fits_files)}):",
                    minvalue=1,
                    maxvalue=len(fits_files),
                    parent=root
                )
                try:
                    root.update()
                except Exception:
                    pass
                root.destroy()
                return value
            except Exception as e:
                print(f"[Info] 无法显示跳转输入框：{e}")
                return None

        key = event.key
        if key in ('right', 'n', ' '):
            _navigate(+1)
        elif key in ('left', 'p', 'backspace'):
            _navigate(-1)
        elif key in ('j', 'J'):
            if not state['busy']:
                value = _prompt_goto_index()
                if value is not None:
                    target = max(1, min(value, len(fits_files))) - 1
                    show(target)
                    fig.canvas.draw(); plt.pause(0.001)
        elif key in ('q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    show(0)
    fig.canvas.draw(); plt.pause(0.001)
    plt.show()

def _choose_directory_via_gui(initial_dir=None, title="选择包含 FITS 文件的文件夹"):
    """Try to open a folder picker dialog using Tkinter. Returns path or None if canceled/fails."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        # On headless systems, this may raise a TclError
        root = tk.Tk()
        root.withdraw()
        # Explicit keyword args to satisfy type checkers
        init_dir = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd()
        selected = filedialog.askdirectory(parent=root,
                                           title=title,
                                           initialdir=init_dir,
                                           mustexist=False)
        try:
            root.update()
        except Exception:
            pass
        root.destroy()
        if selected:
            return selected
        return None
    except Exception as e:
        # GUI unavailable or other error; fallback handled by caller
        print(f"[Info] 无法打开图形化目录选择器，将回退到命令行参数/默认路径。原因：{e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize FITS files in a directory.')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing FITS files to visualize. If not set, a folder dialog will pop up.')
    parser.add_argument('--browse', action='store_true',
                        help='Force opening a folder selection dialog even if --dir is provided.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-frame loading logs (default: off).')
    parser.add_argument('--mask', nargs='?', const=True, default=False,
                        help='Enable mask overlay. Optionally provide MASK_DIR; if omitted, a folder dialog will ask for it.')
    parser.add_argument('--mask-alpha', type=float, default=0.35,
                        help='Alpha (opacity) for mask overlay in [0,1]. Default: 0.35')
    args = parser.parse_args()

    DEFAULT_DIR = "/home/cbm/deRFI/output"

    dir_path = None
    # If --browse is requested or --dir not provided, try GUI first
    if args.browse or not args.dir:
        dir_path = _choose_directory_via_gui(initial_dir=args.dir or os.getcwd())

    # If GUI canceled or failed, fallback to provided --dir
    if not dir_path and args.dir:
        dir_path = args.dir

    # Final fallback to the previous hardcoded default
    if not dir_path:
        print(f"[Info] 使用默认路径：{DEFAULT_DIR}")
        dir_path = DEFAULT_DIR

    # Validate path exists
    if not os.path.isdir(dir_path):
        print(f"[Error] 目标路径不存在：{dir_path}")
        raise SystemExit(1)

    # Resolve mask directory if overlay enabled
    mask_dir = None
    if args.mask:
        if isinstance(args.mask, str) and args.mask:
            mask_dir = args.mask
        else:
            mask_dir = _choose_directory_via_gui(initial_dir=dir_path, title="选择包含 掩码PNG 的文件夹")
        if mask_dir and not os.path.isdir(mask_dir):
            print(f"[Warn] 提供的掩码路径无效：{mask_dir}，将禁用掩码叠加。")
            mask_dir = None

    # Clamp mask alpha
    mask_alpha = max(0.0, min(1.0, float(args.mask_alpha)))

    test_load_fits_image(dir_path, verbose=args.verbose, mask_dir=mask_dir, mask_alpha=mask_alpha)