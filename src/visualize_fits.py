#!/usr/bin/env python3
"""
Simple FITS file visualization script - directly displays the output of the load_fits_image function
"""

import os
import sys
import numpy as np
import fitsio
import matplotlib


def _setup_mpl_backend():
    """彻底解决显示连接问题：优先尝试交互式后端，若环境不支持则自动降级为非交互式后端。"""
    env_backend = os.environ.get("DERFI_MPL_BACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            return
        except Exception:
            pass

    # 尝试 TkAgg 并验证是否真能连接到 Display
    try:
        import tkinter
        # 尝试创建一个极小的 Tk 根窗口来测试 Display 连接
        test_root = tkinter.Tk()
        test_root.withdraw() # 隐藏测试窗口
        test_root.destroy()  # 销毁测试窗口
        matplotlib.use("TkAgg", force=True)
        return
    except Exception:
        # 如果报错（如 couldn't connect to display），则静默降级到 Agg
        print("⚠️  Warning: No graphical display detected. Falling back to 'Agg' backend (saving to file only).")
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
    # Explicitly disable default quit bindings (often includes 'q')
    for _k in ('keymap.quit', 'keymap.quit_all', 'keymap.close'):
        if _k in matplotlib.rcParams:
            matplotlib.rcParams[_k] = []
    # Disable axis offset formatting like +1.37e+2 on x-axis
    matplotlib.rcParams['axes.formatter.useoffset'] = False
except Exception:
    pass

import matplotlib.pyplot as plt

def load_fits_image(fits_path):
    """
    Load raw image data from FITS file, without normalization.
    Returns (image, tbin), where tbin is the time width of each time sample (seconds).
    If TBIN/TSAMP is not in the header, tbin defaults to 1.0.
    """
    # Use a safer way to read FITS files
    with fitsio.FITS(fits_path, 'r') as fits:
            fits_header = fits[1].read_header()
            fits_data = fits[1].read()
        
    # More robust size inference: no longer assume nsamp = NBLOCKS*NSBLK, because the file may be written as a single row (DATA vector length=nsamp*nchan),
    # or written as multiple rows (each row DATA is a number of time samples). Concatenate by row uniformly.
    nchan = int(fits_header["NCHAN"])  # Number of channels

    # Restore DATA of each row as (nsamp_row, nchan), and apply DAT_SCL/DAT_OFFS of this row
    rows = fits_data.shape[0] if hasattr(fits_data, 'shape') else len(fits_data)
    pieces = []
    for r in range(rows):
        raw = np.asarray(fits_data[r]["DATA"])  # uint8 flat array
        if raw.size % nchan != 0:
            raise ValueError(f"DATA length {raw.size} is not divisible by NCHAN {nchan} at row {r}")
        nsamp_row = raw.size // nchan
        arr = raw.reshape(nsamp_row, nchan).astype(np.float32, copy=False)

        # Independent scale/offset per row (length= nchan)
        dat_scl = np.asarray(fits_data[r]["DAT_SCL"], dtype=np.float32)
        dat_offs = np.asarray(fits_data[r]["DAT_OFFS"], dtype=np.float32)
        if dat_scl.size != nchan or dat_offs.size != nchan:
            raise ValueError(f"DAT_SCL/DAT_OFFS length mismatch with NCHAN={nchan} at row {r}")
        # Apply scale offset (broadcast to time dimension)
        arr *= dat_scl[np.newaxis, :]
        arr += dat_offs[np.newaxis, :]
        pieces.append(arr)

    # Concatenate all rows along the time dimension -> (nsamp_total, nchan)
    if len(pieces) == 0:
        raise ValueError("No rows found in SUBINT table.")
    data = np.vstack(pieces)

    # Convert to (nchan, nsamp) and flip up/down to match existing display orientation
    image = np.flipud(data.T)

    # Extract time resolution
    tbin = None
    # Common key names: TBIN (PSRFITS), or TSAMP etc.
    if 'TBIN' in fits_header:
        tbin = float(fits_header['TBIN'])
    elif 'TSAMP' in fits_header:
        tbin = float(fits_header['TSAMP'])
    else:
        tbin = 1.0
        print("[Info] Header TBIN/TSAMP not found, tbin defaults to 1.0 seconds")

    return image, tbin

def load_psrfits_row(fits_path, row_idx):
    """
    Read a single Subint (row) from a large PSRFITS file and convert it to an image.
    """
    with fitsio.FITS(fits_path, 'r') as fits:
        # Usually SUBINT is the 2nd HDU (index 1)
        if 'SUBINT' in fits:
            hdu = fits['SUBINT']
        else:
            hdu = fits[1]
        header = hdu.read_header()
        
        # Read data of the specified row
        # fitsio returns a structured array, even if reading only one row, it is an array of length 1
        row_data = hdu.read(rows=[row_idx])
        
    # Extract single row record
    record = row_data[0]
    
    nchan = int(header["NCHAN"])
    nsblk = int(header["NSBLK"]) # Number of samples per subint
    
    # Process DATA
    raw_data = np.asarray(record["DATA"])
    
    # Automatically handle dimensions
    # Target is to get (nsblk, nchan) array (Time, Freq)
    # 1. If multi-dimensional array, squeeze first to remove extra 1s
    if raw_data.ndim > 1:
        raw_data = raw_data.squeeze()
        
    # 2. Judge based on shape
    if raw_data.ndim == 2:
        if raw_data.shape == (nsblk, nchan):
            # Already (Time, Freq)
            arr = raw_data.astype(np.float32)
        elif raw_data.shape == (nchan, nsblk):
            # Is (Freq, Time), need transpose
            arr = raw_data.T.astype(np.float32)
        else:
            # Shape incorrect, try forcing reshape to (nsblk, nchan)
            # Assume data stored as Time-major (consistent with load_fits_image logic)
            arr = raw_data.reshape(nsblk, nchan).astype(np.float32)
    else:
        # Flat array, prioritize trying reshape to (nsblk, nchan) (Time-major)
        try:
            arr = raw_data.reshape(nsblk, nchan).astype(np.float32)
        except ValueError:
            # If failed (e.g. nchan/nsblk definition mismatch), try another
            arr = raw_data.reshape(nchan, nsblk).T.astype(np.float32)

    # Process Scale and Offset
    # They are usually (NCHAN,) or (NCHAN*NPOL,)
    dat_scl = np.asarray(record["DAT_SCL"], dtype=np.float32)
    dat_offs = np.asarray(record["DAT_OFFS"], dtype=np.float32)
    
    # Ensure dimension match (take only first nchan, ignore multi-polarization if exists)
    if dat_scl.size >= nchan:
        dat_scl = dat_scl[:nchan]
    if dat_offs.size >= nchan:
        dat_offs = dat_offs[:nchan]
        
    # Apply scaling: arr is (nsblk, nchan), scl/offs is (nchan,)
    # Use broadcasting mechanism
    arr *= dat_scl[np.newaxis, :]
    arr += dat_offs[np.newaxis, :]

    # Convert to (nchan, nsamp) and flip up/down (low freq at bottom, high freq at top)
    # image final shape: (nchan, nsblk)
    image = np.flipud(arr.T)

    # Get time resolution
    tbin = 1.0
    if 'TBIN' in header:
        tbin = float(header['TBIN'])
    elif 'TSAMP' in header:
        tbin = float(header['TSAMP'])
        
    return image, tbin

import argparse
from typing import Optional

def test_load_fits_image(input_source, mode='dir', verbose: bool=False, mask_dir: Optional[str]=None, mask_alpha: float=0.7, blocksperread: int=1, show_class: Optional[str]=None, nomargin: bool=False):
    """Use left/right arrow keys for bidirectional browsing (←/→), press J to jump to index, Esc to quit.
    When verbose=True, per-frame loading logs will be printed, default off to reduce I/O.
    input_source: directory path (mode='dir') or file path (mode='file')
    mode: 'dir' | 'file'
    """
    import glob
    import re

    # --- Data Source Abstraction ---
    data_provider = {}
    
    if mode == 'dir':
        print(f"Searching for .fits and .fit files in: {input_source}")
        fits_files = glob.glob(os.path.join(input_source, '*.fits'))
        fits_files += glob.glob(os.path.join(input_source, '*.fit'))
        
        def get_block_number(filename):
            match = re.search(r'block(\d+)\.(fits|fit)', os.path.basename(filename))
            if match: return int(match.group(1))
            return -1
        fits_files.sort(key=get_block_number)
        
        if not fits_files:
            print(f"No FITS files found in: {input_source}")
            return

        data_provider['count'] = len(fits_files)
        data_provider['get_name'] = lambda i: os.path.basename(fits_files[i])
        data_provider['get_path'] = lambda i: fits_files[i] # Used for mask matching
        # Load function
        def _load_idx(i):
            return load_fits_image(fits_files[i])
        data_provider['load'] = _load_idx
        
    elif mode == 'file':
        if not os.path.isfile(input_source):
            print(f"File not found: {input_source}")
            return
        
        # Pre-read Header once to get number of rows
        with fitsio.FITS(input_source, 'r') as f:
            # Assume SUBINT is HDU 1 (the second one)
            if 'SUBINT' in f:
                hdu = f['SUBINT']
            else:
                hdu = f[1] # Fallback
            
            # header = hdu.read_header()
            n_rows = hdu.get_nrows()
            
        print(f"Opened Large FITS: {input_source}")
        print(f"Total Subints (Rows): {n_rows}, Blocks per read: {blocksperread}")

        # Store for later use (e.g., mask stitching)
        data_provider['psrfits_n_rows'] = int(n_rows)
        
        data_provider['count'] = (n_rows + blocksperread - 1) // blocksperread
        data_provider['get_name'] = lambda i: f"Rows {i*blocksperread} to {min((i+1)*blocksperread-1, n_rows-1)}"
        data_provider['get_path'] = lambda i: f"row_{i*blocksperread}" # Virtual path used for mask
        # Load function
        def _load_idx(i):
            if blocksperread <= 1:
                return load_psrfits_row(input_source, i)
            
            images = []
            tbin = 1.0
            for r in range(i * blocksperread, min((i + 1) * blocksperread, n_rows)):
                img, tb = load_psrfits_row(input_source, r)
                images.append(img)
                tbin = tb
            return np.concatenate(images, axis=1), tbin
        data_provider['load'] = _load_idx

    total_frames = data_provider['count']
    print(f"Total frames to view: {total_frames}")

    print("Right/Left Arrow: Step Forward/Back; J: Jump to index; Esc: Quit")

    # Optional: prepare mask file mapping by basename (filename without extension)
    mask_map = {}
    if mask_dir:
        import glob as _glob
        if not os.path.isdir(mask_dir):
            print(f"[Warn] Mask dir not found: {mask_dir}, overlay disabled.")
            mask_dir = None
        else:
            # Accept common case variants
            pngs = []
            for pat in ('*.png', '*.PNG', '*.Png'):
                pngs.extend(_glob.glob(os.path.join(mask_dir, pat)))
            # Map by basename (no extension) for exact matching with FITS basenames
            for p in pngs:
                bn = os.path.splitext(os.path.basename(p))[0]
                mask_map[bn] = p
            print(f"[Info] Found {len(mask_map)} mask PNG(s) in: {mask_dir}")

    # In PSRFITS mode, AI_RFI writes masks as: <fits_basename>_sub{row_idx}.png
    # Capture basename/row count once so we can match correctly.
    psrfits_bn = None
    psrfits_n_rows = 0
    if mode == 'file':
        psrfits_bn = os.path.splitext(os.path.basename(input_source))[0]
        try:
            psrfits_n_rows = int(data_provider.get('psrfits_n_rows', 0))
        except Exception:
            psrfits_n_rows = 0

    def _mask_key_candidates(base_name: str, idx: int):
        """Generate common basename candidates for a given mask index."""
        idx4 = f"{idx:04d}"
        idx_plain = str(idx)
        return [
            # 支持带有 _downsampX 这种新命名的后缀
            f"{base_name}_downsamp", # 前缀匹配
            f"{base_name}_sub{idx}",
            f"{base_name}_sub{idx4}",
            f"{base_name}_block{idx}",
            f"{base_name}_block{idx4}",
            f"mask_{idx}",
            f"mask_{idx4}",
            f"sub{idx}",
            f"sub{idx4}",
            f"block{idx}",
            f"block{idx4}",
            idx_plain,
            idx4,
        ]

    def _find_mask_path(mask_map_local: dict, base_name: str, idx: int):
        """Find mask path by trying multiple compatible naming schemes."""
        # 先尝试完全匹配（针对带有 downsamp 后缀的情况）
        # 现在的命名是 {base_name}_downsamp{X}_sub{idx:04d}.png
        # 我们遍历所有的 key，看是否有包含 base_name 和 sub{idx:04d} 的
        target_sub = f"sub{idx:04d}"
        for key in mask_map_local.keys():
            if base_name in key and target_sub in key:
                return mask_map_local[key]

        for key in _mask_key_candidates(base_name, idx):
            p = mask_map_local.get(key)
            if p:
                return p
        return None

    # Class names and color mapping (for legend and rendering), can be adjusted as needed
    # Standard mapping (0=bkg, 1=horiz, 2=vert, 3=point, 4=block, 5=pulsar)
    # Legacy/Simulator might use 6, 7, 8 for point, block, pulsar.
    class_names = {
        1: 'Horizontal',
        2: 'Vertical',
        3: 'Point',
        4: 'Block',
        5: 'Pulsar',
        6: 'Point',
        7: 'Block',
        8: 'Pulsar',
    }
    class_color_map = {
        1: (0.0, 1.0, 1.0),   # cyan
        2: (1.0, 0.0, 1.0),   # magenta
        3: (0.0, 0.5, 1.0),   # sky blue (point)
        4: (1.0, 1.0, 0.0),   # yellow (block)
        5: (1.0, 0.0, 0.0),   # red (pulsar)
        6: (0.0, 0.0, 1.0),   # blue (legacy point)
        7: (1.0, 0.6, 0.0),   # orange (legacy block)
        8: (0.8, 0.0, 0.0),   # dark red (legacy pulsar)
    }

    # Filter classes if show_class is provided
    allowed_classes = None
    if show_class:
        try:
            allowed_classes = {int(c) for c in show_class if c.isdigit()}
            # Filter maps for legend and consistency
            class_names = {k: v for k, v in class_names.items() if k in allowed_classes}
            class_color_map = {k: v for k, v in class_color_map.items() if k in allowed_classes}
        except Exception as e:
            print(f"[Warn] Failed to parse --showclass '{show_class}': {e}")

    # Create figure and axes with expanded layout adding two sigma panels:
    fig = plt.figure(figsize=(10, 7))
    from matplotlib.gridspec import GridSpec
    # Appropriately enlarge the main plot when margins are hidden
    if nomargin:
        wr = [1.0, 8.8, 0.35, 0.6, 0.01]
        hr = [0.6, 9.2, 0.6]
    else:
        wr = [1.4, 8.0, 0.35, 1.4, 0.01]
        hr = [1.2, 8.0, 1.2]
    
    gs = GridSpec(
        3, 5,
        width_ratios=wr,
        height_ratios=hr,
        wspace=0,
        hspace=0,
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[1, 1])
    # Shared axes only if margins are visible to avoid unwanted coupling when hidden
    if nomargin:
        ax_top = fig.add_subplot(gs[0, 1]); ax_top.axis('off')
        ax_bottom = fig.add_subplot(gs[2, 1]); ax_bottom.axis('off')
        ax_left = fig.add_subplot(gs[1, 0]); ax_left.axis('off')
        ax_right_sigma = fig.add_subplot(gs[1, 3]); ax_right_sigma.axis('off')
    else:
        ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
        ax_bottom = fig.add_subplot(gs[2, 1], sharex=ax_main)
        ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)
        ax_right_sigma = fig.add_subplot(gs[1, 3], sharey=ax_main)

    ax_blank_tl = fig.add_subplot(gs[0, 0]); ax_blank_tl.axis('off')
    ax_blank_bl = fig.add_subplot(gs[2, 0]); ax_blank_bl.axis('off')
    ax_blank_tr = fig.add_subplot(gs[0, 3]); ax_blank_tr.axis('off')
    ax_blank_br = fig.add_subplot(gs[2, 3]); ax_blank_br.axis('off')
    cax = fig.add_subplot(gs[1, 2])

    # If mask overlay is enabled, add legend in the top right corner (created once)
    legend_ax = None
    if mask_dir:
        try:
            import matplotlib.patches as mpatches
            # Place at top left of figure, using figure relative coordinates (avoid obscuring main plot)
            # Adjust legend position and size to avoid overlap with enlarged main plot
            if nomargin:
                legend_ax = fig.add_axes((0.005, 0.91, 0.18, 0.08), frameon=False)
            else:
                legend_ax = fig.add_axes((0.01, 0.83, 0.17, 0.17), frameon=False)
            legend_ax.axis('off')
            legend_patches = []
            legend_labels = []
            for cid in sorted(class_names.keys()):
                color = class_color_map.get(cid, (0.5, 0.5, 0.5))
                patch = mpatches.Patch(color=color, label=f"{class_names[cid]} ({cid})")
                legend_patches.append(patch)
                legend_labels.append(f"{class_names[cid]} ({cid})")
            # Draw legend with smaller font size
            legend_ax.legend(handles=legend_patches, labels=legend_labels, loc='upper left', frameon=False, fontsize='small')
        except Exception:
            legend_ax = None

    # Leave space for a figure-level title (suptitle) above the top panel
    try:
        if nomargin:
            fig.subplots_adjust(top=0.91, bottom=0.06, left=0.04, right=0.96)
        else:
            fig.subplots_adjust(top=0.90, bottom=0.07, left=0.045, right=0.95)
    except Exception:
        pass

    # Main image and mask overlay
    image_display = ax_main.imshow(np.zeros((1, 1)), aspect='auto', cmap='gist_heat')
    mask_display = ax_main.imshow(np.zeros((1, 1, 4), dtype=float), aspect='auto', interpolation='nearest')
    colorbar = fig.colorbar(image_display, cax=cax)
    colorbar.set_label('Intensity')

    # Initialize profile lines (use dark colors to be visible on white background)
    top_line, = ax_top.plot([], [], color='tab:blue', lw=1.5, visible=not nomargin)
    left_line, = ax_left.plot([], [], color='tab:blue', lw=1.5, visible=not nomargin)
    right_sigma_line, = ax_right_sigma.plot([], [], color='tab:red', lw=1.5, visible=not nomargin)
    bottom_sigma_line, = ax_bottom.plot([], [], color='tab:red', lw=1.5, visible=not nomargin)
    # Initialize threshold lines for each panel: median ± 3σ (computed from the respective 1D profile)
    # Top (time mean profile): horizontal lines
    top_thr_low = ax_top.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    top_thr_high = ax_top.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    # Left (frequency mean profile): vertical lines
    left_thr_low = ax_left.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    left_thr_high = ax_left.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    # Right (frequency sigma profile): vertical lines
    right_thr_low = ax_right_sigma.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    right_thr_high = ax_right_sigma.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    # Bottom (time sigma profile): horizontal lines
    bottom_thr_low = ax_bottom.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    bottom_thr_high = ax_bottom.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5, visible=not nomargin)
    # Configure ticks/labels per panels
    # Top panel: show x ticks/labels on TOP side, outward; hide bottom labels
    ax_top.xaxis.set_ticks_position('top')
    ax_top.xaxis.set_label_position('top')
    ax_top.tick_params(axis='x', which='both', direction='out', top=True, labeltop=True, bottom=False, labelbottom=False)
    # Move top-panel y-axis ticks/labels/label to RIGHT side (user request)
    ax_top.yaxis.set_ticks_position('right')
    ax_top.yaxis.set_label_position('right')
    ax_top.tick_params(axis='y', which='both', direction='out', left=False, labelleft=False, right=True, labelright=True)
    # Left panel: show y ticks/labels on LEFT side, outward; hide right side labels
    ax_left.tick_params(axis='y', which='both', direction='out', left=True, labelleft=True, right=False, labelright=False)
    # Left panel: move x-axis label/ticks back to BOTTOM
    ax_left.xaxis.set_ticks_position('bottom')
    ax_left.xaxis.set_label_position('bottom')
    ax_left.tick_params(axis='x', which='both', direction='out', top=False, labeltop=False, bottom=True, labelbottom=True)
    # Main panel: ticks handling
    if nomargin:
        ax_main.tick_params(axis='both', which='both', direction='out', 
                          left=True, labelleft=True, bottom=True, labelbottom=True)
        ax_main.set_xlabel('Time (s)')
        ax_main.set_ylabel('Channel')
    else:
        # remove left (y) and top/bottom (x) ticks/labels (bottom handled by bottom sigma panel)
        ax_main.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
        ax_main.tick_params(axis='x', which='both', top=False, labeltop=False, bottom=False, labelbottom=False)
    # Bottom sigma panel: show x ticks/labels, y-axis moved to RIGHT side (user request)
    ax_bottom.xaxis.set_ticks_position('bottom')
    ax_bottom.xaxis.set_label_position('bottom')
    ax_bottom.tick_params(axis='x', which='both', direction='out', bottom=True, labelbottom=True)
    ax_bottom.yaxis.set_ticks_position('right')
    ax_bottom.yaxis.set_label_position('right')
    ax_bottom.tick_params(axis='y', which='both', direction='out', left=False, labelleft=False, right=True, labelright=True)
    # Right sigma panel: customize similar to left but on right side
    ax_right_sigma.tick_params(axis='y', which='both', direction='out', left=False, labelleft=False, right=True, labelright=True)
    ax_right_sigma.xaxis.set_ticks_position('bottom')
    ax_right_sigma.xaxis.set_label_position('bottom')
    ax_right_sigma.tick_params(axis='x', which='both', direction='out', bottom=True, labelbottom=True)

    # idx: current index; busy: rendering in progress; pending: pending direction (-1/0/+1)
    # thr_mult: per-panel sigma multiplier for threshold lines
    state = {
        'idx': 0,
        'busy': False,
        'pending': 0,
        'mask_visible': True,  # 新增：控制 Mask 的显示状态
        'thr_mult': {
            'top': 3.0,     # time mean panel (top)
            'left': 3.0,    # freq mean panel (left)
            'right': 3.0,   # freq sigma panel (right)
            'bottom': 3.0,  # time sigma panel (bottom)
        }
    }

    def show(index):
        # Clamp index
        total = data_provider['count']
        index = max(0, min(index, total - 1))
        # If navigating to a different sample, reset per-panel threshold multipliers to defaults
        try:
            if index != state.get('idx', 0):
                state['thr_mult']['top'] = 3.0
                state['thr_mult']['left'] = 2.0
                state['thr_mult']['right'] = 3.0
                state['thr_mult']['bottom'] = 3.0
        except Exception:
            pass
        
        path = data_provider['get_path'](index)
        name = data_provider['get_name'](index)
        
        # Only check existence if it looks like a real file path (not virtual)
        if mode == 'dir' and not os.path.exists(path):
            print(f"Missing file: {path}")
            return

        if verbose:
            print(f"Loading: {name} ({index+1}/{total})")
            
        try:
            image, tbin = data_provider['load'](index)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return
            
        if image is None:
            print("Failed to load image (None returned)")
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

        # Set coordinate range:
        #  x axis: time relative to the 1st sample, block_index*nsamp*tbin + i*tbin (block is 0-based)
        #  y axis: channel 0..nchan
        nchan, nsamp = image.shape
        
        # Try to guess block index from filename if possible, else use index
        block_idx = index
        if mode == 'dir':
            import re as _re
            m = _re.search(r'block(\d+)\.(fits|fit)$', os.path.basename(path))
            if m:
                block_idx = int(m.group(1))
        
        x_left = (block_idx * nsamp) * tbin
        x_right = x_left + nsamp * tbin
        y_bottom, y_top = 0, nchan
        
        # Apply extent to BOTH data and mask layers to ensure alignment
        extent = (x_left, x_right, y_bottom, y_top)
        try:
            image_display.set_extent(extent)
            mask_display.set_extent(extent)
            ax_main.set_xlim(x_left, x_right)
            ax_main.set_ylim(y_bottom, y_top)
        except Exception:
            pass

        # If mask overlay enabled, try to overlay corresponding PNG
        if mask_dir:
            # In directory mode, path is a real FITS file; in PSRFITS mode, path is virtual (row_*)
            fits_bn = os.path.splitext(os.path.basename(path))[0] if mode == 'dir' else (psrfits_bn or str(path))
            overlay = None
            candidate_path = None

            if mode == 'file':
                # Match naming: <base>_sub{idx} or <base>_block{idx}
                # - blocksperread==1: idx == current subint (== index)
                # - blocksperread>1 : current frame covers rows [start..end]
                start_row = index * blocksperread
                candidate_path = _find_mask_path(mask_map, fits_bn, start_row)

                if candidate_path is None and blocksperread > 1:
                    # If user produced merged masks with a range suffix, try it as well
                    end_row = min((index + 1) * blocksperread - 1, (start_row + blocksperread - 1))
                    candidate_path = mask_map.get(f"{fits_bn}_sub{start_row}-{end_row}")
            else:
                # Directory mode: exact basename match
                candidate_path = mask_map.get(fits_bn)

            # If no exact match, try a couple of common variants (suffixes/prefixes)
            if candidate_path is None:
                # try with common suffixes
                for suf in ("_mask", "-mask", "_seg", "-seg"):
                    candidate_path = mask_map.get(fits_bn + suf)
                    if candidate_path:
                        break
            if candidate_path is None:
                # try stripping common prefixes from mask filenames (e.g., 'mask_{basename}.png')
                for key in mask_map.keys():
                    if key.endswith(fits_bn):
                        candidate_path = mask_map[key]
                        break
            if candidate_path:
                try:
                    def _read_mask_png(png_path: str):
                        """Read a mask PNG and return a 2D uint8 array of class ids."""
                        mask_idx_local = None
                        try:
                            from PIL import Image as _PIL_Image
                            _im = _PIL_Image.open(png_path)
                            if _im.mode == 'P':
                                mask_idx_local = np.array(_im, dtype=np.uint16)
                            elif _im.mode in ('L',):
                                mask_idx_local = np.array(_im, dtype=np.uint8)
                            elif _im.mode.startswith('I'):
                                mask_idx_local = np.array(_im, dtype=np.int32)
                            else:
                                mask_idx_local = np.array(_im.convert('L'), dtype=np.uint8)
                        except Exception:
                            import matplotlib.image as mpimg
                            mask_img = mpimg.imread(png_path)
                            if mask_img.ndim == 2:
                                mask_idx_local = (mask_img * 255.0 + 0.5).astype(np.uint8)
                            elif mask_img.ndim == 3:
                                mask_idx_local = (mask_img[..., 0] * 255.0 + 0.5).astype(np.uint8)

                        if mask_idx_local is None:
                            return None
                        if mask_idx_local.ndim != 2:
                            return None

                        # Backward compatibility (old AI_RFI saved id*255)
                        try:
                            # If all non-zero values are 255, it's likely a binary mask that should be class 1
                            # If they are multiples of some value, or just very large, we check the range.
                            mx = int(np.max(mask_idx_local))
                            if mx == 255:
                                # Special case: if it's 255, it might be a binary mask (class 1) or scaled class IDs.
                                # Check if there are values other than 0 and 255.
                                unique_vals = np.unique(mask_idx_local)
                                if len(unique_vals) == 2 and 0 in unique_vals and 255 in unique_vals:
                                    # Binary mask, map 255 to 1
                                    mask_idx_local = (mask_idx_local // 255).astype(np.uint8, copy=False)
                                elif mx > 20: 
                                    # Scaled IDs? Try to recover.
                                    mask_idx_local = (mask_idx_local // 255).astype(np.uint8, copy=False)
                        except Exception:
                            pass

                        return mask_idx_local.astype(np.uint8, copy=False)

                    # Read + (if needed) stitch masks for PSRFITS blocksperread>1
                    mask_idx = None
                    if mode == 'file' and blocksperread > 1:
                        start_row = index * blocksperread
                        # Cap by total row count if known
                        nrows_cap = psrfits_n_rows if isinstance(psrfits_n_rows, int) and psrfits_n_rows > 0 else (start_row + blocksperread)
                        end_row = min(start_row + blocksperread - 1, nrows_cap - 1)
                        pieces = []
                        for r in range(start_row, end_row + 1):
                            p = _find_mask_path(mask_map, fits_bn, r)
                            if not p:
                                if verbose:
                                    print(f"[Info] Missing mask for subint/block {r}: expected basename like {fits_bn}_sub{r} or {fits_bn}_block{r:04d}")
                                pieces = []
                                break
                            mi = _read_mask_png(p)
                            if mi is None:
                                pieces = []
                                break
                            pieces.append(mi)
                        if pieces:
                            # Expect each piece is (nchan, nsamp_single); stitch along time axis
                            mask_idx = np.concatenate(pieces, axis=1)
                    else:
                        mask_idx = _read_mask_png(candidate_path)

                    if mask_idx is not None and mask_idx.ndim == 2:
                        # Try to match orientation to displayed image (nchan x nsamp)
                        # Note: image is already np.flipud(data.T) which puts chan 0 at the bottom.
                        # Now our mask PNGs (from both C-core and Simulator) also put chan 0 at the bottom.
                        mh, mw = mask_idx.shape
                        ih, iw = image.shape
                        if (mh, mw) == (ih, iw):
                            mask_aligned = mask_idx
                        elif (mh, mw) == (iw, ih):
                            mask_aligned = mask_idx.T
                        else:
                            print(f"[Warn] Mask shape {mh}x{mw} mismatches image {ih}x{iw}; skip overlay for {block_idx}")
                            mask_aligned = None

                        if mask_aligned is not None and allowed_classes is not None:
                            # Filter mask content
                            mask_aligned = np.where(np.isin(mask_aligned, list(allowed_classes)), mask_aligned, 0)

                        if mask_aligned is not None:
                            rgba = np.zeros((ih, iw, 4), dtype=float)
                            cls_ids = np.unique(mask_aligned)
                            cls_ids = cls_ids[cls_ids != 0]
                            for cid in cls_ids:
                                sel = (mask_aligned == cid)
                                cid_int = int(cid)
                                if cid_int in class_color_map:
                                    color = class_color_map[cid_int]
                                else:
                                    # Fallback to some default
                                    color = (0.5, 0.5, 0.5)
                                rgba[..., 0][sel] = float(color[0])
                                rgba[..., 1][sel] = float(color[1])
                                rgba[..., 2][sel] = float(color[2])
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
                # 结合全局 state 和是否存在 overlay 来决定可见性
                mask_display.set_visible(state['mask_visible'])
            else:
                mask_display.set_visible(False)
        else:
            mask_display.set_visible(False)

        # Update marginal profiles on top and left panels
        if not nomargin:
            # Time profile: mean over frequency (axis=0) -> length nsamp
            time_profile = image.mean(axis=0)
            time_x = np.linspace(x_left + 0.5 * tbin, x_right - 0.5 * tbin, nsamp)
            top_line.set_data(time_x, time_profile)
            # Set y-limits with small padding
            if np.isfinite(time_profile).any():
                tmin_val = float(np.nanmin(time_profile))
                tmax_val = float(np.nanmax(time_profile))
                if not np.isfinite(tmin_val) or not np.isfinite(tmax_val):
                    tmin_val, tmax_val = 0.0, 1.0
                span = (tmax_val - tmin_val)
                pad = 0.05 * span
                if span <= 0:
                    pad = max(1.0, abs(tmax_val) * 0.05 + 1e-3)
                ax_top.set_ylim(tmin_val - pad, tmax_val + pad)
            ax_top.set_xlim(x_left, x_right)
            # Compute thresholds for top panel (time mean profile) and ensure visible
            try:
                valid = np.isfinite(time_profile)
                if valid.any():
                    med = float(np.nanmedian(time_profile))
                    sig = float(np.nanstd(time_profile))
                else:
                    med = 0.0
                    sig = 0.0
                mult = float(state['thr_mult'].get('top', 3.0))
                low = med - mult * sig
                high = med + mult * sig
                top_thr_low.set_ydata([low, low])
                top_thr_high.set_ydata([high, high])
                top_thr_low.set_visible(True); top_thr_high.set_visible(True)
                # Expand ylim to include thresholds if necessary
                y0, y1 = ax_top.get_ylim()
                new_min = min(y0, y1, low, high)
                new_max = max(y0, y1, low, high)
                if new_min < min(y0, y1) or new_max > max(y0, y1):
                    span = new_max - new_min
                    pad = 0.05 * span
                    if span <= 0:
                        pad = max(1.0, abs(new_max) * 0.05 + 1e-3)
                    ax_top.set_ylim(new_min - pad, new_max + pad)
            except Exception:
                top_thr_low.set_visible(True); top_thr_high.set_visible(True)

            # Frequency mean profile: mean over time (axis=1) -> length nchan
            freq_mean_profile = image.mean(axis=1)
            freq_y = np.linspace(y_top - 0.5, y_bottom + 0.5, nchan)
            left_line.set_data(freq_mean_profile, freq_y)
            # Frequency sigma profile: std over time (axis=1)
            freq_sigma_profile = image.std(axis=1)
            right_sigma_line.set_data(freq_sigma_profile, freq_y)
            # Set x-limits with small padding for mean panel
            if np.isfinite(freq_mean_profile).any():
                fmin_val = float(np.nanmin(freq_mean_profile))
                fmax_val = float(np.nanmax(freq_mean_profile))
                if not np.isfinite(fmin_val) or not np.isfinite(fmax_val):
                    fmin_val, fmax_val = 0.0, 1.0
                span_mean = (fmax_val - fmin_val)
                pad_mean = 0.05 * span_mean
                if span_mean <= 0:
                    pad_mean = max(1.0, abs(fmax_val) * 0.05 + 1e-3)
                ax_left.set_xlim(fmin_val - pad_mean, fmax_val + pad_mean)
            # Compute thresholds for left panel (frequency mean profile) and ensure visible
            try:
                valid = np.isfinite(freq_mean_profile)
                if valid.any():
                    med = float(np.nanmedian(freq_mean_profile))
                    sig = float(np.nanstd(freq_mean_profile))
                else:
                    med = 0.0
                    sig = 0.0
                mult = float(state['thr_mult'].get('left', 3.0))
                low = med - mult * sig
                high = med + mult * sig
                left_thr_low.set_xdata([low, low])
                left_thr_high.set_xdata([high, high])
                left_thr_low.set_visible(True); left_thr_high.set_visible(True)
                # Expand xlim to include thresholds if necessary
                x0, x1 = ax_left.get_xlim()
                new_min = min(x0, x1, low, high)
                new_max = max(x0, x1, low, high)
                if new_min < min(x0, x1) or new_max > max(x0, x1):
                    span = new_max - new_min
                    pad = 0.05 * span
                    if span <= 0:
                        pad = max(1.0, abs(new_max) * 0.05 + 1e-3)
                    ax_left.set_xlim(new_min - pad, new_max + pad)
            except Exception:
                left_thr_low.set_visible(True); left_thr_high.set_visible(True)
            # Set x-limits for sigma panel
            if np.isfinite(freq_sigma_profile).any():
                fs_min = float(np.nanmin(freq_sigma_profile))
                fs_max = float(np.nanmax(freq_sigma_profile))
                if not np.isfinite(fs_min) or not np.isfinite(fs_max):
                    fs_min, fs_max = 0.0, 1.0
                span_sig = (fs_max - fs_min)
                pad_sig = 0.05 * span_sig
                if span_sig <= 0:
                    pad_sig = max(1.0, abs(fs_max) * 0.05 + 1e-3)
                ax_right_sigma.set_xlim(fs_min - pad_sig, fs_max + pad_sig)
            ax_left.set_ylim(y_bottom, y_top)
            ax_right_sigma.set_ylim(y_bottom, y_top)
            # Compute thresholds for right sigma panel (frequency sigma profile) and ensure visible
            try:
                valid = np.isfinite(freq_sigma_profile)
                if valid.any():
                    med = float(np.nanmedian(freq_sigma_profile))
                    sig = float(np.nanstd(freq_sigma_profile))
                else:
                    med = 0.0
                    sig = 0.0
                mult = float(state['thr_mult'].get('right', 3.0))
                low = med - mult * sig
                high = med + mult * sig
                right_thr_low.set_xdata([low, low])
                right_thr_high.set_xdata([high, high])
                right_thr_low.set_visible(True); right_thr_high.set_visible(True)
                # Expand xlim to include thresholds if necessary
                x0, x1 = ax_right_sigma.get_xlim()
                new_min = min(x0, x1, low, high)
                new_max = max(x0, x1, low, high)
                if new_min < min(x0, x1) or new_max > max(x0, x1):
                    span = new_max - new_min
                    pad = 0.05 * span
                    if span <= 0:
                        pad = max(1.0, abs(new_max) * 0.05 + 1e-3)
                    ax_right_sigma.set_xlim(new_min - pad, new_max + pad)
            except Exception:
                right_thr_low.set_visible(True); right_thr_high.set_visible(True)

            # Update title and labels (use a figure-level title to avoid being covered by the top panel)
            # Bottom sigma profile: std over frequency (axis=0)
            time_sigma_profile = image.std(axis=0)
            bottom_sigma_line.set_data(time_x, time_sigma_profile)
            if np.isfinite(time_sigma_profile).any():
                ts_min = float(np.nanmin(time_sigma_profile))
                ts_max = float(np.nanmax(time_sigma_profile))
                if not np.isfinite(ts_min) or not np.isfinite(ts_max):
                    ts_min, ts_max = 0.0, 1.0
                span_ts = ts_max - ts_min
                pad_ts = 0.05 * span_ts
                if span_ts <= 0:
                    pad_ts = max(1.0, abs(ts_max) * 0.05 + 1e-3)
                ax_bottom.set_ylim(ts_min - pad_ts, ts_max + pad_ts)
            ax_bottom.set_xlim(x_left, x_right)
            # Compute thresholds for bottom sigma panel (time sigma profile) and ensure visible
            try:
                valid = np.isfinite(time_sigma_profile)
                if valid.any():
                    med = float(np.nanmedian(time_sigma_profile))
                    sig = float(np.nanstd(time_sigma_profile))
                else:
                    med = 0.0
                    sig = 0.0
                mult = float(state['thr_mult'].get('bottom', 3.0))
                low = med - mult * sig
                high = med + mult * sig
                bottom_thr_low.set_ydata([low, low])
                bottom_thr_high.set_ydata([high, high])
                bottom_thr_low.set_visible(True); bottom_thr_high.set_visible(True)
                # Expand ylim to include thresholds if necessary
                y0, y1 = ax_bottom.get_ylim()
                new_min = min(y0, y1, low, high)
                new_max = max(y0, y1, low, high)
                if new_min < min(y0, y1) or new_max > max(y0, y1):
                    span = new_max - new_min
                    pad = 0.05 * span
                    if span <= 0:
                        pad = max(1.0, abs(new_max) * 0.05 + 1e-3)
                    ax_bottom.set_ylim(new_min - pad, new_max + pad)
            except Exception:
                bottom_thr_low.set_visible(True); bottom_thr_high.set_visible(True)

            ax_top.set_ylabel(r'$\bar{x}_{Time}$')
            # Make top panel y-label horizontal on the right
            try:
                ax_top.yaxis.label.set_rotation(0)
                ax_top.yaxis.label.set_verticalalignment('center')
                ax_top.yaxis.label.set_horizontalalignment('left')
                ax_top.yaxis.set_label_coords(1.02, 0.5)
            except Exception:
                pass
            ax_left.set_xlabel(r'$\bar{x}_{Freq}$')
            ax_right_sigma.set_xlabel(r'$\sigma_{Time}$')
            ax_right_sigma.set_ylabel('Channel')
            ax_bottom.set_ylabel(r'$\sigma_{Freq}$')
            # Make bottom panel y-label horizontal on the right
            try:
                ax_bottom.yaxis.label.set_rotation(0)
                ax_bottom.yaxis.label.set_verticalalignment('center')
                ax_bottom.yaxis.label.set_horizontalalignment('left')
                ax_bottom.yaxis.set_label_coords(1.02, 0.5)
            except Exception:
                pass
            ax_bottom.set_xlabel('Time (s)')
        else:
            # When nomargin is True, Ensure ax_main has labels and ticks are shown
            ax_main.tick_params(axis='both', which='both', labelleft=True, labelbottom=True)
            ax_main.set_xlabel('Time (s)')
            ax_main.set_ylabel('Channel')

        fig.suptitle(f"File: {name} (Frame {index+1}/{total})")

        # Do not draw here; draw synchronously in navigation to avoid frame skipping
        state['idx'] = index

    def _do_step(direction):
        """Perform one navigation step synchronously (direction in {-1, +1}).
        Ensures the frame is drawn before accepting another step.
        """
        if direction == 0:
            return
        cur = state['idx']
        total = data_provider['count']
        target = cur + (1 if direction > 0 else -1)
        target = max(0, min(target, total - 1))
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
                total = data_provider['count']
                value = simpledialog.askinteger(
                    title="Jump to index",
                    prompt=f"Jump to (1-{total}):",
                    minvalue=1,
                    maxvalue=total,
                    parent=root
                )
                try:
                    root.update()
                except Exception:
                    pass
                root.destroy()
                return value
            except Exception as e:
                print(f"[Info] Cannot show jump input box: {e}")
                return None

        def _prompt_sigma(panel_key: str, initial: float) -> Optional[float]:
            """Prompt for a sigma multiplier (float). Returns value or None if canceled/unavailable."""
            try:
                import tkinter as tk
                from tkinter import simpledialog
                root = tk.Tk()
                root.withdraw()
                title_map = {
                    'top': 'Set threshold multiplier - Top (time mean)',
                    'left': 'Set threshold multiplier - Left (freq mean)',
                    'right': 'Set threshold multiplier - Right (freq σ)',
                    'bottom': 'Set threshold multiplier - Bottom (time σ)',
                }
                prompt = "Please enter σ multiplier (e.g. 2 means ±2σ):"
                value = simpledialog.askfloat(
                    title=title_map.get(panel_key, 'Set threshold multiplier'),
                    prompt=prompt,
                    initialvalue=float(initial),
                    minvalue=0.0,
                    parent=root
                )
                try:
                    root.update()
                except Exception:
                    pass
                root.destroy()
                return value
            except Exception as e:
                print(f"[Info] Cannot show threshold multiplier input box: {e}")
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
                    total = data_provider['count']
                    target = max(1, min(value, total)) - 1
                    show(target)
                    fig.canvas.draw(); plt.pause(0.001)
        elif key in ('w', 'W'):
            # Adjust top panel (time mean) sigma multiplier
            if not state['busy']:
                cur = float(state['thr_mult'].get('top', 3.0))
                val = _prompt_sigma('top', cur)
                if val is not None:
                    state['thr_mult']['top'] = max(0.0, float(val))
                    show(state['idx'])
                    fig.canvas.draw(); plt.pause(0.001)
        elif key in ('a', 'A'):
            # Adjust left panel (freq mean) sigma multiplier
            if not state['busy']:
                cur = float(state['thr_mult'].get('left', 3.0))
                val = _prompt_sigma('left', cur)
                if val is not None:
                    state['thr_mult']['left'] = max(0.0, float(val))
                    show(state['idx'])
                    fig.canvas.draw(); plt.pause(0.001)
        elif key in ('d', 'D'):
            # Adjust right panel (freq sigma) sigma multiplier
            if not state['busy']:
                cur = float(state['thr_mult'].get('right', 3.0))
                val = _prompt_sigma('right', cur)
                if val is not None:
                    state['thr_mult']['right'] = max(0.0, float(val))
                    show(state['idx'])
                    fig.canvas.draw(); plt.pause(0.001)
        elif key in ('s', 'S'):
            # Adjust bottom panel (time sigma) sigma multiplier
            if not state['busy']:
                cur = float(state['thr_mult'].get('bottom', 3.0))
                val = _prompt_sigma('bottom', cur)
                if val is not None:
                    state['thr_mult']['bottom'] = max(0.0, float(val))
                    show(state['idx'])
                    fig.canvas.draw(); plt.pause(0.001)
        elif key in ('m', 'M'):
            # 切换 Mask 显示状态
            if not state['busy']:
                state['mask_visible'] = not state['mask_visible']
                print(f"[Info] Mask visibility: {'ON' if state['mask_visible'] else 'OFF'}")
                show(state['idx'])
                fig.canvas.draw(); plt.pause(0.001)
        elif key in ('escape',):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    show(0)

    # 彻底解决：根据后端类型决定是弹窗还是保存预览图
    if matplotlib.get_backend().lower() == 'agg':
        preview_path = "output/visualize_preview.png"
        os.makedirs("output", exist_ok=True)
        fig.savefig(preview_path)
        print(f"\n📸 [Headless Mode] Initial frame saved to: {os.path.abspath(preview_path)}")
        print("   (Tip: In VS Code, Ctrl+Click the path above to view the image)")
        plt.close(fig)
    else:
        fig.canvas.draw()
        plt.pause(0.001)
        plt.show()

def _choose_file_via_gui(initial_dir=None, title="Select PSRFITS file"):
    """Try to open a file picker dialog using Tkinter. Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        init_dir = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd()
        selected = filedialog.askopenfilename(parent=root,
                                              title=title,
                                              initialdir=init_dir,
                                              filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")])
        try:
            root.update()
        except Exception:
            pass
        root.destroy()
        if selected:
            return selected
        return None
    except Exception as e:
        print(f"[Info] Cannot open GUI file selector: {e}")
        return None

def _choose_directory_via_gui(initial_dir=None, title="Select folder containing FITS files"):
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
        print(f"[Info] Cannot open GUI directory selector, falling back to command line args/default path. Reason: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize FITS files in a directory or a single large PSRFITS.')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing FITS files to visualize.')
    parser.add_argument('--psrfits', nargs='?', const=True, default=None,
                        help='Path to a large PSRFITS file to view row-by-row. If set without value, opens file dialog.')
    parser.add_argument('--blocksperread', type=int, default=1,
                        help='Number of subints to show at once in PSRFITS mode (default: 1).')
    parser.add_argument('--browse', action='store_true',
                        help='Force opening a selection dialog.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-frame loading logs (default: off).')
    parser.add_argument('--mask', nargs='?', const=True, default=True,
                        help='Enable mask overlay (default: on). Optionally provide MASK_DIR.')
    parser.add_argument('--no-mask', action='store_true',
                        help='Disable mask overlay (overrides --mask).')
    parser.add_argument('--mask-alpha', type=float, default=0.7,
                        help='Alpha (opacity) for mask overlay in [0,1]. Default: 0.7')
    parser.add_argument('--showclass', type=str, default=None, 
                        help='Optional string of class IDs to show (e.g. "12678").')
    parser.add_argument('--nomargin', action='store_true',
                        help='Do not plot the four marginal integration curves around the main plot.')
    args = parser.parse_args()

    mode = 'dir'
    target_path = None
    
    # Check if --psrfits is used
    if args.psrfits:
        mode = 'file'
        if isinstance(args.psrfits, str):
            target_path = args.psrfits
        else:
            # Flag present but no value -> open dialog
            target_path = _choose_file_via_gui()
            if not target_path:
                print("No file selected.")
                raise SystemExit(0)
    else:
        # Directory mode
        mode = 'dir'
        DEFAULT_DIR = "/home/cbm/deRFI/output"
        
        if args.dir:
            target_path = args.dir
        elif args.browse:
             target_path = _choose_directory_via_gui(initial_dir=os.getcwd())
        else:
             # No args provided -> try GUI, fallback to default
             target_path = _choose_directory_via_gui(initial_dir=DEFAULT_DIR)
             if not target_path:
                 target_path = DEFAULT_DIR
                 print(f"[Info] Using default path: {DEFAULT_DIR}")

    # Validate
    if target_path is None:
         print("[Error] No target path selected")
         raise SystemExit(1)

    if mode == 'dir':
        if not os.path.isdir(target_path):
            print(f"[Error] Target directory does not exist: {target_path}")
            raise SystemExit(1)
    else:
        if not os.path.isfile(target_path):
            print(f"[Error] Target file does not exist: {target_path}")
            raise SystemExit(1)

    # Handle Mask
    mask_dir = None
    # argparse default can't distinguish: "user didn't pass --mask" vs "user passed --mask".
    # We use argv presence to control behavior:
    # - PSRFITS mode: default OFF unless user explicitly passes --mask
    # - Directory mode: keep previous default ON (ask user if needed)
    argv_has_mask = ('--mask' in sys.argv)

    if not args.no_mask:
        if isinstance(args.mask, str):
            # User explicitly provided a mask directory via: --mask PATH
            mask_dir = args.mask
        elif args.mask is True:
            if mode == 'file':
                # PSRFITS mode: only enable overlay when user explicitly asked for it.
                if argv_has_mask:
                    mask_dir = os.path.join(os.getcwd(), "results", "AI_RFI")
            else:
                # Directory mode: keep default ON (ask user when --mask has no path)
                print("Select Mask Directory (Cancel to disable overlay)...")
                init_mask_dir = target_path
                mask_dir = _choose_directory_via_gui(initial_dir=init_mask_dir, title="Select Mask Directory (Cancel to disable overlay)")

        if mask_dir and not os.path.isdir(mask_dir):
            print(f"[Warn] Mask dir not found: {mask_dir}, overlay disabled.")
            mask_dir = None

    test_load_fits_image(target_path, mode=mode, verbose=args.verbose, mask_dir=mask_dir, mask_alpha=args.mask_alpha, blocksperread=args.blocksperread, show_class=args.showclass, nomargin=args.nomargin)
