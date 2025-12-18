#!/usr/bin/env python3
"""
Simple FITS file visualization script - directly displays the output of the load_fits_image function
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

def test_load_fits_image(input_source, mode='dir', verbose: bool=False, mask_dir: Optional[str]=None, mask_alpha: float=0.7):
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
        print(f"Total Subints (Rows): {n_rows}")
        
        data_provider['count'] = n_rows
        data_provider['get_name'] = lambda i: f"Row/Subint {i}"
        data_provider['get_path'] = lambda i: f"row_{i}" # Virtual path used for mask
        # Load function
        def _load_idx(i):
            return load_psrfits_row(input_source, i)
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

    # Class names and color mapping (for legend and rendering), can be adjusted as needed
    # Convention: 0=background, 1=horizontal, 2=vertical, 3=point, 4=block
    class_names = {
        1: 'horizontal',
        2: 'vertical',
        6: 'point',
        4: 'block',
    }
    class_color_map = {
        1: (0.0, 1.0, 1.0),  # horizontal -> cyan
        2: (1.0, 0.0, 1.0),   # vertical -> magenta
        6: (0.0, 0.0, 1.0),   # point -> blue
        4: (0.0, 1.0, 0.0),   # block -> green
    }

    # Create figure and axes with expanded layout adding two sigma panels:
    #  - left panel: frequency mean profile (mean over time), shares y with main
    #  - top panel: time mean profile (mean over frequency), shares x with main
    #  - main panel: time-frequency image
    #  - colorbar panel: narrow colorbar
    #  - right panel: frequency sigma profile (std over time)
    #  - bottom panel: time sigma profile (std over frequency)
    fig = plt.figure(figsize=(13, 9))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        3, 5,
        width_ratios=[1.4, 8.0, 0.35, 1.4, 0.01],  # last tiny column for spacing
        height_ratios=[1.2, 8.0, 1.2],
        wspace=0,
        hspace=0,
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[1, 1])
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
            # Position fine-tuned to be near the left blank area
            legend_ax = fig.add_axes((0.02, 0.80, 0.17, 0.18), frameon=False)
            legend_ax.axis('off')
            legend_patches = []
            legend_labels = []
            for cid in sorted(class_names.keys()):
                color = class_color_map.get(cid, (0.5, 0.5, 0.5))
                patch = mpatches.Patch(color=color, label=f"{class_names[cid]} ({cid})")
                legend_patches.append(patch)
                legend_labels.append(f"{class_names[cid]} ({cid})")
            # Draw legend (manually placed in legend_ax)
            legend_ax.legend(handles=legend_patches, labels=legend_labels, loc='upper left', frameon=False)
        except Exception:
            legend_ax = None

    # Leave space for a figure-level title (suptitle) above the top panel
    try:
        fig.subplots_adjust(top=0.90, bottom=0.07, left=0.045, right=0.95)
    except Exception:
        pass

    # Main image and mask overlay
    image_display = ax_main.imshow(np.zeros((1, 1)), aspect='auto', cmap='gist_heat')
    mask_display = ax_main.imshow(np.zeros((1, 1, 4), dtype=float), aspect='auto', interpolation='nearest')
    colorbar = fig.colorbar(image_display, cax=cax)
    colorbar.set_label('Intensity')

    # Initialize profile lines (use dark colors to be visible on white background)
    top_line, = ax_top.plot([], [], color='tab:blue', lw=1.5)
    left_line, = ax_left.plot([], [], color='tab:blue', lw=1.5)
    right_sigma_line, = ax_right_sigma.plot([], [], color='tab:red', lw=1.5)
    bottom_sigma_line, = ax_bottom.plot([], [], color='tab:red', lw=1.5)
    # Initialize threshold lines for each panel: median ± 3σ (computed from the respective 1D profile)
    # Top (time mean profile): horizontal lines
    top_thr_low = ax_top.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    top_thr_high = ax_top.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    # Left (frequency mean profile): vertical lines
    left_thr_low = ax_left.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    left_thr_high = ax_left.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    # Right (frequency sigma profile): vertical lines
    right_thr_low = ax_right_sigma.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    right_thr_high = ax_right_sigma.axvline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    # Bottom (time sigma profile): horizontal lines
    bottom_thr_low = ax_bottom.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
    bottom_thr_high = ax_bottom.axhline(0.0, color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=5)
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
    # Main panel: remove left (y) and top/bottom (x) ticks/labels (bottom handled by bottom sigma panel)
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
        try:
            image_display.set_extent((x_left, x_right, y_bottom, y_top))
            ax_main.set_xlim(x_left, x_right)
            ax_main.set_ylim(y_bottom, y_top)
        except Exception:
            pass

        # If mask overlay enabled, try to overlay corresponding PNG by block index
        if mask_dir:
            # Derive basename from FITS filename and look for exact basename match in mask_map
            fits_bn = os.path.splitext(os.path.basename(path))[0]
            overlay = None
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
                            # Explicit color mapping for known classes to improve visibility:
                            # class index mapping expected: 0=background, 1=horizontal, 2=vertical, 3=point, 4=block
                            # Desired colors: point=blue, vertical=magenta, horizontal=orange, block=green
                            # Prefer the shared class_color_map defined above; fallback to local map
                            try:
                                color_map = class_color_map
                            except Exception:
                                color_map = {
                                    1: np.array([1.0, 0.55, 0.0]),
                                    2: np.array([1.0, 0.0, 1.0]),
                                    3: np.array([0.0, 0.0, 1.0]),
                                    4: np.array([0.0, 1.0, 0.0]),
                                }
                            # Fallback palette for any other unexpected class ids
                            palette = np.array([
                                [0.5, 0.5, 0.5],
                                [0.0, 1.0, 1.0],
                                [0.0, 0.85, 0.70],
                                [0.0, 0.70, 1.00],
                            ], dtype=float)

                            cls_ids = np.unique(mask_aligned)
                            cls_ids = cls_ids[cls_ids != 0]
                            for cid in cls_ids:
                                sel = (mask_aligned == cid)
                                cid_int = int(cid)
                                if cid_int in color_map:
                                    color = color_map[cid_int]
                                else:
                                    color = palette[(cid_int - 1) % len(palette)]
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
                mask_display.set_visible(True)
            else:
                mask_display.set_visible(False)
        else:
            mask_display.set_visible(False)

        # Update marginal profiles on top and left panels
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

        # Force plain tick labels on x-axis: no scientific, no offset string
        try:
            from matplotlib.ticker import ScalarFormatter
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax_main.xaxis.set_major_formatter(sf)
            # Alternatively, ensure style plain
            ax_main.ticklabel_format(axis='x', style='plain', useOffset=False)
        except Exception:
            pass

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

        fig.suptitle(f'[{index+1}/{total}] {name}  (←/→:step, J:jump-to, Esc:quit)')
        ax_main.set_xlabel('Time since first sample (s)')
        ax_main.set_ylabel('Channel (index)')
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
        elif key in ('escape',):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    show(0)
    fig.canvas.draw(); plt.pause(0.001)
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
    if not args.no_mask:
        if isinstance(args.mask, str):
            mask_dir = args.mask
        elif args.mask is True:
            # If user didn't specify mask path, ask for it
            print("Select Mask Directory (Cancel to disable overlay)...")
            init_mask_dir = target_path if mode == 'dir' else os.path.dirname(target_path)
            mask_dir = _choose_directory_via_gui(initial_dir=init_mask_dir, title="Select Mask Directory (Cancel to disable overlay)")
        
        if mask_dir and not os.path.isdir(mask_dir):
            print(f"[Warn] Invalid mask path provided: {mask_dir}, mask overlay will be disabled.")
            mask_dir = None

    test_load_fits_image(target_path, mode=mode, verbose=args.verbose, mask_dir=mask_dir, mask_alpha=args.mask_alpha)
