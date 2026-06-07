#!/usr/bin/env python3
"""
Visualize training data (FITS image and corresponding PNG mask).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import fitsio
import cv2

def load_fits_image(fits_path):
    """
    Load raw image data from FITS file.
    Returns (image, tbin), where tbin is the time width of each time sample (seconds).
    """
    with fitsio.FITS(fits_path, 'r') as fits:
        fits_header = fits[1].read_header()
        fits_data = fits[1].read()
        
    nchan = int(fits_header["NCHAN"])
    tbin = float(fits_header.get("TBIN", 1.0))

    rows = fits_data.shape[0] if hasattr(fits_data, 'shape') else len(fits_data)
    pieces = []
    for r in range(rows):
        raw = np.asarray(fits_data[r]["DATA"])
        nsamp_row = raw.size // nchan
        arr = raw.reshape(nsamp_row, nchan).astype(np.float32, copy=False)

        dat_scl = np.asarray(fits_data[r]["DAT_SCL"], dtype=np.float32)
        dat_offs = np.asarray(fits_data[r]["DAT_OFFS"], dtype=np.float32)
        
        arr *= dat_scl[np.newaxis, :]
        arr += dat_offs[np.newaxis, :]
        pieces.append(arr)

    data = np.vstack(pieces)
    
    # Transpose to (nchan, nsamp)
    data = data.T
    
    # Flip up/down to match standard display orientation (high freq at top)
    data = np.flipud(data)
    
    return data, tbin

def visualize_training_data(image_path, mask_path, output_path=None):
    """
    Visualize a FITS image and its corresponding PNG mask as an overlay,
    with marginal standard deviation (sigma) plots.
    """
    print(f"Loading image: {image_path}")
    img_data, tbin = load_fits_image(image_path)
    
    print(f"Loading mask: {mask_path}")
    mask_data = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_data is None:
        raise ValueError(f"Failed to load mask from {mask_path}")
        
    # Ensure mask matches image shape
    if img_data.shape != mask_data.shape:
        print(f"Warning: Shape mismatch. Image: {img_data.shape}, Mask: {mask_data.shape}")
        # Resize mask to match image if necessary
        if img_data.shape != mask_data.shape:
            mask_data = cv2.resize(mask_data, (img_data.shape[1], img_data.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Calculate marginal standard deviations excluding RFI pixels (mask > 0)
    masked_img = np.ma.masked_where(mask_data > 0, img_data)
    sigma_time = np.ma.std(masked_img, axis=0)  # Std dev along frequency axis
    sigma_freq = np.ma.std(masked_img, axis=1)  # Std dev along time axis
    
    # Identify rows (frequency channels) containing Horizontal RFI (class 1)
    # Mask these rows for the right sigma plot as requested previously
    horiz_mask = np.any(mask_data == 1, axis=1)
    sigma_freq_filtered = sigma_freq.copy()
    sigma_freq_filtered[horiz_mask] = np.nan

    # Create figure with GridSpec for marginal plots
    plt.rcParams.update({'font.size': 36})
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.85, bottom=0.1, top=0.85,
                          wspace=0.0, hspace=0.0)

    # Main overlay plot
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Top marginal plot (sigma over time)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    
    # Right marginal plot (sigma over frequency)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Plot main overlay
    vmin, vmax = np.percentile(img_data, [1, 99])
    extent = (0, 50.33, 1031.25, 1468.75)
    ax_main.imshow(img_data, aspect='auto', cmap='gist_heat', vmin=vmin, vmax=vmax, extent=extent)
    
    # Overlay mask with alpha
    from matplotlib.colors import ListedColormap
    class_colors = {
        0: (0.0, 0.0, 0.0, 0.0),  # Transparent background
        1: (0.0, 1.0, 1.0, 1.0),  # cyan
        2: (1.0, 0.0, 1.0, 1.0),  # magenta
        3: (0.0, 1.0, 0.0, 1.0),  # bright green
        4: (1.0, 1.0, 0.0, 1.0),  # yellow
        5: (1.0, 0.0, 0.0, 1.0),  # red
        6: (0.0, 1.0, 0.0, 1.0),  # bright green
        7: (1.0, 0.6, 0.0, 1.0),  # orange
        8: (0.8, 0.0, 0.0, 1.0),  # dark red
    }
    colors = [class_colors.get(i, (0.5, 0.5, 0.5, 1.0)) for i in range(10)]
    custom_cmap = ListedColormap(colors)

    mask_overlay = np.ma.masked_where(mask_data == 0, mask_data)
    im_mask = ax_main.imshow(mask_overlay, aspect='auto', cmap=custom_cmap, alpha=0.8, interpolation='nearest', vmin=0, vmax=9, extent=extent)
    
    ax_main.set_xlabel('Time(ms)')
    ax_main.set_ylabel('Frequency(MHz)')
    ax_main.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_main.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    # Plot top marginal (sigma over time)
    time_samples = np.linspace(0, 50.33, len(sigma_time))
    ax_top.plot(time_samples, sigma_time, color='black', linewidth=1)
    ax_top.set_ylabel(r'Std Dev $\sigma_\text{y}$')
    ax_top.xaxis.tick_top()
    ax_top.xaxis.set_label_position('top')
    ax_top.set_xlabel('Time(ms)')
    ax_top.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_top.tick_params(labelbottom=False)  # Hide bottom labels (shared with ax_main)
    ax_top.grid(True, alpha=0.3)

    # Plot right marginal (sigma over frequency)
    freq_channels = np.linspace(1468.75, 1031.25, len(sigma_freq))
    ax_right.plot(sigma_freq_filtered, freq_channels, color='black', linewidth=1)
    ax_right.set_xlabel(r'Std Dev $\sigma_\text{x}$')
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position('right')
    ax_right.set_ylabel('Frequency(MHz)')
    ax_right.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_right.tick_params(labelleft=False)  # Hide left labels (shared with ax_main)
    ax_right.grid(True, alpha=0.3)
    
    # Add a legend for the mask classes (matching plot_multisub.py)
    # Place it in the empty top-right corner (gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off')
    
    import matplotlib.patches as mpatches
    class_names = {
        1: 'Horizontal',
        2: 'Vertical',
        3: 'Point',
        4: 'Block',
        5: 'Pulsar',
    }
    handles = [mpatches.Patch(color=class_colors[cid][:3], label=class_names[cid]) for cid in range(1, 6)]
    # Use slightly smaller fontsize for legend and move it up-right
    # Increase vertical anchor to move legend further upward to avoid overlap
    ax_legend.legend(handles=handles, loc='center', bbox_to_anchor=(0.6, 0.88), 
                     title='RFI Classes', fontsize=28, title_fontsize=30, frameon=True)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize training data (FITS image and PNG mask)")
    parser.add_argument("-i", "--image", required=True, help="Path to the FITS image file")
    parser.add_argument("-m", "--mask", required=True, help="Path to the corresponding PNG mask file")
    parser.add_argument("-o", "--output", help="Path to save the visualization (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
        
    if not os.path.exists(args.mask):
        print(f"Error: Mask file not found: {args.mask}")
        sys.exit(1)
        
    visualize_training_data(args.image, args.mask, args.output)

if __name__ == "__main__":
    main()
