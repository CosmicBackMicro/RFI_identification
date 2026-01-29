#!/usr/bin/env python3
"""RFISimulator: generate synthetic time–frequency data for RFI detection."""
import argparse
from datetime import datetime, timezone
import multiprocessing
import os
import time
from typing import Tuple

import fitsio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

BKG = 0             # background
CHAN_RFI = 1        # broadband channel RFI (Horizontal)
POINT_VERTICAL = 2  # vertical (broadband/Lightning)
POINT_POINT = 6     # point RFI (Random + Periodic)
POINT_BLOCK = 7     # block-like point RFI
PULSAR = 8          # pulsar signal (Pulse)

# ReadFASTData/mask.c Convention:
# 1: Horizontal, 2: Vertical, 6: Point, 7: Block, 8: Pulse


def add_pulsar_signal(
        data: np.ndarray, 
        t_start_abs: float,
        tbin: float,
        nchan: int,
        obsfreq: float,
        obsbw: float,
        period: float,
        dm: float,
        width_s: float = 0.02,
        amplitude: float = 20.0
    ) -> np.ndarray:
    """Inject dispersed pulsar signal into data block."""
    nsamp, _ = data.shape
    
    # Calculate frequency array (Low->High)
    chan_bw = obsbw / nchan
    freqs = (obsfreq - obsbw / 2.0) + (chan_bw / 2.0) + chan_bw * np.arange(nchan, dtype=np.float32)
    
    # k_DM constant in equivalent units (MHz^2 cm^3 s)
    # Delay(f) = k_DM * DM * (f^-2)
    # We calculate delay relative to the highest frequency (arrival time reference)
    f_ref = freqs[-1] # Highest freq
    k_dm = 4.148808e3
    
    # Delay in seconds for each channel relative to f_ref
    delays = k_dm * dm * (freqs**(-2) - f_ref**(-2)) # Shape: (nchan,)

    # Time indices for this block
    t_indices = np.arange(nsamp)
    t_abs = t_start_abs + t_indices * tbin  # Shape: (nsamp,)
    
    # Vectorize calculation over (Time, Freq)
    # t_abs shape: (nsamp, 1)
    # delays shape: (1, nchan)
    # Effective time at source = t_arrival - delay
    t_eff = t_abs[:, np.newaxis] - delays[np.newaxis, :]
    
    # Pulse phase phi = (t_eff % Period)
    phase = (t_eff % period)
    
    # Distance to nearest peak in phase domain (0 or Period)
    dist = np.minimum(phase, period - phase)
    
    # Gaussian intensity
    sigma = width_s
    intensity = amplitude * np.exp(-0.5 * (dist / sigma)**2)
    
    data += intensity.astype(data.dtype)
    return data


class RFIConfig:
    """Holds fixed RFI parameters for the entire FITS file to ensure consistency."""
    def __init__(self, nchan, nsamp, bg_sigma, seed):
        self.rng = np.random.default_rng(seed)
        self.nchan = nchan
        self.nsamp = nsamp
        self.bg_sigma = bg_sigma
        
        # 1. Periodic/Broadband Points Params (Fixed channels, amplitude, duty cycle)
        self.periodic_rfis = []
        count_periodic = 2
        for _ in range(count_periodic):
            p = {}
            p['amp_sigma'] = self.rng.random() * 4.0 + 1.0
            p['width'] = self.rng.integers(1, 20)
            p['duty'] = self.rng.random() * 0.1 + 0.05
            p['start_ch'] = int(self.rng.integers(0, max(1, nchan - p['width'] + 1)))
            p['end_ch'] = min(nchan, p['start_ch'] + p['width'])
            p['flag'] = POINT_POINT if p['duty'] < 0.95 else CHAN_RFI
            self.periodic_rfis.append(p)

        # 2. Channel RFI Params (Fixed frequencies and profiles)
        self.chan_rfis = []
        n_gaussian = 3
        n_uniform = 1
        modes = ['gaussian']*n_gaussian + ['uniform']*n_uniform
        self.rng.shuffle(modes)
        for mode in modes:
            c = {}
            c['mode'] = mode
            c['std_dev'] = self.rng.random() * 3.0 + 2.0
            while True:
                amp_sigma = (self.rng.random() * 10.0 - 5.0)
                if abs(amp_sigma) >= 1.0: break
            c['amplitude'] = amp_sigma * bg_sigma
            c['center_chan'] = self.rng.integers(0, nchan)
            w = self.rng.integers(3, 26)
            if w % 2 == 0: w += 1
            c['width'] = w
            self.chan_rfis.append(c)

    def print_config(self):
        """Print the RFI configuration details."""
        print("\n" + "="*50)
        print("🔍 RFI Configuration Summary")
        print("="*50)
        
        print(f"\n[1] Periodic/Broadband RFI (Count: {len(self.periodic_rfis)})")
        for i, p in enumerate(self.periodic_rfis):
            print(f"  #{i+1}: Channels [{p['start_ch']}-{p['end_ch']}] (Width={p['width']}) | "
                  f"Duty={p['duty']:.2f} | Amp={p['amp_sigma']:.2f}σ")

        print(f"\n[2] Channel/Band RFI (Count: {len(self.chan_rfis)})")
        for i, c in enumerate(self.chan_rfis):
            print(f"  #{i+1}: Center={c['center_chan']}, Width={c['width']} | "
                  f"Mode={c['mode']} | Amp={c['amplitude']/self.bg_sigma:.2f}σ")
            
        print(f"\n[3] Transient RFI (Per-Subint Randomized)")
        print(f"  - Random Points: ~1000 pixels")
        print(f"  - Blob RFI: 5 blobs")
        print("="*50 + "\n")

    def apply_consistent_rfi(self, data, mask, current_seed):
        """Apply the fixed RFI features to the data, using current_seed for time-variability (e.g. periodic on/off)."""
        # Local RNG for time-variable aspects (but using fixed structural parameters)
        sub_rng = np.random.default_rng(current_seed)
        
        # Apply Periodic
        for p in self.periodic_rfis:
            amp_val = p['amp_sigma'] * self.bg_sigma
            # Time behavior is random per subint
            on = sub_rng.random(self.nsamp) < p['duty']
            data[on, p['start_ch']:p['end_ch']] += amp_val
            # Only mark mask where RFI is actually present
            mask[on, p['start_ch']:p['end_ch']] = p['flag']

        # Apply Channels (Fixed structure)
        for c in self.chan_rfis:
            # Re-use the add_chan logic but with pre-calculated params
            # We need to call add_chan or duplicate logic. Since add_chan is stateless, we can call it.
            add_chan(data, mask, c['amplitude'], c['center_chan'], c['std_dev'], c['width'], c['mode'])

        return data, mask

def generate_background(nsamp: int, nchan: int, bg_mu: float, bg_sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian background and an empty mask with shape (nsamp, nchan)."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    data = rng.normal(loc=bg_mu, scale=bg_sigma, size=(nsamp, nchan)).astype(np.float32)
    mask = np.zeros_like(data, dtype=np.uint8)
    return data, mask


def add_many_periodic_points(data: np.ndarray, mask: np.ndarray, count: int, bg_sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Add several periodic point RFIs with random amplitude/width/duty."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    
    for i in range(count):
        rand_amp_sigma = rng.random() * 4.0 + 1.0  # amplitude in [1, 5) sigma
        rand_width = rng.integers(1, 20)           # width in [1, 20)
        rand_duty = rng.random() * 0.1 + 0.05      # duty cycle in [0.05, 0.15)

        amp_val = rand_amp_sigma * bg_sigma
        flag = POINT_POINT if rand_duty < 0.95 else CHAN_RFI

        start_ch = int(rng.integers(0, max(1, nchan - rand_width + 1)))
        end_ch = min(nchan, start_ch + rand_width)
        on = rng.random(nsamp) < rand_duty
        
        data[on, start_ch:end_ch] += amp_val
        # Only mark mask where RFI is actually present
        mask[on, start_ch:end_ch] = flag
    return data, mask

def add_chan(
        data: np.ndarray, mask: np.ndarray,
        amplitude: float, center_chan: float, std_dev: float, width: int,
        mode: str = 'gaussian'
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Add one band RFI (Gaussian or uniform) and return (data, mask)."""
    nsamp, nchan = data.shape
    if width % 2 == 0:
        raise ValueError("Width must be odd.")
    half_width = (width - 1) // 2
    start_ch = int(max(0, np.floor(center_chan) - half_width))
    end_ch = int(min(nchan, start_ch + width))
    CHAN_SUBTYPE = CHAN_RFI 
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mode == 'gaussian':
        ch_indices = np.arange(start_ch, end_ch)
        gauss_profile = np.exp(-0.5 * ((ch_indices - center_chan) / std_dev) ** 2)
        data[:, start_ch:end_ch] += gauss_profile * amplitude
        
    elif mode == 'uniform':
        data[:, start_ch:end_ch] += amplitude
    else:
        raise ValueError(f"Unknown mode: {mode}, must be 'gaussian' or 'uniform'")

    mask[:, start_ch:end_ch] = CHAN_SUBTYPE
    
    return data, mask
        
def add_many_random_points(
        data: np.ndarray, mask: np.ndarray, point_count: int, point_amp_sigma: float, bg_sigma: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Add random point RFIs and label them as POINT_RANDOM."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    if mask is None:
        mask = np.zeros_like(data, dtype=np.uint8)
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if point_count > 0:
        available = np.where(mask == 0)
        total_avail = available[0].size
        if total_avail < point_count:
            print(f"Warning: only {total_avail} unmarked pixels, but point_count={point_count}. Will use all available.")
            select_count = total_avail
        else:
            select_count = point_count
        idx = rng.choice(total_avail, size=select_count, replace=False)
        ts = available[0][idx]
        cs = available[1][idx]
        data[ts, cs] += (point_amp_sigma * bg_sigma)
        mask[ts, cs] = POINT_POINT
    return data, mask


def add_blob_point(
        data: np.ndarray, mask: np.ndarray,
        amplitude: float, x: int, y: int, sigma_x: float, sigma_y: float,
        mask_threshold_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Add one 2D Gaussian blob RFI around (x, y)."""
    nsamp, nchan = data.shape

    win_x_half = int(np.ceil(3 * sigma_x))
    win_y_half = int(np.ceil(3 * sigma_y))

    start_x = max(0, x - win_x_half)
    end_x = min(nsamp, x + win_x_half + 1)
    start_y = max(0, y - win_y_half)
    end_y = min(nchan, y + win_y_half + 1)

    if start_x >= end_x or start_y >= end_y:
        return data, mask

    xx, yy = np.meshgrid(np.arange(start_y, end_y), np.arange(start_x, end_x))

    gauss_val = amplitude * np.exp(-0.5 * (((yy - x) / sigma_x)**2 + ((xx - y) / sigma_y)**2))

    data[start_x:end_x, start_y:end_y] += gauss_val.astype(data.dtype)
    
    threshold = amplitude * mask_threshold_ratio
    mask_window = gauss_val > threshold
    
    mask[start_x:end_x, start_y:end_y][mask_window] = POINT_POINT
    
    return data, mask


def add_many_blob_points(
        data: np.ndarray, mask: np.ndarray,
        count: int, bg_sigma: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Add several Gaussian blob RFIs with random parameters."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape

    for i in range(count):
        rand_x = rng.integers(0, nsamp)
        rand_y = rng.integers(0, nchan)
        # Modified: Lower amplitude range to avoid overly severe RFI
        rand_amplitude = rng.uniform(3, 10) * bg_sigma
        rand_sigma_x = rng.uniform(1.5, 2.0)
        rand_sigma_y = rng.uniform(1.5, 4.0)

        data, mask = add_blob_point(
            data, mask,
            amplitude=rand_amplitude,
            x=rand_x, y=rand_y,
            sigma_x=rand_sigma_x, sigma_y=rand_sigma_y
        )
    
    return data, mask


def plot_synth_result(data: np.ndarray, mask: np.ndarray, plot_path: str) -> None:
    """Plot data/mask preview and save as PNG."""
    fig = plt.figure(figsize=(15, 8))  # widened to accommodate mask
    gs = gridspec.GridSpec(
        3, 6, 
        hspace=0, wspace=0, 
        figure=fig,
        width_ratios =  [0.5, 1, 1, 0.1, 0.2, 2],
        height_ratios = [0.7, 1, 1]
    )

    labels_fontsize = 10
    ax_time = fig.add_subplot(gs[0, 1:3])
    time_profile = data.mean(axis=1)  # (nsamp,)
    ax_time.plot(np.arange(data.shape[0]), time_profile, color='blue', linewidth=0.5)
    ax_time.set_title('Time Profile', fontsize=labels_fontsize)
    ax_time.set_xlabel('Time samples', fontsize=labels_fontsize)
    ax_time.set_ylabel('Mean', fontsize=labels_fontsize)
    ax_time.tick_params(axis='x', which='both', direction='in', bottom=True, top=True, labeltop=True)

    ax_freq = fig.add_subplot(gs[1:3, 0])
    freq_profile = data.mean(axis=0)  # (nchan,)
    ax_freq.plot(freq_profile, np.arange(data.shape[1]), color='red', linewidth=0.5)
    ax_freq.set_title('Frequency Profile', fontsize=labels_fontsize)
    ax_freq.set_ylabel('Frequency channels', fontsize=labels_fontsize)
    ax_freq.set_xlabel('Mean', fontsize=labels_fontsize)

    ax_data = fig.add_subplot(gs[1:3, 1:3])
    im1 = ax_data.imshow(
        data.T, 
        aspect='auto', 
        origin='lower', 
        cmap='gist_heat', 
        vmin=float(np.percentile(data, 0.27)), 
        vmax=float(np.percentile(data, 99.73))
    )
    ax_data.set_xlabel('Time samples')
    ax_data.set_yticklabels([])

    cbar_ax = fig.add_subplot(gs[1:3, 3])
    plt.colorbar(im1, cax=cbar_ax, orientation='vertical')

    mask_cmap = 'tab10'
    ax_mask = fig.add_subplot(gs[1:3, 5])
    im_mask = ax_mask.imshow(
        mask.T,
        aspect='auto',
        origin='lower',
        cmap=mask_cmap,
        vmin=0,
        vmax=8,
        interpolation='none'
    )
    ax_mask.set_title('Mask')
    ax_mask.set_xlabel('Time samples')
    ax_mask.set_ylabel('Frequency channels')
    ax_mask.set_yticklabels([])  # hide y tick labels to save space

    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    mask_labels = {
        0: 'Background',
        1: 'Chan RFI (coarse)',
        2: 'Point RFI (coarse)',
        3: 'Chan RFI - Bright',
        4: 'Chan RFI - Dark',
        5: 'Chan RFI - Complex',
        6: 'Point RFI - Random',
        7: 'Point RFI - Periodic',
        8: 'Point RFI - Block',
        9: 'Point RFI - Vertical',
    }
    cmap = plt.get_cmap(mask_cmap)
    norm = mcolors.Normalize(vmin=0, vmax=9)
    
    handles = []
    for k, v in mask_labels.items():
        color = cmap(norm(k))
        patch = mpatches.Patch(color=color, label=f'{k}: {v}')
        handles.append(patch)
    ax_mask.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8, frameon=False)

    ax_time.set_xlim(ax_data.get_xlim())  # time profile x-range matches main image
    ax_freq.set_ylim(ax_data.get_ylim())  # frequency profile y-range matches main image
    ax_mask.set_xlim(ax_data.get_xlim())  # mask x-range matches main image
    ax_mask.set_ylim(ax_data.get_ylim())  # mask y-range matches main image

    im_mask.set_clim(0, 9)
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    




def quantize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_min = float(data.min())
    data_max = float(data.max())
    scale_val = (data_max - data_min) / 255.0 if data_max > data_min else 1.0
    offset_val = data_min
    nchan = data.shape[1]
    scale = np.full(nchan, scale_val, dtype=np.float32)
    offset = np.full(nchan, offset_val, dtype=np.float32)
    quantized = ((data - offset_val) / scale_val).clip(0, 255).astype(np.uint8)
    data_flat = quantized.flatten()
    return data_flat, offset, scale


def save_mask_png(mask: np.ndarray, mask_png_path: str) -> None:
    try:
        from PIL import Image
        # To match C-code convention (mask.c), we put Channel 0 at the bottom of the image.
        # mask is (nsamp, nchan), mask.T is (nchan, nsamp).
        # np.flipud(mask.T) puts nchan[0] at the last row.
        mask_img = np.flipud(mask.T).astype(np.uint8)
        im = Image.fromarray(mask_img, mode='L')
        im.save(mask_png_path, format='PNG')
        
    except Exception:
        import matplotlib.image as mpimg
        # Match same convention
        mpimg.imsave(mask_png_path, np.flipud(mask.T), cmap='gray', vmin=0, vmax=8)
        


def init_psrfits(
        psrfits_path: str,
        nchan: int,
        nsblk: int,
        tbin: float,
        obsfreq: float,
        obsbw: float,
        src_name: str
    ) -> Tuple[fitsio.FITS, np.dtype, np.ndarray, np.ndarray]:
    fitsio.write(psrfits_path, np.zeros((1, 1), dtype=np.uint8), clobber=True)
    fits = fitsio.FITS(psrfits_path, mode='rw')
    # fits[0].write_key('HDRVER', '3.4')
    # fits[0].write_key('FITSTYPE', 'PSRFITS')
    fits[0].write_key('DATE', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'))
    # fits[0].write_key('OBSERVER', 'SIM')
    # fits[0].write_key('PROJID', 'SIM')
    # fits[0].write_key('TELESCOP', 'FAST')
    # fits[0].write_key('BACKEND', 'SIM')
    # fits[0].write_key('OBS_MODE', 'SEARCH')
    fits[0].write_key('OBSFREQ', obsfreq)
    fits[0].write_key('OBSBW', obsbw)
    fits[0].write_key('OBSNCHAN', nchan)
    fits[0].write_key('SRC_NAME', src_name)

    chan_bw = obsbw / nchan
    fits[0].write_key('CHAN_BW', chan_bw)
    fits[0].write_key('NCHAN', nchan)

    dtype = np.dtype([
        ('TSUBINT', 'f8'),
        ('OFFS_SUB', 'f8'),
        ('LST_SUB', 'f8'),
        ('RA_SUB', 'f8'),
        ('DEC_SUB', 'f8'),
        ('GLON_SUB', 'f8'),
        ('GLAT_SUB', 'f8'),
        ('FD_ANG', 'f4'),
        ('POS_ANG', 'f4'),
        ('PAR_ANG', 'f4'),
        ('TEL_AZ', 'f4'),
        ('TEL_ZEN', 'f4'),
        ('DAT_FREQ', 'f4', nchan),
        ('DAT_WTS', 'f4', nchan),
        ('DAT_OFFS', 'f4', nchan),
        ('DAT_SCL', 'f4', nchan),
        ('DATA', 'u1', nsblk * nchan)
    ])
    fits.create_table_hdu(dtype=dtype, extname='SUBINT')
    fits[1].write_key('INT_TYPE', 'TIME')
    fits[1].write_key('INT_UNIT', 'SEC')
    fits[1].write_key('SCALE', 'FluxDen')
    fits[1].write_key('NPOL', 1)
    fits[1].write_key('POL_TYPE', 'AA+BB')
    fits[1].write_key('TBIN', tbin)
    fits[1].write_key('NBIN', 1)
    fits[1].write_key('NBITS', 8)
    fits[1].write_key('NSBLK', nsblk)
    fits[1].write_key('NCHAN', nchan)
    fits[1].write_key('CHAN_BW', chan_bw)
    # TDIM17 will be written at the end to avoid fitsio shape validation issues during append
    # fits[1].write_key('TDIM17', f'(1,{nchan},1,{nsblk})')
    fits[1].write_key('TUNIT17', 'Jy')
    fits[1].write_key('EXTNAME', 'SUBINT')

    dat_freq = (obsfreq - obsbw / 2.0) + (chan_bw / 2.0) + chan_bw * np.arange(nchan, dtype=np.float32)
    dat_wts = np.ones(nchan, dtype=np.float32)
    return fits, dtype, dat_freq, dat_wts


def append_psrfits_subint(
        fits: fitsio.FITS,
        dtype: np.dtype,
        data: np.ndarray,
        subint_index: int,
        tbin: float,
        dat_freq: np.ndarray,
        dat_wts: np.ndarray
    ) -> None:
    nsblk, nchan = data.shape
    data_flat, offset, scale = quantize_data(data)
    
    tsubint = nsblk * tbin
    offs_sub = (subint_index + 0.5) * tsubint
    row = {
        'TSUBINT': np.array([tsubint], dtype=np.float64),
        'OFFS_SUB': np.array([offs_sub], dtype=np.float64),
        'LST_SUB': np.array([0.0], dtype=np.float64),
        'RA_SUB': np.array([0.0], dtype=np.float64),
        'DEC_SUB': np.array([0.0], dtype=np.float64),
        'GLON_SUB': np.array([0.0], dtype=np.float64),
        'GLAT_SUB': np.array([0.0], dtype=np.float64),
        'FD_ANG': np.array([0.0], dtype=np.float32),
        'POS_ANG': np.array([0.0], dtype=np.float32),
        'PAR_ANG': np.array([0.0], dtype=np.float32),
        'TEL_AZ': np.array([0.0], dtype=np.float32),
        'TEL_ZEN': np.array([0.0], dtype=np.float32),
        'DAT_FREQ': dat_freq[np.newaxis, :],
        'DAT_WTS': dat_wts[np.newaxis, :],
        'DAT_OFFS': offset[np.newaxis, :],
        'DAT_SCL': scale[np.newaxis, :],
        'DATA': data_flat[np.newaxis, :]
    }
    fits[1].append(row)


def save_image_mask(data: np.ndarray, mask: np.ndarray, fits_path: str, mask_png_path: str) -> None:
    """Save data as FITS and mask as PNG."""
    nsamp, nchan = data.shape

    total_pixels = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    print("Mask summary before saving:")
    for u, c in zip(unique, counts):
        pct = (c / total_pixels) * 100.0
        print(f"  class {int(u)}: {c} pixels ({pct:.3f}%)")

    data_flat, offset, scale = quantize_data(data)

    fits = fitsio.FITS(fits_path, mode='rw', clobber=True)

    dtype = np.dtype([
        ('DAT_OFFS', 'f4', nchan),
        ('DAT_SCL', 'f4', nchan),
        ('DATA', 'u1', nsamp * nchan)
    ])
    fits.create_table_hdu(dtype=dtype, extname='SUBINT')

    row = np.array([(offset, scale, data_flat)], dtype=dtype)
    fits[1].write(row)

    fits[1].write_key('TBIN', 4.9152e-05, 'Time per sample (s)')
    fits[1].write_key('CHAN_BW', 0.4882812, 'Channel bandwidth (MHz)')
    fits[1].write_key('NSBLK', nsamp, 'Samples per block')
    fits[1].write_key('NCHAN', nchan, 'Frequency channels')
    fits[1].write_key('ORIGIN', 'RFISimulator', 'Source filename')
    fits[1].write_key('BLOCKIDX', 0, 'Original block index')
    fits[1].write_key('NBLOCKS', 1, 'Number of blocks per read')

    fits[1].write_key('TUNIT1', '', 'units for DAT_OFFS')
    fits[1].write_key('TUNIT2', '', 'units for DAT_SCL')
    fits[1].write_key('TUNIT3', 'RAW', 'units for DATA')

    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    fits[1].write_key('DATE', date_str, 'File creation date')

    fits.close()
    if mask_png_path:
        save_mask_png(mask, mask_png_path)


def add_many_chans(
        data: np.ndarray, mask: np.ndarray,
        n_gaussian: int, n_uniform: int, bg_sigma: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Add multiple band RFIs; choose how many use Gaussian vs uniform."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    _, nchan = data.shape

    rfi_modes = []
    rfi_modes.extend(['gaussian'] * n_gaussian)
    rfi_modes.extend(['uniform'] * n_uniform)
    
    rng.shuffle(rfi_modes)
    
    count = n_gaussian + n_uniform

    for i, mode in enumerate(rfi_modes):
        rand_std_dev = rng.random() * 3.0 + 2.0  # std_dev in [2.0, 5.0)
        # Sample amplitude from [-5, -1) U [1, 5) sigma to produce bright/dark bands
        while True:
            rand_amplitude_in_sigma = (rng.random() * 10.0 - 5.0)  # range [-5.0, 5.0)
            if abs(rand_amplitude_in_sigma) >= 1.0:
                break
        rand_amplitude = rand_amplitude_in_sigma * bg_sigma
        
        rand_center_chan = rng.integers(0, nchan)
        rand_width = rng.integers(3, 101)  # Modified: wider band RFI (up to 100 channels)
        if rand_width % 2 == 0:
            rand_width += 1  # ensure odd width
        
        data, mask = add_chan(data, mask, rand_amplitude, rand_center_chan, rand_std_dev, rand_width, mode=mode)
    
    return data, mask


def generate_sample(
        nsamp: int,
        nchan: int,
        bg_mu: float,
        bg_sigma: float,
        seed: int,
        rfi_config=None,
        pulsar_config=None,
        t_start=0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    data, mask = generate_background(nsamp, nchan, bg_mu, bg_sigma, seed)

    if pulsar_config:
        add_pulsar_signal(
            data=data,
            t_start_abs=t_start,
            tbin=pulsar_config['tbin'],
            nchan=nchan,
            obsfreq=pulsar_config['obsfreq'],
            obsbw=pulsar_config['obsbw'],
            period=pulsar_config['period'],
            dm=pulsar_config['dm'],
            width_s=pulsar_config['width'],
            amplitude=pulsar_config['flux']
        )

    if rfi_config:
        # Use consistent RFI parameters
        data, mask = rfi_config.apply_consistent_rfi(data, mask, seed + 1)
    else:
        # Random every time (Moderate settings)
        data, mask = add_many_periodic_points(data, mask, count=2, bg_sigma=bg_sigma, seed=seed + 1)
        data, mask = add_many_chans(data, mask, n_gaussian=2, n_uniform=1, bg_sigma=bg_sigma, seed=seed + 2)

    # Random points and blobs are considered transient/random, so they vary per subint
    data, mask = add_many_random_points(data, mask, point_count=400, point_amp_sigma=5.0, bg_sigma=bg_sigma, seed=seed + 3)

    data, mask = add_many_blob_points(data, mask, count=2, bg_sigma=bg_sigma, seed=seed + 4)
    
    # Add block and vertical RFI
    data, mask = add_block_rfi(data, mask, count=1, bg_sigma=bg_sigma, seed=seed + 5)
    data, mask = add_vertical_rfi(data, mask, count=2, bg_sigma=bg_sigma, seed=seed + 6)
    
    return data, mask


def generate_single_sample(args):
    """Generate one synthetic sample (multiprocessing-safe)."""
    nsamp, nchan, bg_mu, bg_sigma, seed, plot_out, fits_out, mask_png_out, do_plot, rfi_config, pulsar_config, t_start = args
    data, mask = generate_sample(nsamp, nchan, bg_mu, bg_sigma, seed, rfi_config, pulsar_config, t_start)

    if do_plot:
        plot_synth_result(data, mask, plot_out)

    save_image_mask(data, mask, fits_out, mask_png_out)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic RFI data.')
    parser.add_argument('--dataset', action='store_true', help='Write per-sample FITS files (legacy mode).')
    parser.add_argument('--nomask', action='store_true', help='Do not save mask PNGs.')
    args = parser.parse_args()

    config = {
        'num_samples': 500,   # Modified: 500 subints
        'base_seed': 98765,
        'plot_interval': 50,
        'nsamp': 1024,
        'nsblk': 1024,
        'nchan': 1792,
        'bg_mu': 0.0,
        'bg_sigma': 5.0,
        'output_dir': '/home/cbm/deRFI/simulation',
        'tbin': 4.9152e-05,
        'obsfreq': 1249.8779296875,
        'obsbw': 500.0,
        'src_name': 'SIMRFI'
    }
    os.makedirs(config['output_dir'], exist_ok=True)

    # Initialize RFI configuration once for the entire file/batch
    rfi_config = None 
    # rfi_config = RFIConfig(config['nchan'], config['nsamp'], config['bg_sigma'], config['base_seed'])
    # rfi_config.print_config()

    # Initialize Pulsar Injector Configuration
    pulsar_period = 1.0     # 1 second period
    pulsar_width  = 0.002   # 2 ms width (Sigma)
    pulsar_dm     = 50.0    # DM 50
    pulsar_flux   = 10.0    # Realistic flux (dimmer)
    
    # pulsar_config = {
    #     'period': pulsar_period,
    #     'dm': pulsar_dm,
    #     'width': pulsar_width,
    #     'flux': pulsar_flux,
    #     'tbin': config['tbin'],
    #     'obsfreq': config['obsfreq'],
    #     'obsbw': config['obsbw']
    # }
    pulsar_config = None 
    
    print(f"\n[Pulsar Configuration]")
    print(f"  Period: {pulsar_period} s")
    print(f"  DM:     {pulsar_dm} pc/cm^3")
    print(f"  Flux:   {pulsar_flux}")


    if args.dataset:
        tasks = []
        for i in range(config['num_samples']):
            current_seed = config['base_seed'] + i
            
            # Calculate absolute start time for this block
            t_start = i * config['nsblk'] * config['tbin']

            plot_out = f"{config['output_dir']}/plot_{i+5000}.png"
            fits_out = f"{config['output_dir']}/data_{i+5000}.fits"
            mask_png_out = f"{config['output_dir']}/mask_{i+5000}.png" if not args.nomask else None

            do_plot = (i % config['plot_interval'] == 0)

            task_args = (
                config['nsamp'], config['nchan'], config['bg_mu'], config['bg_sigma'], current_seed,
                plot_out, fits_out, mask_png_out, do_plot, rfi_config, pulsar_config, t_start
            )
            tasks.append(task_args)
        
        # Note: RFIConfig must be picklable for multiprocessing. The simple class structure should be fine.
        num_processes = multiprocessing.cpu_count()
        print(f"Starting parallel generation with {num_processes} processes...")

        total_start_time = time.time()

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(generate_single_sample, tasks)

        total_end_time = time.time()
        print(f"Finished generating {config['num_samples']} samples in {total_end_time - total_start_time:.2f} seconds.")
        return

    psrfits_path = os.path.join(config['output_dir'], 'simulation_psrfits.fits')
    fits, dtype, dat_freq, dat_wts = init_psrfits(
        psrfits_path,
        config['nchan'],
        config['nsblk'],
        config['tbin'],
        config['obsfreq'],
        config['obsbw'],
        config['src_name']
    )

    total_start_time = time.time()
    for i in range(config['num_samples']):
        current_seed = config['base_seed'] + i
        do_plot = (i % config['plot_interval'] == 0)

        # Progress update
        if i % 10 == 0 or i == config['num_samples'] - 1:
            elapsed = time.time() - total_start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (config['num_samples'] - i - 1) / rate if rate > 0 else 0
            print(f"\r⏳ Generating subint {i+1}/{config['num_samples']} "
                  f"({(i+1)/config['num_samples']*100:.1f}%) | "
                  f"Speed: {rate:.1f} subints/s | ETA: {eta:.0f}s", end='', flush=True)

        t_start = i * config['nsblk'] * config['tbin']

        data, mask = generate_sample(
            config['nsamp'],
            config['nchan'],
            config['bg_mu'],
            config['bg_sigma'],
            current_seed,
            rfi_config,
            pulsar_config,
            t_start
        )

        if do_plot:
            plot_out = f"{config['output_dir']}/plot_{i}.png"
            plot_synth_result(data, mask, plot_out)

        append_psrfits_subint(fits, dtype, data, i, config['tbin'], dat_freq, dat_wts)

        if not args.nomask:
            mask_png_out = f"{config['output_dir']}/mask_{i}.png"
            save_mask_png(mask, mask_png_out)
    
    print() # Newline after progress bar

    # Write TDIM at the end
    fits[1].write_key('TDIM17', f'(1,{config["nchan"]},1,{config["nsblk"]})')

    fits.close()
    total_end_time = time.time()
    print(f"PSRFITS saved: {psrfits_path}")
    print(f"Finished generating {config['num_samples']} samples in {total_end_time - total_start_time:.2f} seconds.")

def add_block_rfi(
        data: np.ndarray, mask: np.ndarray,
        count: int, bg_sigma: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Add rectangular area contains dense short-time/frequency-extended RFI."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    for i in range(count):
        # 确定block的总体矩形范围
        t0, f0, block_w, block_h = 0, 0, 0, 0
        retry = 0
        time_buffer = 40 # 增加时间缓冲区，防止不同类干扰靠得太近导致连通域合并
        while retry < 5:
            block_w = rng.integers(40, 150) # 块在时间上的总持续范围
            block_h = rng.integers(100, 400) # 块在频率上的总带宽范围
            t0 = rng.integers(0, max(1, nsamp - block_w))
            f0 = rng.integers(0, max(1, nchan - block_h))
            
            # 检查是否有缓冲区内的 Vertical 干扰
            check_t0 = max(0, t0 - time_buffer)
            check_t1 = min(nsamp, t0 + block_w + time_buffer)
            if not np.any(mask[check_t0 : check_t1, :] == POINT_VERTICAL):
                break
            retry += 1
        
        # 在该区域内密集生成具有频率延展的小点
        num_inner_bursts = rng.integers(100, 300) # 内部细节数量
        for _ in range(num_inner_bursts):
            bt = t0 + rng.integers(0, block_w)
            bf = f0 + rng.integers(0, block_h)
            
            # 细节特征：时间窄 (1-2)，频率延展 (10-40)
            burst_dt = rng.integers(1, 3) 
            burst_df = rng.integers(10, 40)
            
            t_end = min(nsamp, bt + burst_dt)
            f_end = min(nchan, bf + burst_df)
            
            amp = rng.uniform(5, 12) * bg_sigma
            data[bt:t_end, bf:f_end] += amp
            mask[bt:t_end, bf:f_end] = POINT_BLOCK
    return data, mask


def add_vertical_rfi(
        data: np.ndarray, mask: np.ndarray,
        count: int, bg_sigma: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Add vertical broadband impulse RFIs (single or clusters like lightning)."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    nsamp, nchan = data.shape
    for i in range(count):
        # 选取一个基准时间点，并避开已有的 Block 干扰
        t_base, cluster_count, cluster_window = 0, 0, 0
        retry = 0
        time_buffer = 40 # 增加时间缓冲区
        while retry < 10:
            t_base = rng.integers(0, nsamp)
            
            # 决定是簇效应（闪电）还是孤立脉冲
            is_cluster = rng.random() > 0.4  # 60% 概率为簇，40% 为孤立脉冲
            
            if is_cluster:
                cluster_count = rng.integers(3, 8) 
                cluster_window = rng.integers(10, 40)
            else:
                cluster_count = 1
                cluster_window = 1
            
            # 碰撞检测：避开 Block 并保持缓冲区距离
            check_t0 = max(0, t_base - time_buffer)
            check_t1 = min(nsamp, t_base + cluster_window + time_buffer)
            if not np.any(mask[check_t0 : check_t1, :] == POINT_BLOCK):
                break
            retry += 1
        
        for _ in range(cluster_count):
            t_pulse = t_base + rng.integers(0, cluster_window)
            if t_pulse >= nsamp: continue
            
            # 特征：极窄 (1-2个采样点)，全频带 (0:nchan)
            w = rng.integers(1, 3) 
            # 降低幅度范围：从 (15, 30) 降至 (6, 15)
            amp = rng.uniform(6, 15) * bg_sigma 
            
            t_end = min(nsamp, t_pulse + w)
            data[t_pulse:t_end, :] += amp
            mask[t_pulse:t_end, :] = POINT_VERTICAL
    return data, mask
    


if __name__ == '__main__':
    main()
