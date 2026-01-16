#!/usr/bin/env python3
"""
Pulse Mask Experiment Tool
用于验证和生成脉冲星脉冲Mask的实验脚本。
基于已知的 DM 和 Period，计算脉冲在时频图上的轨迹，并生成 Mask。
"""

import os
import numpy as np
import fitsio
import matplotlib
import sys


def _setup_mpl_backend() -> tuple[bool, str]:
    """Configure matplotlib backend.

    Returns:
        (interactive_ok, backend_name)
    """
    env_backend = os.environ.get("DERFI_MPL_BACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            return (env_backend.lower() != "agg", env_backend)
        except Exception as e:
            print(f"Warning: DERFI_MPL_BACKEND={env_backend} failed ({e}).")

    # Try an interactive backend first.
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as _plt
        fig_test = _plt.figure()
        _plt.close(fig_test)
        return True, "TkAgg"
    except Exception as e:
        print(f"Warning: Interactive backend 'TkAgg' failed ({e}).")

    # Fallback to a safe non-interactive backend.
    matplotlib.use("Agg", force=True)
    return False, "Agg"


INTERACTIVE_OK, MPL_BACKEND = _setup_mpl_backend()
import matplotlib.pyplot as plt

import argparse
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches

# 复用 visualize_fits.py 中的读取逻辑 (简化版)
def load_fits_data(fits_path, block_idx=0, blocks_per_read=1):
    """
    读取 FITS 数据，返回 (image, header_info)
    image shape: (nchan, nsamp) - 注意这里未做 flipud，保持原始频率顺序以便计算
    """
    header_info = {}
    
    with fitsio.FITS(fits_path, 'r') as fits:
        # 尝试定位 SUBINT 或 Primary HDU
        if 'SUBINT' in fits:
            hdu = fits['SUBINT']
        else:
            hdu = fits[1]
            
        header = hdu.read_header()
        n_rows = hdu.get_nrows()

        # 读取指定块
        # IMPORTANT: `block_idx` is the starting SUBINT row index (0-based).
        # The number of rows to read is `blocks_per_read` (aka --numtoread).
        start_row = int(block_idx)
        end_row = min(start_row + int(blocks_per_read), n_rows)

        if start_row >= n_rows:
            raise ValueError(f"Block index {block_idx} out of range (Total rows: {n_rows})")
            
        data_list = []
        tbin = 1.0
        
        # 读取 Header 信息
        if 'TBIN' in header: tbin = float(header['TBIN'])
        elif 'TSAMP' in header: tbin = float(header['TSAMP'])
        
        # 频率信息
        # PSRFITS 通常有 OBSFREQ (中心频率 MHz) and OBSBW (带宽 MHz)
        # 或者在 SUBINT 表中有 DAT_FREQ
        f_center = float(header.get('OBSFREQ', 1250.0))
        bw = float(header.get('OBSBW', 500.0))
        nchan = int(header['NCHAN'])
        
        header_info['tbin'] = tbin
        header_info['f_center'] = f_center
        header_info['bw'] = bw
        header_info['nchan'] = nchan
        header_info['n_subints'] = n_rows
        
        # 读取数据
        for r in range(start_row, end_row):
            row_data = hdu.read(rows=[r])[0]
            raw = np.asarray(row_data["DATA"])
            nsblk = int(header["NSBLK"])
            
            # Reshape logic (similar to visualize_fits.py)
            if raw.ndim > 1: raw = raw.squeeze()
            try:
                arr = raw.reshape(nsblk, nchan).astype(np.float32)
            except:
                arr = raw.reshape(nchan, nsblk).T.astype(np.float32)
                
            # Apply Scale/Offset
            dat_scl = np.asarray(row_data["DAT_SCL"], dtype=np.float32)
            dat_offs = np.asarray(row_data["DAT_OFFS"], dtype=np.float32)
            if dat_scl.size >= nchan: dat_scl = dat_scl[:nchan]
            if dat_offs.size >= nchan: dat_offs = dat_offs[:nchan]
            
            arr *= dat_scl[np.newaxis, :]
            arr += dat_offs[np.newaxis, :]
            
            data_list.append(arr.T) # (nchan, nsblk)
            
    full_data = np.concatenate(data_list, axis=1)

    # Expose where in the file timeline this view starts.
    # NOTE: In PSRFITS SUBINT table, each row contains NSBLK samples.
    header_info["start_row"] = int(start_row)
    header_info["end_row"] = int(end_row)
    header_info["nsblk"] = int(header.get("NSBLK", full_data.shape[1]))
    header_info["start_time_s"] = float(start_row) * float(header_info["nsblk"]) * float(header_info["tbin"])

    return full_data, header_info

def calculate_dispersion_delay(f_mhz, f_ref_mhz, dm):
    """
    计算色散延迟 (秒)
    dt = 4.148808e3 * DM * (f^-2 - f_ref^-2)
    """
    k_dm = 4.148808e3
    # 避免除以0
    f_mhz = np.maximum(f_mhz, 1e-6)
    f_ref_mhz = max(f_ref_mhz, 1e-6)
    return k_dm * dm * (1.0/(f_mhz**2) - 1.0/(f_ref_mhz**2))

def generate_mask(data_shape, freqs, tbin, dm, period, t0, width_s, 
                  enable_interpulse=False, interpulse_width_s=0.0, interpulse_t0=None,
                  lo_freq_mhz=None, hi_freq_mhz=None):
    """
    生成脉冲 Mask (支持主脉冲 + 可选的间脉冲)
    """
    nchan, nsamp = data_shape
    mask = np.zeros(data_shape, dtype=np.float32)
    
    # 频率范围过滤
    chan_valid = np.ones(nchan, dtype=bool)
    if lo_freq_mhz is not None:
        chan_valid &= (freqs >= lo_freq_mhz)
    if hi_freq_mhz is not None:
        chan_valid &= (freqs <= hi_freq_mhz)
    
    # 确定参考频率 (通常取最高频，因为高频先到)
    f_ref = np.max(freqs)
    
    # 计算每个通道相对于 f_ref 的延迟
    delays_sec = calculate_dispersion_delay(freqs, f_ref, dm)
    duration_sec = nsamp * tbin

    def draw_pulse_sequence(start_t0, w_s):
        width_samp = int(w_s / tbin)
        if width_samp < 1: width_samp = 1
        
        k_min = int(np.floor((-np.max(delays_sec) - start_t0) / period))
        k_max = int(np.ceil((duration_sec - start_t0) / period))
        
        delays_samp_float = delays_sec / tbin

        for k in range(k_min, k_max + 2):
            pulse_arrival_t0 = start_t0 + k * period
            t_starts_idx = (pulse_arrival_t0 / tbin + delays_samp_float).astype(int)
            
            for c in range(nchan):
                if not chan_valid[c]:
                    continue
                center_samp = t_starts_idx[c]
                s0 = max(0, center_samp - width_samp // 2)
                s1 = min(nsamp, center_samp + width_samp // 2)
                
                if s0 < s1:
                    mask[c, s0:s1] = 1.0

    # Draw Main Pulse
    draw_pulse_sequence(t0, width_s)
    
    # Draw Interpulse
    if enable_interpulse:
        it0 = interpulse_t0 if interpulse_t0 is not None else (t0 + 0.5 * period)
        draw_pulse_sequence(it0, interpulse_width_s)
                
    return mask

def main():
    parser = argparse.ArgumentParser(description="Experiment with Pulse Mask Generation")
    parser.add_argument("fits_file", help="Path to FITS file")
    parser.add_argument("--dm", type=float, default=50.0, help="Dispersion Measure")
    parser.add_argument("--period", type=float, default=1.0, help="Period in seconds")
    parser.add_argument(
        "--t0",
        type=float,
        default=0.1,
        help=(
            "Time offset of the first pulse at the reference (highest) frequency (s). "
            "This t0 is interpreted on the *full file timeline*; the script will automatically "
            "shift it to match the selected block/subint window."
        ),
    )
    parser.add_argument("--width", type=float, default=0.02, help="Pulse width (s)")
    parser.add_argument("--interpulse", action="store_true", help="Enable interpulse")
    parser.add_argument("--interpulseWidth", type=float, default=0.01, help="Interpulse width (s)")
    parser.add_argument("--interpulset0", type=float, default=None, help="Interpulse T0 (s). If None, defaults to t0 + 0.5*period")
    parser.add_argument("--pulselofreq", type=float, default=None, help="Lowest frequency (MHz) to mask the pulse")
    parser.add_argument("--pulsehifreq", type=float, default=None, help="Highest frequency (MHz) to mask the pulse")
    parser.add_argument(
        "--block",
        type=int,
        default=0,
        help=(
            "Starting SUBINT row index to read (0-based). "
            "With --numtoread=N, this reads rows [block, block+N)."
        ),
    )
    parser.add_argument(
        "--numtoread",
        type=int,
        default=1,
        help="Number of subints to read and concatenate per view (default: 1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path when running headless (default: auto name next to cwd).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Force non-interactive mode: save PNG and exit.",
    )
    parser.add_argument(
        "--nomask",
        action="store_true",
        help="Disable pulse mask overlay (for visual comparison).",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fits_file):
        print(f"File not found: {args.fits_file}")
        return

    print(f"Loading {args.fits_file}...")
    try:
        data, info = load_fits_data(args.fits_file, args.block, args.numtoread)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Debug: what rows/time window we actually read.
    print(
        "Read SUBINT rows: "
        f"[{int(info.get('start_row', -1))}, {int(info.get('end_row', -1))}) "
        f"(numtoread={args.numtoread}), start_time={float(info.get('start_time_s', 0.0)):.6f} s"
    )

    nchan, nsamp = data.shape
    print(f"Data loaded. Shape: {data.shape}")
    print(f"Freq Center: {info['f_center']} MHz, BW: {info['bw']} MHz")
    print(f"Time Bin: {info['tbin']*1000:.4f} ms")

    # Title info: filename + samples + duration
    fits_basename = os.path.basename(args.fits_file)
    tbin_s = float(info['tbin'])
    duration_s = nsamp * tbin_s

    # Time offset of this view relative to the full file.
    start_time_s = float(info.get("start_time_s", 0.0))
    
    # 构建频率数组
    # 假设线性分布。注意：如果 BW < 0，说明通道 0 是高频。
    # 通常 PSRFITS: f_channel = f_center - bw/2 + ch * (bw/nchan) ? 
    # 或者 f_center 是中点。
    # 简单起见：
    f_start = info['f_center'] - info['bw'] / 2.0
    f_end = info['f_center'] + info['bw'] / 2.0
    # 注意：fitsio 读取的数据顺序。
    # 如果我们假设 data[0] 对应最低频 (或最高频取决于 BW 符号)
    # 我们生成一个频率数组对应 data 的行。
    # 这里的逻辑可能需要根据实际望远镜数据调整。
    # 假设 BW > 0 时，index 0 是低频；BW < 0 时，index 0 是高频。
    # 但通常 load_fits_data 读出来的是原始顺序。
    
    # 让我们生成一个频率轴
    if info['bw'] > 0:
        freqs = np.linspace(f_start, f_end, nchan)
    else:
        # BW 为负，说明起始频率是高频
        freqs = np.linspace(f_end, f_start, nchan) # f_end is actually lower value numerically if bw is negative? No.
        # 通常 BW 负值表示 f_chan_0 > f_chan_N
        # 让我们用绝对值处理范围，然后根据符号翻转
        f_low = info['f_center'] - abs(info['bw'])/2
        f_high = info['f_center'] + abs(info['bw'])/2
        if info['bw'] < 0:
            freqs = np.linspace(f_high, f_low, nchan)
        else:
            freqs = np.linspace(f_low, f_high, nchan)

    # 交互式绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.35)
    
    # 显示原始数据 (flipud 以便低频在下，符合直觉)
    # 注意：如果 freqs[0] 是高频，flipud 后 row 0 (bottom) 变成高频？
    # matplotlib imshow origin='lower' means index 0 is at bottom.
    # 如果我们不 flipud，index 0 在上。
    # 为了物理直觉：y轴向上频率增加。
    # 如果 freqs 是递增的 (index 0 是低频)，则 index 0 应在下。
    # 如果 freqs 是递减的 (index 0 是高频)，则 index 0 应在上。
    
    # 简单处理：我们总是把数据画成频率向上增加。
    # Strategy: Always prepare `display_img` such that row 0 is high frequency (top), row N is low frequency (bottom).
    if freqs[0] > freqs[-1]:
        # 数据已经是高频在前 (index 0 is high freq)
        display_img = data
        display_freqs = freqs
    else:
        # 数据是低频在前 (index 0 is low freq) -> 翻转
        display_img = np.flipud(data)
        display_freqs = np.flipud(freqs)
        
    extent = [start_time_s, start_time_s + nsamp*info['tbin'], display_freqs[-1], display_freqs[0]]
    origin = 'upper'

    # 归一化用于显示：整幅图像 mean ± 5σ
    mean = float(np.mean(display_img))
    std = float(np.std(display_img))
    vmin, vmax = mean - 5 * std, mean + 5 * std
    
    img_handle = ax.imshow(
        display_img,
        aspect='auto',
        cmap='gist_heat',
        vmin=float(vmin),
        vmax=float(vmax),
        extent=(float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])),
        origin='upper',
    )
    
    # Mask Overlay (RGBA)
    mask_overlay = None
    if not args.nomask:
        rgba0 = np.zeros((nchan, nsamp, 4), dtype=np.float32)
        rgba0[..., 2] = 1.0  # Blue
        rgba0[..., 3] = 0.0  # Alpha
        mask_overlay = ax.imshow(
            rgba0,
            aspect='auto',
            extent=(float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])),
            origin='upper',
        )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(
        f"{fits_basename}\n"
        f"nsamp={nsamp} (nsblk*numtoread), T={duration_s:.6f} s, tbin={tbin_s:.6e} s",
        fontsize=9,
    )

    # Legend: pulse mask color (blue)
    if not args.nomask:
        pulse_patch = mpatches.Patch(color=(0.0, 0.0, 1.0, 0.5), label='Pulse')
        ax.legend(handles=[pulse_patch], loc='upper right', frameon=True, fontsize=9)

    # Sliders (only meaningful when interactive)
    slider_height = 0.03
    slider_spacing = 0.04
    s_bottom = 0.05
    
    ax_t0 = plt.axes((0.15, s_bottom, 0.65, slider_height))
    ax_p0 = plt.axes((0.15, s_bottom + slider_spacing, 0.65, slider_height))
    ax_dm = plt.axes((0.15, s_bottom + 2 * slider_spacing, 0.65, slider_height))
    ax_width = plt.axes((0.15, s_bottom + 3 * slider_spacing, 0.65, slider_height))
    
    s_t0 = Slider(ax_t0, 'T0 (s)', 0, args.period, valinit=((args.t0 - start_time_s) % args.period))
    s_p0 = Slider(ax_p0, 'Period (s)', args.period * 0.8, args.period * 1.2, valinit=args.period)
    s_dm = Slider(ax_dm, 'DM', args.dm * 0.8, args.dm * 1.2, valinit=args.dm)
    s_width = Slider(ax_width, 'Width (s)', 0.0001, 0.2, valinit=args.width)

    s_iwidth = None
    s_it0 = None
    if args.interpulse:
        ax_iwidth = plt.axes((0.15, s_bottom + 4 * slider_spacing, 0.65, slider_height))
        s_iwidth = Slider(ax_iwidth, 'Inter W', 0.0001, 0.2, valinit=args.interpulseWidth)
        
        ax_it0 = plt.axes((0.15, s_bottom + 5 * slider_spacing, 0.65, slider_height))
        # Default it0 local
        if args.interpulset0 is not None:
            it0_init = (args.interpulset0 - start_time_s) % args.period
        else:
            it0_init = (s_t0.val + 0.5 * args.period) % args.period
        s_it0 = Slider(ax_it0, 'Inter T0', 0, args.period, valinit=it0_init)

    def update(val):
        if args.nomask:
            return

        dm = s_dm.val
        period = s_p0.val
        t0_local = s_t0.val
        width = s_width.val
        iwidth = s_iwidth.val if s_iwidth else 0.0
        it0_local = s_it0.val if s_it0 else None
        
        mask = generate_mask(
            data.shape, freqs, info['tbin'], dm, period, t0_local, width,
            enable_interpulse=args.interpulse,
            interpulse_width_s=iwidth,
            interpulse_t0=it0_local,
            lo_freq_mhz=args.pulselofreq,
            hi_freq_mhz=args.pulsehifreq
        )
        
        if freqs[0] < freqs[-1]:
            display_mask = np.flipud(mask)
        else:
            display_mask = mask
            
        rgba = np.zeros((nchan, nsamp, 4))
        rgba[..., 2] = 1.0  # Blue
        rgba[..., 3] = display_mask * 0.5  # Alpha

        if mask_overlay is not None:
            mask_overlay.set_data(rgba)
        fig.canvas.draw_idle()

    s_dm.on_changed(update)
    s_p0.on_changed(update)
    s_t0.on_changed(update)
    s_width.on_changed(update)
    if s_iwidth:
        s_iwidth.on_changed(update)
    if s_it0:
        s_it0.on_changed(update)
    
    # --- Keyboard Navigation ---
    current_block_idx = args.block

    def change_block(step):
        nonlocal current_block_idx, data, info
        
        # Calculate phase continuity to keep pulse position consistent
        current_period = s_p0.val
        current_t0_local = s_t0.val
        old_start_time = float(info.get('start_time_s', 0.0))
        abs_t0_ref = old_start_time + current_t0_local
        
        abs_it0_ref = None
        if s_it0:
            abs_it0_ref = old_start_time + s_it0.val

        new_idx = current_block_idx + step
        if new_idx < 0:
            print("Already at the start.")
            return

        print(f"Loading block {new_idx}...")
        try:
            new_data, new_info = load_fits_data(args.fits_file, new_idx, args.numtoread)
        except Exception as e:
            print(f"Could not load block {new_idx} (maybe end of file?): {e}")
            return
            
        current_block_idx = new_idx
        data = new_data
        info = new_info
        
        # Update display image
        if freqs[0] > freqs[-1]:
            display_img = data
        else:
            display_img = np.flipud(data)
            
        img_handle.set_data(display_img)
        
        # Update contrast
        mean_v = float(np.mean(display_img))
        std_v = float(np.std(display_img))
        img_handle.set_clim(mean_v - 5*std_v, mean_v + 5*std_v)

        # Update Extent (time axis might change if block size varies)
        nsamp_new = new_data.shape[1]
        new_start_time = float(info.get('start_time_s', 0.0))
        new_extent = (new_start_time, new_start_time + nsamp_new*info['tbin'], float(display_freqs[-1]), float(display_freqs[0]))
        img_handle.set_extent(new_extent)
        if mask_overlay is not None:
            mask_overlay.set_extent(new_extent)

        # Update Title
        duration_s = nsamp_new * info['tbin']
        ax.set_title(
            f"{fits_basename} (Block {current_block_idx})\n"
            f"nsamp={nsamp_new}, T={duration_s:.6f} s, tbin={info['tbin']:.6e} s",
            fontsize=9,
        )
        
        # Calculate new local T0 and update slider (triggers mask update)
        new_start_time = float(info.get('start_time_s', 0.0))
        new_t0_local = (abs_t0_ref - new_start_time) % current_period
        s_t0.set_val(new_t0_local)
        
        if s_it0 and abs_it0_ref is not None:
             new_it0_local = (abs_it0_ref - new_start_time) % current_period
             s_it0.set_val(new_it0_local)

    def on_key_press(event):
        if event.key == 'right':
            change_block(args.numtoread)
        elif event.key == 'left':
            change_block(-args.numtoread)
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    print("Tip: Use Left/Right arrow keys to navigate between blocks.")

    # Initial update
    update(None)

    # Headless/non-interactive fallback: save and exit cleanly.
    if args.no_gui or (not INTERACTIVE_OK) or (MPL_BACKEND.lower() == 'agg'):
        out_path = args.out
        if not out_path:
            base = os.path.splitext(os.path.basename(args.fits_file))[0]
            out_path = (
                f"{base}_pulse_mask_block{args.block}_numtoread{args.numtoread}"
                f"_startrow{int(info.get('start_row', 0))}.png"
            )
        try:
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"[Info] Headless mode: saved figure to {out_path}")
        finally:
            plt.close(fig)
        return

    plt.show()

if __name__ == "__main__":
    main()
