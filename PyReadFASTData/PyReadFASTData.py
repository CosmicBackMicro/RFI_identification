#!/usr/bin/env python3
import os

# 禁用 Numba 的内部并行，确保每个 Worker 线程独占一个 CPU 核心
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import fitsio
import cv2
from numba import njit, prange

@njit(fastmath=True, cache=True)
def _gaussian_iter_1d(data, n_sigma, mask, use_median, max_iter, med_thr, std_thr):
    total_outliers = 0
    last_center = 0.0
    last_std = 0.0
    EPS_STD = 1e-12
    
    valid_indices = np.where(~mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid < 3:
        return 0

    for i in range(max_iter):
        if n_valid < 3:
            break
        vals = data[valid_indices]
        if use_median:
            curr_center = np.median(vals)
        else:
            curr_center = np.mean(vals)
        diff = vals - curr_center
        curr_std = np.sqrt(np.mean(diff**2))

        if curr_std <= EPS_STD:
            break
        if i > 0:
            change_center = abs(curr_center - last_center)
            denom = last_std if last_std > EPS_STD else EPS_STD
            change_std_rate = abs(curr_std - last_std) / denom
            if change_center < med_thr and change_std_rate < std_thr:
                break
        upper = curr_center + n_sigma * curr_std
        lower = curr_center - n_sigma * curr_std
        new_valid_list = []
        new_found = 0
        for idx in valid_indices:
            val = data[idx]
            if val > upper or val < lower:
                mask[idx] = True
                new_found += 1
            else:
                new_valid_list.append(idx)
        if new_found == 0:
            break
        total_outliers += new_found
        valid_indices = np.array(new_valid_list)
        n_valid = len(valid_indices)
        last_center = curr_center
        last_std = curr_std
    return total_outliers

@njit(fastmath=True)
def _gaussian_iter_2d_serial(data, n_sigma, mask, use_median, max_iter, med_thr, std_thr):
    nchan = data.shape[0]
    outliers_per_chan = np.zeros(nchan, dtype=np.int32)
    for i in range(nchan):
        outliers_per_chan[i] = _gaussian_iter_1d(
            data[i], n_sigma, mask[i], use_median, max_iter, med_thr, std_thr
        )
    return np.sum(outliers_per_chan)

def gaussian_iteration(data, n_sigma, mask=None, use_median=True, max_iterations=15):
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    
    med_thr = 1e-6
    std_thr = 0.0001
    
    if data.ndim == 1:
        count = _gaussian_iter_1d(
            data, n_sigma, mask, use_median, max_iterations, med_thr, std_thr
        )
    elif data.ndim == 2:
        count = _gaussian_iter_2d_serial(
            data, n_sigma, mask, use_median, max_iterations, med_thr, std_thr
        )
    else:
        raise ValueError("Data must be 1D or 2D")
        
    return count, mask

def inChan(data, n_sigma, mask=None):
    return gaussian_iteration(data, n_sigma, mask=mask, use_median=True)

def outChan(channel_stats, n_sigma, mask=None):
    return gaussian_iteration(channel_stats, n_sigma, mask=mask, use_median=True)

@njit(fastmath=True)
def _substitute_1d(data, mask):
    valid_indices = np.where(~mask)[0]
    if len(valid_indices) == 0:
        return 0
        
    masked_indices = np.where(mask)[0]
    n_substituted = len(masked_indices)
    
    if n_substituted == 0:
        return 0
        
    for idx in masked_indices:
        rand_idx = np.random.randint(0, len(valid_indices))
        data[idx] = data[valid_indices[rand_idx]]
            
    return n_substituted

@njit(fastmath=True)
def _substitute_2d_serial(data, mask):
    nchan = data.shape[0]
    counts = np.zeros(nchan, dtype=np.int32)
    for i in range(nchan):
        counts[i] = _substitute_1d(data[i], mask[i])
    return np.sum(counts)

@njit(fastmath=True)
def _calc_chan_stds(data, mask):
    nchan = data.shape[0]
    stds = np.zeros(nchan, dtype=np.float32)
    for i in range(nchan):
        vals = data[i][~mask[i]]
        if len(vals) > 1:
            stds[i] = np.std(vals)
        else:
            stds[i] = np.std(data[i])
    return stds

@njit(fastmath=True)
def _calc_chan_stats(data, mask):
    nchan = data.shape[0]
    means = np.zeros(nchan, dtype=np.float32)
    stds = np.zeros(nchan, dtype=np.float32)
    for i in range(nchan):
        vals = data[i][~mask[i]]
        if len(vals) > 1:
            m = np.mean(vals)
            means[i] = m
            stds[i] = np.std(vals)
        else:
            means[i] = np.mean(data[i])
            stds[i] = np.std(data[i])
    return means, stds

@njit(fastmath=True)
def _columns_similar(data, ta, tb, abs_epsilon, rel_sigma):
    """
    判断两个时间采样列是否相似 (重复列检测)
    data: (nchan, nsamp)
    """
    nchan = data.shape[0]
    sum_abs_diff = 0.0
    sum_abs_base = 0.0
    eq_cnt = 0
    
    for c in range(nchan):
        va = data[c, ta]
        vb = data[c, tb]
        ad = abs(va - vb)
        sum_abs_diff += ad
        sum_abs_base += 0.5 * (abs(va) + abs(vb))
        if ad <= abs_epsilon:
            eq_cnt += 1
            
    mean_diff = sum_abs_diff / nchan
    mean_base = sum_abs_base / nchan
    
    thr = min(abs_epsilon, rel_sigma * mean_base)
    eq_frac_min = 0.98
    eq_frac = eq_cnt / nchan
    
    return (mean_diff <= thr) and (eq_frac >= eq_frac_min)

@njit(fastmath=True)
def detect_vertical_repeated_columns(data, min_run=3, abs_epsilon=1e-5, rel_sigma=0.01):
    """
    识别垂直方向上的重复列 (数据异常/硬拷贝)
    """
    nchan, nsamp = data.shape
    v_mask = np.zeros((nchan, nsamp), dtype=np.bool_)
    
    t = 1
    while t < nsamp:
        if _columns_similar(data, t, t - 1, abs_epsilon, rel_sigma):
            anchor = t
            start = t - 1
            end = t
            
            # 向右扩展
            u = end + 1
            while u < nsamp and _columns_similar(data, u, anchor, abs_epsilon, rel_sigma):
                end = u
                u += 1
                
            # 向左扩展
            v = start - 1
            while v >= 0 and _columns_similar(data, v, anchor, abs_epsilon, rel_sigma):
                start = v
                v -= 1
                
            if (end - start + 1) >= min_run:
                for k in range(start, end + 1):
                    v_mask[:, k] = True
            t = end + 1
        else:
            t += 1
            
    return v_mask

@njit(fastmath=True)
def detect_vertical_rfi(data, point_mask, horizontal_mask, nsigma_mean=4.0):
    nchan, nsamp = data.shape
    vertical_mask = np.zeros(nsamp, dtype=np.bool_)
    time_meds = np.zeros(nsamp, dtype=np.float32)
    # 计算列内受干扰像素的比例，如果比例过高，说明这一列本身可能就是 RFI 密集的
    point_ratios = np.zeros(nsamp, dtype=np.float32)
    
    for t in range(nsamp):
        valid_list = []
        p_cnt = 0
        h_cnt = 0
        for c in range(nchan):
            if horizontal_mask[c]:
                h_cnt += 1
                continue
            if point_mask[c, t]:
                p_cnt += 1
            valid_list.append(data[c, t])
        
        # 记录该列中 Point RFI 在非坏道中的占比
        if nchan > h_cnt:
            point_ratios[t] = p_cnt / (nchan - h_cnt)

        if len(valid_list) > 0:
            time_meds[t] = np.median(np.array(valid_list))
        else:
            time_meds[t] = 0.0

    valid_meds = time_meds[time_meds > 0]
    if len(valid_meds) > 1:
        overall_med = np.median(valid_meds)
        mad = np.median(np.abs(valid_meds - overall_med)) * 1.4826
        thr = overall_med + nsigma_mean * mad
        
        for t in range(nsamp):
            # 只有当：1. 中值超过阈值 且 2. 这一列本身原本就包含一定比例的异常点时
            # 才认为这是一个宽带垂直干扰，而不是由于统计涨落误触发
            # 如果整列非常干净（Point RFI 极少），则不太可能是宽带 RFI
            if time_meds[t] > thr and point_ratios[t] > 0.05:
                vertical_mask[t] = True
                
    v_mask_2d = np.zeros((nchan, nsamp), dtype=np.bool_)
    for t in range(nsamp):
        if vertical_mask[t]:
            v_mask_2d[:, t] = True
            
    return v_mask_2d

def detect_block_rfi(point_mask_2d, min_area=5000, min_density=0.5, dilate_radius=7, dilate_iterations=1):
    mask_u8 = point_mask_2d.astype(np.uint8)
    
    if dilate_radius > 0 and dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_radius, dilate_radius))
        dilated = cv2.dilate(mask_u8, kernel, iterations=dilate_iterations)
    else:
        dilated = mask_u8

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    block_mask = np.zeros_like(point_mask_2d, dtype=bool)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        bb_area = w * h
        if bb_area > 0:
            density = area / bb_area
            wh_ratio = w / h
            if area >= min_area and density >= min_density and 0.2 <= wh_ratio <= 5.0:
                block_mask[y:y+h, x:x+w] = True
                
    return block_mask

def dispersion_delay(f_mhz, f_ref_mhz, dm):
    return 4.148808 * 1e6 * dm * (1.0/(f_mhz**2) - 1.0/(f_ref_mhz**2))

def detect_pulsar_mask(nchan, nsamp, freqs_mhz, tbin_s, dm, period, t0_local, width_s):
    pulse_mask = np.zeros((nchan, nsamp), dtype=bool)
    if dm <= 0 or period <= 0:
         return pulse_mask
         
    f_ref = freqs_mhz[0]
    total_duration = nsamp * tbin_s
    max_delay = dispersion_delay(freqs_mhz[-1], f_ref, dm) / 1000.0
    
    k_min = int(np.floor((-max_delay - t0_local) / period)) - 1
    k_max = int(np.ceil((total_duration - t0_local) / period)) + 1
    
    width_samp = max(1, int(width_s / tbin_s))
    
    for c in range(nchan):
        delay_s = dispersion_delay(freqs_mhz[c], f_ref, dm) / 1000.0
        for k in range(k_min, k_max + 1):
            pulse_arrival_t0 = t0_local + k * period
            t_start = (pulse_arrival_t0 + delay_s) / tbin_s
            
            s_start = int(round(t_start - width_samp / 2.0))
            s_end = int(round(t_start + width_samp / 2.0))
            
            if s_start < nsamp and s_end >= 0:
                s_s = max(0, s_start)
                s_e = min(nsamp - 1, s_end)
                pulse_mask[c, s_s:s_e+1] = True
                
    return pulse_mask

def identification_workflow(data, header, n_sigma_in=6.0, n_sigma_out=3.0, do_substitute=True, pulsar_params=None):
    if data.ndim != 2:
        data = np.squeeze(data)
        if data.ndim != 2:
             raise ValueError(f"Expected 2D array (nchan, nsamp), got shape {data.shape}")

    nchan, nsamp = data.shape
    f_cen = float(header.get('OBSFREQ', 1250.0))
    bw = float(header.get('OBSBW', -0.0))
    tbin_s = float(header.get('TBIN', 4.9152e-05))
    
    df = bw / nchan
    freqs_mhz = np.array([f_cen - bw/2.0 + (i + 0.5)*df for i in range(nchan)])
    if bw < 0:
        freqs_mhz = freqs_mhz[::-1]

    point_mask = np.zeros_like(data, dtype=bool)
    chan_flagged = np.zeros(nchan, dtype=bool)
    pulse_mask = np.zeros((nchan, nsamp), dtype=bool)
    
    if pulsar_params is not None and pulsar_params.get('has_pulse', False):
        dm = pulsar_params.get('dm', 0.0)
        period = pulsar_params.get('period', 1.0)
        t0_list = pulsar_params.get('t0', [0.0])
        width_list = pulsar_params.get('width', [0.01])
        
        for t0_val, width_val in zip(t0_list, width_list):
            pm = detect_pulsar_mask(nchan, nsamp, freqs_mhz, tbin_s, dm, period, t0_val, width_val)
            pulse_mask |= pm
            
    work_data = data.copy()
    if np.any(pulse_mask):
         global_med = np.median(work_data)
         work_data[pulse_mask] = global_med
    
    inChan(work_data, n_sigma_in, mask=point_mask)
    
    if do_substitute:
        _substitute_2d_serial(work_data, point_mask)
        
    chan_stds = _calc_chan_stds(work_data, point_mask)
    
    _gaussian_iter_1d(chan_stds, n_sigma_out, chan_flagged, use_median=True, 
                      max_iter=30, med_thr=1e-6, std_thr=0.01)
    
    for i in range(nchan):
        if chan_flagged[i]:
            point_ratio = np.mean(point_mask[i])
            if point_ratio > 0.30:
                chan_flagged[i] = False
                
    # 5. 时间/宽带垂直识别 -> 标记为 vertical (2)
    # 结合统计量波动 (Stat) 和列重复 (Repeated Columns)
    v_mask_stat = detect_vertical_rfi(work_data, point_mask, chan_flagged, nsigma_mean=4.0)
    v_mask_repeat = detect_vertical_repeated_columns(work_data, min_run=2)
    v_mask_2d = v_mask_stat | v_mask_repeat

    block_rfi_mask = detect_block_rfi(point_mask, min_area=5000)
    processed_point_mask = morphological_post_processing(point_mask, target_scale=3)

    multi_mask = np.zeros((nchan, nsamp), dtype=np.uint8)
    
    multi_mask[v_mask_2d] = 2
    for i in range(nchan):
        if chan_flagged[i]:
            multi_mask[i, :] = 1
            
    multi_mask[processed_point_mask] = 3
    multi_mask[block_rfi_mask] = 4
    multi_mask[pulse_mask] = 5
    
    return multi_mask

def morphological_post_processing(mask_bool, target_scale=3):
    if not np.any(mask_bool):
        return mask_bool
        
    mask_u8 = mask_bool.astype(np.uint8)
    kernel_size = (target_scale, target_scale)
    kernel = np.ones(kernel_size, np.uint8)
    
    neighbor_count = cv2.filter2D(mask_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    mask_u8[neighbor_count <= 2] = 0
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    final_mask = cv2.medianBlur(dilated, target_scale)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    return final_mask > 0

# (删除后面冗余的函数定义)

def simplified_rfi_workflow(data, n_sigma_pixel=6.0, n_sigma_chan=3.0):
    """
    用户要求的识别流程：
    1. 每个通道内像素值高斯迭代 -> Point RFI 掩码
    2. 计算通道均值和标准差 -> 两条 Nchan 序列
    3. 对这两条序列分别执行高斯迭代 -> Horizontal RFI 掩码
    """
    nchan, nsamp = data.shape
    
    # --- 阶段 1: 像素级检测 (Point RFI) ---
    print("Step 1: Detecting outlier pixels (Point RFI)...")
    point_mask = np.zeros((nchan, nsamp), dtype=bool)
    inChan(data, n_sigma_pixel, mask=point_mask)
    
    # --- 阶段 2: 通道统计量准备 ---
    print("Step 2: Calculating channel mean and std sequences...")
    chan_means, chan_stds = _calc_chan_stats(data, point_mask)
    
    # --- 阶段 3: 通道级识别 (Horizontal RFI) ---
    print("Step 3: Detecting outlier channels (Horizontal RFI)...")
    # 建立两个通道标记数组
    chan_flagged_by_mean = np.zeros(nchan, dtype=bool)
    chan_flagged_by_std = np.zeros(nchan, dtype=bool)
    
    # 对均值序列执行迭代 (3.0 sigma)
    _gaussian_iter_1d(chan_means, n_sigma_chan, chan_flagged_by_mean, use_median=True, 
                      max_iter=30, med_thr=1e-6, std_thr=0.01)
    
    # 对标准差序列执行迭代 (3.0 sigma)
    _gaussian_iter_1d(chan_stds, n_sigma_chan, chan_flagged_by_std, use_median=True, 
                      max_iter=30, med_thr=1e-6, std_thr=0.01)
    
    # 合并两种检测方案得到的异常通道
    chan_flagged_final = chan_flagged_by_mean | chan_flagged_by_std
    
    # --- 为 Horizontal 增加 1D 膨胀操作 ---
    # 这可以覆盖因某些通道 RFI 极强导致相邻通道受到泄露（leakage）影响但主检测未触发的情况
    if np.any(chan_flagged_final):
        # 使用 1D 卷积/膨胀：只要邻居被标记，自己也被标记 (1 邻域半径，总宽度 3)
        kernel_1d = np.ones(3, dtype=np.uint8)
        chan_flagged_u8 = chan_flagged_final.astype(np.uint8)
        chan_flagged_dilated = cv2.dilate(chan_flagged_u8, kernel_1d, iterations=1)
        chan_flagged_final = chan_flagged_dilated > 0
    
    # 将通道标记展开为 2D 掩码
    horizontal_mask = np.zeros((nchan, nsamp), dtype=bool)
    for i in range(nchan):
        if chan_flagged_final[i]:
            horizontal_mask[i, :] = True
            
    # --- 阶段 4: 返回结果 ---
    return point_mask, horizontal_mask

def read_psrfits_subint(fits_path, start_row=0, num_rows=1):
    with fitsio.FITS(fits_path, 'r') as fits:
        subint_table = fits['SUBINT']
        header = subint_table.read_header()
        
        nchan = int(header['NCHAN'])
        npol = int(header['NPOL'])
        nsblk = int(header['NSBLK'])
        
        data_rows = subint_table.read(columns=['DATA', 'DAT_SCL', 'DAT_OFFS'], 
                                     rows=range(start_row, start_row + num_rows))
        
    all_pieces = []
    if isinstance(data_rows, np.ndarray) and data_rows.ndim == 0:
        data_rows = [data_rows]

    for i in range(len(data_rows)):
        raw = np.asarray(data_rows[i]['DATA']).astype(np.float32)

        if raw.size == nsblk * npol * nchan:
            raw = raw.reshape((nsblk, npol, nchan))
            raw = raw[:, 0, :]
        else:
            print(f"[Warn] Shape mismatch: raw.size={raw.size} vs expected={nsblk*npol*nchan}")
            if raw.ndim > 2:
                raw = raw.reshape(-1, nchan)
            
        scl = np.asarray(data_rows[i]['DAT_SCL']).astype(np.float32)
        offs = np.asarray(data_rows[i]['DAT_OFFS']).astype(np.float32)
        
        cooked = raw * scl[np.newaxis, :] + offs[np.newaxis, :]
        all_pieces.append(cooked)
        
    full_data = np.vstack(all_pieces).T
    
    return full_data, header

def export_masks_to_png(multi_mask, data, output_dir, base_name, subint_idx, downsamp=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    suffix = f"_downsamp{downsamp}"
    
    # 保存 Mask PNG
    img = np.flipud(multi_mask)
    p_path = os.path.join(output_dir, f"{base_name}{suffix}_sub{subint_idx:04d}.png")
    cv2.imwrite(p_path, img)
    
    # 保存原始数据的 FITS Image (仿照 ReadFASTData C 版本)
    fits_path = os.path.join(output_dir, f"{base_name}{suffix}_sub{subint_idx:04d}.fits")
    if os.path.exists(fits_path):
        os.remove(fits_path)
    with fitsio.FITS(fits_path, 'rw') as fits:
        # 翻转数据以匹配可视化习惯 (高频在上)
        fits.write(np.flipud(data))

def stream_psrfits_subints(fits_path, start_row=0, end_row=None, chunk_size=1):
    with fitsio.FITS(fits_path, 'r') as fits:
        subint_table = fits['SUBINT']
        total_rows = subint_table.get_nrows()
        
        if end_row is None or end_row > total_rows:
            end_row = total_rows
            
        for i in range(start_row, end_row, chunk_size):
            num_to_read = min(chunk_size, end_row - i)
            data, header_info = read_psrfits_subint(fits_path, start_row=i, num_rows=num_to_read)
            yield data, header_info, i

def process_chunk_task(chunk_start, chunk_end, fits_path, out_dir, base_name, sig_p, sig_c, pulsar_params, downsamp):
    try:
        processed_count = 0
        from __main__ import identification_workflow, stream_psrfits_subints, export_masks_to_png
        for data, header_info, idx in stream_psrfits_subints(fits_path, start_row=chunk_start, end_row=chunk_end):
            # 如果指定了下采样，则在检测前先对数据执行物理下采样
            working_data = data
            
            # FITSHDR 对象不支持 .copy()，我们可以用普通字典来传递我们需要修改的参数
            # 这样既能避免修改原始 header，也能解决 AttributeError
            working_header = {
                'OBSFREQ': header_info.get('OBSFREQ', 1250.0),
                'OBSBW': header_info.get('OBSBW', 0.0),
                'TBIN': header_info.get('TBIN', 4.9152e-05)
            }
            
            if downsamp > 1:
                nchan, nsamp = data.shape
                new_nsamp = nsamp // downsamp
                if new_nsamp > 0:
                    # 时间轴均值下采样
                    working_data = data[:, :new_nsamp*downsamp].reshape(nchan, new_nsamp, downsamp).mean(axis=2)
                    # 更新字典中的 TBIN (时间分辨率变大)
                    working_header['TBIN'] = float(working_header['TBIN']) * downsamp
                
            multi_mask = identification_workflow(
                working_data, 
                header=working_header,
                n_sigma_in=sig_p, 
                n_sigma_out=sig_c,
                pulsar_params=pulsar_params
            )
            
            # 直接导出对应分辨率的图像和掩码
            export_masks_to_png(multi_mask, working_data, out_dir, base_name, idx, downsamp)
            
            processed_count += 1
        return chunk_start, processed_count, None
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return chunk_start, 0, err

if __name__ == "__main__":
    import argparse
    import time
    import os

    import numpy as np
    import fitsio
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="PyReadFASTData: High-performance PSRFITS RFI Detection Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Path to the input PSRFITS file")
    parser.add_argument("-o", "--output", default="./output_masks", help="Directory to save output masks (default: ./output_masks)")
    parser.add_argument("--sigma-inchan", type=float, default=6.0, help="N-sigma threshold for pixel identification (default: 6.0)")
    parser.add_argument("--sigma-outchan", type=float, default=3.0, help="N-sigma threshold for channel identification (default: 3.0)")
    parser.add_argument("--start", type=int, default=0, help="Start subint index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End subint index (default: end of file)")
    parser.add_argument("--ncpus", type=int, default=18, help="Number of worker threads (default: 18 for 20-thread CPU)")
    parser.add_argument("--downsamp", type=int, default=1, help="Downsampling factor identifier for output (default: 1)")
    
    # Pulsar arguments
    parser.add_argument("--dm", type=float, default=0.0, help="Dispersion Measure")
    parser.add_argument("--period", type=float, default=0.0, help="Pulsar period (s)")
    parser.add_argument("--t0", type=float, nargs='+', default=[0.0], help="Time offset of pulse (s). Allow up to 2 values for main and interpulse")
    parser.add_argument("--width", type=float, nargs='+', default=[0.01], help="Pulse width (s). Allow up to 2 values for main and interpulse")
    
    args = parser.parse_args()

    pulsar_params = None
    if args.dm > 0 or args.period > 0:
        if args.dm <= 0 or args.period <= 0:
            print("Warning: Both --dm and --period must be provided and greater than 0 to enable pulsar masking. Pulsar masking will be DISABLED.")
        else:
            if len(args.t0) != len(args.width):
                print("Error: --t0 and --width must have the same number of values (e.g., both 1 for main pulse, or both 2 for main and interpulse).")
                exit(1)
                
            pulsar_params = {
                'has_pulse': True,
                'dm': args.dm,
                'period': args.period,
                't0': args.t0,
                'width': args.width
            }

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # 1. 获取任务总量用于进度条和分片
    with fitsio.FITS(args.input, 'r') as fits:
        total_rows_in_file = fits['SUBINT'].get_nrows()
        end_idx = args.end if args.end is not None else total_rows_in_file
        actual_total = end_idx - args.start

    print("="*60)
    print(f"🚀 Starting Multi-process Detection on: {args.input}")
    print(f"📂 Output directory: {args.output}")
    print(f"🧵 Parallelism: {args.ncpus} processes (ProcessPool)")
    print("="*60)

    # 简化的块级任务分配：
    # 每个进程将获得大约 10 个 subint 的 Chunk，既保证负载均衡，又能降低进程间通信开销
    chunk_size = 10
    chunks = []
    for c_start in range(args.start, end_idx, chunk_size):
        c_end = min(c_start + chunk_size, end_idx)
        chunks.append((c_start, c_end))

    start_time = time.time()

    try:
        with tqdm(total=actual_total, desc="Processing", unit="subint") as pbar:
            with ProcessPoolExecutor(max_workers=args.ncpus) as executor:
                futures = []
                for c_start, c_end in chunks:
                    f = executor.submit(
                        process_chunk_task,
                        c_start, c_end,
                        args.input, args.output, base_name,
                        args.sigma_inchan, args.sigma_outchan,
                        pulsar_params, args.downsamp
                    )
                    futures.append(f)
                
                for f in as_completed(futures):
                    chunk_start, processed_count, err = f.result()
                    if err:
                        print(f"\n[Fatal Error in chunk {chunk_start}] {err}")
                    else:
                        pbar.update(processed_count)

        end_time = time.time()
        wall_time = end_time - start_time
        print("="*60)
        print(f"✅ Multi-process Pipeline complete!")
        print(f"⏱️  Total Wall-clock Time: {wall_time:.2f} seconds")
        if wall_time > 0:
            print(f"🚀 Average Throughput: {actual_total / wall_time:.2f} subints/s")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()

