#!/usr/bin/env python3
"""
AI RFI Mitigation for PSRFITS Data (Production-grade Inference Pipeline)

This script provides a high-performance, end-to-end solution for radio frequency interference (RFI) 
mitigation in radio astronomy data. 

Key Features:
- 3-Stage Pipeline: Asynchronous Reader (CPU) -> Predictor (GPU/TensorRT) -> Writer (CPU) 
  to maximize hardware utilization and throughput.
- TensorRT Acceleration: Leverages pre-built TensorRT engines for ultra-fast deep learning inference.
- Scientific Data Integration: Directly processes and repairs PSRFITS files, handling complex 
  data structures (subints, poly-phase data, etc.).
- Robust RFI Cleaning: Includes pixel-level mask generation and statistical-based sampling 
  to replace RFI-contaminated pixels with clean background estimates.

Usage:
    python src/AI_RFI.py --fits <path_to_fits> --engine <path_to_trt_engine> --batch 4
"""
import fitsio
import numpy as np
import os
import time
import cv2
import argparse
import sys
import threading
import gc
import shutil
from queue import Queue, Empty
from threading import Thread

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    if cuda is not None:
        cuda.init()  # type: ignore[attr-defined]
except ImportError:
    trt = None
    cuda = None


class TRTInference:
    def __init__(self, engine_path):
        if trt is None or cuda is None:
            raise RuntimeError("tensorrt or pycuda not installed.")
        
        self.logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:  # type: ignore[attr-defined]
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()  # type: ignore[attr-defined]
        
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = []
        self.output_names = []
        for name in self.tensor_names:
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:  # type: ignore[attr-defined]
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:  # type: ignore[attr-defined]
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError("No input tensors found.")
        self.primary_input = self.input_names[0]

        # Pre-allocate buffers
        profile_shape = self.engine.get_tensor_profile_shape(self.primary_input, 0)
        self.target_shape = tuple(profile_shape[1])
        max_input_shape = tuple(profile_shape[2])
        input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.primary_input))
        
        self.h_input = cuda.pagelocked_empty(max_input_shape, input_dtype)  # type: ignore[attr-defined]
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)  # type: ignore[attr-defined]
        
        self.context.set_input_shape(self.primary_input, max_input_shape)
        
        self.primary_output = self.output_names[0]
        self.output_host_buffers = {}
        self.output_device_buffers = {}
        for name in self.output_names:
            max_output_shape = tuple(self.context.get_tensor_shape(name))
            output_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_buf = cuda.pagelocked_empty(max_output_shape, output_dtype)  # type: ignore[attr-defined]
            device_buf = cuda.mem_alloc(host_buf.nbytes)  # type: ignore[attr-defined]
            self.output_host_buffers[name] = host_buf
            self.output_device_buffers[name] = device_buf
        
        self.context.set_input_shape(self.primary_input, self.target_shape)
        self.max_batch_size = max_input_shape[0]
        self.height, self.width = self.target_shape[2], self.target_shape[3]
        
        print(f"🚀 Engine loaded. Target: {self.width}x{self.height}, Max Batch: {self.max_batch_size}")

    @staticmethod
    def normalize_image_mean_std(image: np.ndarray, k: float = 5.0) -> np.ndarray:
        """Normalize image to [0,1] using mean and k*std."""
        img = image.astype(np.float32)
        mean = img.mean()
        std = img.std()
        if std <= 1e-6:
            return np.full(img.shape, 0.5, dtype=np.float32)
        
        lo, hi = mean - k * std, mean + k * std
        np.clip(img, lo, hi, out=img)
        img -= lo
        img /= (hi - lo)
        return img

    def predict_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """Inference for pre-processed (N, 1, H, W) data."""
        batch_size = batch_data.shape[0]
        self.h_input[:batch_size] = batch_data
        
        self.context.set_input_shape(self.primary_input, (batch_size, 1, self.target_shape[2], self.target_shape[3]))
        self.context.set_tensor_address(self.primary_input, int(self.d_input))
        for name, device_buf in self.output_device_buffers.items():
            self.context.set_tensor_address(name, int(device_buf))
        
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)  # type: ignore[attr-defined]
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        primary_host = self.output_host_buffers[self.primary_output]
        cuda.memcpy_dtoh_async(primary_host[:batch_size], self.output_device_buffers[self.primary_output], self.stream)  # type: ignore[attr-defined]
        self.stream.synchronize()
        
        # Return copy to allow safe memory cleanup in GPU thread
        return np.array(primary_host[:batch_size])


def replace_masked_pixels(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace RFI pixels using random sampling from clean regions."""
    if data.ndim > 2: data = np.squeeze(data)
    if mask.ndim > 2: mask = np.squeeze(mask)
    C, T = data.shape
    cleaned = data.copy()
    
    # Class definitions: 0:bkg, 1:horizontal(chan), 2:vertical, 3:point, 4:block, 5:pulsar
    # Priority: Horizontal RFI > Vertical/Point/Block RFI > Pulsar. 
    # Overlapping regions should be treated as RFI to avoid false positives in pulse search.
    PULSAR_LABEL, CHAN_LABEL = 5, 1
    
    clean_mask = (mask == 0)
    
    clean_counts = np.sum(clean_mask, axis=1)
    rows_to_fix = np.where((clean_counts > 0) & (clean_counts > 0.05 * T))[0]
    
    for c in rows_to_fix:
        row_m = mask[c]
        # 如果该通道包含了横向(chan)干扰，则该行内所有非背景像素（包括识别为 pulsar 的）均进行替换。
        # 这是为了从根本上消除可能残留在脉冲特征中的 RFI。
        if np.any(row_m == CHAN_LABEL):
            m_idx = (row_m > 0)
        else:
            # 否则（如只有点状或块状干扰），默认依然保护 pulsar 区域以保留信号。
            # 但如果这些干扰与脉冲重叠，在训练阶段通过调换优先级已经将其打上了 RFI 标签，
            # 因此这里 row_m 会直接被识别为 RFI 类别而不受保护。
            m_idx = (row_m > 0) & (row_m != PULSAR_LABEL)

        if not np.any(m_idx): continue
        bg_data = data[c, clean_mask[c]]
        cleaned[c, m_idx] = bg_data[np.random.randint(0, bg_data.size, np.count_nonzero(m_idx))]

    # 第二遍（时间维度/列）：补漏，逻辑保持一致（排除 pulsar 以免二次误伤）
    to_fix_mask = (mask > 0) & (mask != PULSAR_LABEL)
    still_to_fix = to_fix_mask & (np.abs(cleaned - data) < 1e-7)
    clean_counts_col = np.sum(clean_mask, axis=0)
    cols_to_fix = np.where(clean_counts_col > 0)[0]
    
    for t in cols_to_fix:
        col_m_mask = still_to_fix[:, t]
        if not np.any(col_m_mask): continue
        bg_data = data[clean_mask[:, t], t]
        cleaned[col_m_mask, t] = bg_data[np.random.randint(0, bg_data.size, np.count_nonzero(col_m_mask))]

    return cleaned


def process_psrfits_example(fits_path, engine_path, start_subint=0, ntodo=0, gpu_batch=1, nomask=False,
                            pipeline_depth=4, morph_size=7, nowriteback=False, mask_output_dir=None,
                            timing_out=None):
    """3-stage pipeline: Reader (CPU) -> Predictor (GPU) -> Writer (CPU)."""
    file_lock = threading.Lock()
    inference_times, read_times, write_times = [], [], []
    read_stage_times, infer_stage_times, save_stage_times, post_stage_times = [], [], [], []
    row_stage_times = []

    # 1. 自动探测 Engine 实际分辨率，确保 Reader 的缩放比例完全正确
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")
    
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:  # type: ignore[attr-defined]
        tmp_engine = runtime.deserialize_cuda_engine(f.read())
        in_name = tmp_engine.get_tensor_name(0)
        profile = tmp_engine.get_tensor_profile_shape(in_name, 0)
        target_h, target_w = profile[1][2], profile[1][3]
    
    print(f"🔍 Auto-detected Engine Resolution: {target_w}x{target_h}")

    base_name = os.path.basename(fits_path).replace(".fits", "")
    # Mask Directory Logic
    mask_dir = None
    if not nomask:
        if mask_output_dir:
             mask_dir = mask_output_dir
        else:
             mask_dir = "results/AI_RFI" # Default Repo location

    if mask_dir:
        # Safety: If user specified a custom dir, we respect it.
        # But we still clear it? Be careful. Let's ONLY clear default "results/AI_RFI".
        # If user provides --maskdir /home/cbm/data, we shouldn't wipe it.
        
        target = os.path.normpath(mask_dir)
        is_default_dir = (target == os.path.normpath("results/AI_RFI"))

        # Create directory if missing
        os.makedirs(target, exist_ok=True)
        
        # Only clear contents if it is the default directory to avoid data loss in custom dirs
        if is_default_dir:
            try:
                if os.listdir(target):
                    for name in os.listdir(target):
                        p = os.path.join(target, name)
                        if os.path.isdir(p) and not os.path.islink(p):
                            shutil.rmtree(p)
                        else:
                            os.remove(p)
            except Exception as e:
                print(f"⚠️ Warning: Failed to clear default mask dir {target}: {e}")
    
    # NOTE on File Mode:
    # If nowriteback is True, we only need 'r' (read-only).
    # If nowriteback is False (default), we need 'rw' (read-write).
    fits_mode = 'r' if nowriteback else 'rw'

    ready_queue = Queue(maxsize=pipeline_depth)
    done_queue = Queue(maxsize=pipeline_depth)

    # Open file once and share handle with lock
    run_t0 = time.perf_counter()
    with fitsio.FITS(fits_path, mode=fits_mode) as fits:
        if 'SUBINT' not in fits:
            print("❌ Error: SUBINT table not found"); return
        
        subint_table = fits['SUBINT']
        header = subint_table.read_header()
        nchan, nsblk, npol = header.get('NCHAN', 1), header.get('NSBLK', 1), header.get('NPOL', 1)
        total_rows = subint_table.get_nrows()
        total_subints = min(ntodo, total_rows - start_subint) if ntodo > 0 else (total_rows - start_subint)
        
        if total_subints <= 0:
            print("⚠️ No data to process"); return
        print(f"📊 Processing: {total_subints} subints, Batch={gpu_batch}")

        def reader_worker():
            """Stage 1: Read + Pre-process"""
            warned = False
            try:
                for i in range(total_subints):
                    idx = start_subint + i
                    row_t0 = time.perf_counter()

                    read_t0 = time.perf_counter()
                    t0 = time.perf_counter()
                    with file_lock:
                        raw = subint_table['DATA'][idx].flatten()
                    read_elapsed = time.perf_counter() - read_t0
                    read_times.append(read_elapsed)
                    read_stage_times.append(read_elapsed)
                    
                    prep_t0 = time.perf_counter()
                    try:
                        data_3d = raw.reshape(nsblk, npol, nchan)
                        subint_2d = data_3d[:, 0, :].T
                    except:
                        subint_2d = raw.reshape(nchan, nsblk); data_3d = None
                    
                    img_prep = np.flipud(subint_2d)
                    img_norm = TRTInference.normalize_image_mean_std(img_prep)
                    
                    # Adaptive sizing logic
                    cur_h, cur_w = img_norm.shape # (nchan, nsblk)
                    if (cur_h != target_h or cur_w != target_w) and not warned:
                        print(f"⚠️  Input data size ({cur_h}x{cur_w}) does not match model target ({target_h}x{target_w}).")
                        print(f"   Action: Downsampling/Upscaling subints to match --width and --height.")
                        warned = True
                    
                    # Use INTER_AREA for downsampling (better quality), INTER_LINEAR for upscaling
                    interp = cv2.INTER_AREA if (cur_h > target_h or cur_w > target_w) else cv2.INTER_LINEAR
                    img_resized = cv2.resize(img_norm, (target_w, target_h), interpolation=interp)
                    prep_elapsed = time.perf_counter() - prep_t0
                    
                    meta = {'data_3d': data_3d, 'raw_dtype': raw.dtype, 'orig_c': nchan, 'orig_t': nsblk, 'npol': npol}
                    meta['read_elapsed'] = read_elapsed
                    meta['prep_elapsed'] = prep_elapsed
                    meta['row_t0'] = row_t0
                    ready_queue.put((idx, img_resized, subint_2d, meta))
                ready_queue.put(None)
            except Exception as e:
                print(f"❌ Reader Error: {e}"); ready_queue.put(None)

        def gpu_worker():
            """Stage 2: GPU Inference"""
            ctx = cuda.Device(0).make_context()  # type: ignore[attr-defined]
            try:
                predictor = TRTInference(engine_path)
                actual_batch = min(gpu_batch, predictor.max_batch_size)
                while True:
                    batch_tasks, stop = [], False
                    for _ in range(actual_batch):
                        task = ready_queue.get()
                        if task is None: stop = True; break
                        batch_tasks.append(task)
                    
                    if batch_tasks:
                        t0 = time.perf_counter()
                        if len(batch_tasks) == 1:
                            imgs = batch_tasks[0][1][np.newaxis, np.newaxis, ...]
                        else:
                            imgs = np.stack([t[1] for t in batch_tasks])[:, np.newaxis, ...]
                        
                        outputs = predictor.predict_batch(imgs)
                        infer_elapsed = time.perf_counter() - t0
                        it = infer_elapsed / len(batch_tasks)
                        
                        for i, task in enumerate(batch_tasks):
                            task_meta = dict(task[3])
                            task_meta['infer_elapsed'] = it
                            done_queue.put((task[0], task[2], outputs[i], task_meta, it))
                    if stop: done_queue.put(None); break
            except Exception as e:
                print(f"❌ GPU Worker Error: {e}")
                done_queue.put(None)
            finally:
                ctx.pop()

        Thread(target=reader_worker, daemon=True).start()
        Thread(target=gpu_worker, daemon=True).start()

        processed = 0
        while processed < total_subints:
            item = done_queue.get()
            if item is None: break
            
            idx, subint_2d, raw_out, meta, it = item
            if it: inference_times.append(it)
            if 'infer_elapsed' in meta:
                infer_stage_times.append(meta['infer_elapsed'])
            
            # Post-process
            post_t0 = time.perf_counter()
            if raw_out is not None:
                # --- Class Prioritization during Argmax ---
                # Boost Point(3) and Horizontal(1) class probability to improve Recall/Precision
                # raw_out shape: (num_classes, H, W)
                probs = raw_out.copy()
                probs[3] *= 1.25  # Boost Point class
                probs[1] *= 1.15  # Boost Horizontal class
                mask_s = np.argmax(probs, axis=0).astype(np.uint8)
                
                # --- Horizontal(1) Line Repair ---
                # Connect fragmented horizontal RFI (often missed in low-contrast areas)
                h_mask = (mask_s == 1).astype(np.uint8)
                # Reduced horizontal kernel size to 31 to be less aggressive and reduce FPs
                h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
                h_closed = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, h_kernel)
                mask_s[(h_closed == 1) & (mask_s == 0)] = 1

                # --- Point-to-Block Post-processing ---
                if morph_size > 0:
                    # Merge Point(3) and Block(4) features
                    pts_mask = ((mask_s == 3) | (mask_s == 4) | (mask_s == 2)).astype(np.uint8)

                    # --- CHANGE: allow larger morphology kernel so clustered points inside large
                    # rectangular regions are merged into connected components and can be
                    # classified as Block(4). Previously kernel_size was capped very small
                    # (<=3) which prevented large-scale merging.
                    # We cap at a reasonable upper bound to avoid excessive dilation.
                    kernel_size = max(1, min(morph_size, 31))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                    closed = cv2.morphologyEx(pts_mask, cv2.MORPH_CLOSE, kernel)

                    # Analyze connected components to distinguish isolated points from blocks
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]
                        aspect_ratio = float(w) / h if h > 0 else 0.0

                        # Time-fill ratio (Solidity in time domain): periodic points are sparse,
                        # while Blocks/Verticals are dense. Use kernel-based thresholds to be
                        # adaptive to chosen morphology size.
                        time_fill_ratio = area / float(max(1, (w * h)))

                        # HEURISTIC 1: Keep as Point(3) if extremely sparse in time-filling
                        if time_fill_ratio < 0.45 and w > max(10, kernel_size * 2):
                            mask_s[labels == i] = 3
                            continue

                        # HEURISTIC 2: Detect substantial Blocks
                        # Lower thresholds compared to previous conservative values so that
                        # dense clusters of points within large rectangles are recognized.
                        block_condition = (
                            (area > 800 and w >= max(20, kernel_size * 2) and h >= max(20, kernel_size * 2) and 0.25 < aspect_ratio < 4.0)
                            or (area > 1500 and 0.2 < aspect_ratio < 5.0)
                        )

                        if block_condition:
                            mask_s[(labels == i) & ((mask_s == 0) | (mask_s == 3) | (mask_s == 2))] = 4
                        elif area < 400:
                            # Very small components are almost certainly points
                            mask_s[labels == i] = 3
                        else:
                            # Keep original prediction for ambiguous cases
                            pass
                
                mask = np.flipud(cv2.resize(mask_s, (meta['orig_t'], meta['orig_c']), interpolation=cv2.INTER_NEAREST))
            else:
                mask = np.zeros_like(subint_2d, dtype=np.uint8)
            post_elapsed = time.perf_counter() - post_t0
            post_stage_times.append(post_elapsed)

            save_t0 = time.perf_counter()
            if mask_dir:
                # Save categorical mask (class ids) for visualization/overlay.
                # Note: AI_RFI内部 mask 已用于清洗，此处仅修正保存方向，避免可视化上下颠倒。
                mask_to_save = np.flipud(mask).astype(np.uint8, copy=False)
                # Unified naming convention: {basename}_block{idx}.png
                cv2.imwrite(os.path.join(mask_dir, f"{base_name}_block{idx:04d}.png"), mask_to_save)
            save_elapsed = time.perf_counter() - save_t0
            save_stage_times.append(save_elapsed)
            
            cleaned_2d = replace_masked_pixels(subint_2d, mask)
            
            if not nowriteback:
                try:
                    if meta['data_3d'] is not None:
                        c3d = meta['data_3d'].copy(); c3d[:, 0, :] = cleaned_2d.T; final = c3d.flatten()
                    else: final = cleaned_2d.flatten()
                    
                    final_r = final.reshape(1, meta['orig_t'], meta['npol'], meta['orig_c'], 1).astype(meta['raw_dtype'])
                    with file_lock:
                        subint_table.write_column('DATA', final_r, firstrow=idx)
                    print(f"✅ Subint {idx} fixed.")
                except Exception as e: print(f"❌ Subint {idx} write failed: {e}")
            else:
                 # Debug mode or No-Write mode: just log progress
                 if processed % 10 == 0:
                     print(f"✅ Subint {idx} processed (No Write).")
            
            write_elapsed = time.perf_counter() - save_t0
            write_times.append(write_elapsed)
            row_elapsed = time.perf_counter() - meta.get('row_t0', save_t0)
            row_stage_times.append(row_elapsed)
            processed += 1

    def summarize(label, values):
        if not values: return
        times = np.array(values) * 1000
        print(f"\n⏱️ {label} stats ({len(times)} calls):")
        print(f"   Min:    {np.min(times):.2f} ms")
        print(f"   Max:    {np.max(times):.2f} ms")
        print(f"   Mean:   {np.mean(times):.2f} ms")
        print(f"   Median: {np.median(times):.2f} ms")
        print(f"   Std:    {np.std(times):.2f} ms")

    summarize("Read", read_times)
    summarize("Inference", inference_times)
    summarize("Write", write_times)

    total_elapsed = time.perf_counter() - run_t0

    def stat_line(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return "n=0"
        return f"n={arr.size}, total={arr.sum():.3f}s, mean={arr.mean():.4f}s, median={np.median(arr):.4f}s, min={arr.min():.4f}s, max={arr.max():.4f}s"

    print(f"\n⏱️ Total time: {total_elapsed:.3f}s")
    print(f"⏱️ Avg per subint: {total_elapsed / max(1, total_subints):.4f}s")
    print(f"⏱️ Read/lock:      {stat_line(read_stage_times)}")
    print(f"⏱️ Inference batch:{stat_line(infer_stage_times)}")
    print(f"⏱️ Post-process:   {stat_line(post_stage_times)}")
    print(f"⏱️ Save mask:      {stat_line(save_stage_times)}")
    print(f"⏱️ Row total:      {stat_line(row_stage_times)}")

    report_path = timing_out or "results/AI_RFI_timing.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AI_RFI timing report\n")
        f.write(f"fits={fits_path}\n")
        f.write(f"engine={engine_path}\n")
        f.write(f"start_subint={start_subint}\n")
        f.write(f"ntodo={ntodo}\n")
        f.write(f"batch={gpu_batch}\n")
        f.write(f"processed_subints={total_subints}\n")
        f.write(f"total_elapsed={total_elapsed:.3f}s\n")
        f.write(f"avg_per_subint={(total_elapsed / max(1, total_subints)):.4f}s\n")
        f.write(f"read: {stat_line(read_stage_times)}\n")
        f.write(f"infer: {stat_line(infer_stage_times)}\n")
        f.write(f"post: {stat_line(post_stage_times)}\n")
        f.write(f"save: {stat_line(save_stage_times)}\n")
        f.write(f"row: {stat_line(row_stage_times)}\n")
    print(f"⏱️ Timing report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI RFI Mitigation for PSRFITS")
    parser.add_argument("--fits", 
                        default="/mnt/d/FASTData/FITSFiles/G90.22-0.08_20220808_snapshot-M19-P4-c2048b1_test.fits",
                        help="Path to the PSRFITS file")
    parser.add_argument("--engine", required=True, help="Path to the TensorRT engine")
    parser.add_argument("--start", type=int, default=0, help="Start subint index")
    parser.add_argument("--ntodo", type=int, default=0, help="Total subints to process")
    parser.add_argument("--batch", type=int, default=1, help="GPU batch size")
    parser.add_argument("--nomask", action="store_true", help="Do not save mask PNG")
    parser.add_argument("--nowriteback", action="store_true", help="Do not modify original FITS file (Dry run)")
    parser.add_argument("--maskdir", type=str, default=None, help="Directory to save masks (Overrides default results/AI_RFI)")
    parser.add_argument("--pipeline", type=int, default=1, help="Pipeline depth multiplier")
    parser.add_argument("--morph", type=int, default=7, help="Morphological kernel size to merge points into blocks (0 to disable)")
    parser.add_argument("--timing-out", type=str, default=None, help="Path to timing report file (default: results/AI_RFI_timing.txt)")
    
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(1)

    args = parser.parse_args()
    if os.path.exists(args.fits):
        print("="*60)
        print(f"🌟 Starting AI RFI Mitigation Pipeline")
        print(f"📂 FITS File: {os.path.abspath(args.fits)}")
        print(f"⚙️  Engine:    {os.path.abspath(args.engine)}")
        print(f"🚀 Batch Size: {args.batch}")
        print(f"🛠️  Morph Size: {args.morph}")
        if args.nowriteback: print("🔒 No Writeback Mode Enabled (Original File Safe)")
        if args.maskdir: print(f"🖼️  Custom Mask Dir: {args.maskdir}")
        print("="*60)
        process_psrfits_example(args.fits, args.engine, args.start, args.ntodo, args.batch, args.nomask, args.batch * args.pipeline, morph_size=args.morph, nowriteback=args.nowriteback, mask_output_dir=args.maskdir, timing_out=args.timing_out)
    else:
        print(f"❌ File not found: {args.fits}")

