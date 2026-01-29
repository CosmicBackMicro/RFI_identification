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
        cuda.init()
except ImportError:
    trt = None
    cuda = None


class TRTInference:
    def __init__(self, engine_path):
        if trt is None or cuda is None:
            raise RuntimeError("tensorrt or pycuda not installed.")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = []
        self.output_names = []
        for name in self.tensor_names:
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError("No input tensors found.")
        self.primary_input = self.input_names[0]

        # Pre-allocate buffers
        profile_shape = self.engine.get_tensor_profile_shape(self.primary_input, 0)
        self.target_shape = tuple(profile_shape[1])
        max_input_shape = tuple(profile_shape[2])
        input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.primary_input))
        
        self.h_input = cuda.pagelocked_empty(max_input_shape, input_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        
        self.context.set_input_shape(self.primary_input, max_input_shape)
        
        self.primary_output = self.output_names[0]
        self.output_host_buffers = {}
        self.output_device_buffers = {}
        for name in self.output_names:
            max_output_shape = tuple(self.context.get_tensor_shape(name))
            output_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_buf = cuda.pagelocked_empty(max_output_shape, output_dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)
            self.output_host_buffers[name] = host_buf
            self.output_device_buffers[name] = device_buf
        
        self.context.set_input_shape(self.primary_input, self.target_shape)
        self.max_batch_size = max_input_shape[0]
        
        print(f"🚀 Engine loaded. Target: {self.target_shape}, Max: {max_input_shape}")

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
        
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        primary_host = self.output_host_buffers[self.primary_output]
        cuda.memcpy_dtoh_async(primary_host[:batch_size], self.output_device_buffers[self.primary_output], self.stream)
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
                            pipeline_depth=4):
    """3-stage pipeline: Reader (CPU) -> Predictor (GPU) -> Writer (CPU)."""
    file_lock = threading.Lock()
    inference_times, read_times, write_times = [], [], []
    base_name = os.path.basename(fits_path).replace(".fits", "")
    # Save inference masks under results/AI_RFI (repo-conventional results folder)
    mask_dir = "results/AI_RFI" if not nomask else None
    if mask_dir:
        # Safety: only ever delete contents inside the intended directory.
        safe_root = os.path.normpath("results/AI_RFI")
        target = os.path.normpath(mask_dir)
        if target != safe_root:
            raise RuntimeError(f"Refusing to clear unexpected mask_dir: {mask_dir}")

        # If directory exists and is non-empty, clear it before writing new results.
        if os.path.isdir(target):
            try:
                if os.listdir(target):
                    for name in os.listdir(target):
                        p = os.path.join(target, name)
                        if os.path.isdir(p) and not os.path.islink(p):
                            shutil.rmtree(p)
                        else:
                            os.remove(p)
            except Exception as e:
                raise RuntimeError(f"Failed to clear existing contents under {target}: {e}")
        else:
            os.makedirs(target, exist_ok=True)

    ready_queue = Queue(maxsize=pipeline_depth)
    done_queue = Queue(maxsize=pipeline_depth)

    # Open file once and share handle with lock
    with fitsio.FITS(fits_path, mode='rw') as fits:
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
            try:
                for i in range(total_subints):
                    idx = start_subint + i
                    t0 = time.perf_counter()
                    with file_lock:
                        raw = subint_table['DATA'][idx].flatten()
                    read_times.append(time.perf_counter() - t0)
                    
                    try:
                        data_3d = raw.reshape(nsblk, npol, nchan)
                        subint_2d = data_3d[:, 0, :].T
                    except:
                        subint_2d = raw.reshape(nchan, nsblk); data_3d = None
                    
                    img_prep = np.flipud(subint_2d)
                    img_norm = TRTInference.normalize_image_mean_std(img_prep)
                    img_resized = cv2.resize(img_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
                    
                    meta = {'data_3d': data_3d, 'raw_dtype': raw.dtype, 'orig_c': nchan, 'orig_t': nsblk, 'npol': npol}
                    ready_queue.put((idx, img_resized, subint_2d, meta))
                ready_queue.put(None)
            except Exception as e:
                print(f"❌ Reader Error: {e}"); ready_queue.put(None)

        def gpu_worker():
            """Stage 2: GPU Inference"""
            ctx = cuda.Device(0).make_context()
            predictor = None
            try:
                if os.path.exists(engine_path):
                    predictor = TRTInference(engine_path)
                
                actual_batch = min(gpu_batch, predictor.max_batch_size) if predictor else gpu_batch
                while True:
                    batch_tasks, stop = [], False
                    for _ in range(actual_batch):
                        task = ready_queue.get()
                        if task is None: stop = True; break
                        batch_tasks.append(task)
                    
                    if batch_tasks:
                        if predictor:
                            t0 = time.perf_counter()
                            if len(batch_tasks) == 1:
                                imgs = batch_tasks[0][1][np.newaxis, np.newaxis, ...]
                            else:
                                imgs = np.stack([t[1] for t in batch_tasks])[:, np.newaxis, ...]
                            
                            outputs = predictor.predict_batch(imgs)
                            it = (time.perf_counter() - t0) / len(batch_tasks)
                        else:
                            outputs, it = [None] * len(batch_tasks), None
                        
                        for i, task in enumerate(batch_tasks):
                            done_queue.put((task[0], task[2], outputs[i], task[3], it))
                    if stop: done_queue.put(None); break
            finally:
                if predictor: del predictor
                gc.collect()
                ctx.pop()

        Thread(target=reader_worker, daemon=True).start()
        Thread(target=gpu_worker, daemon=True).start()

        processed = 0
        while processed < total_subints:
            item = done_queue.get()
            if item is None: break
            
            idx, subint_2d, raw_out, meta, it = item
            if it: inference_times.append(it)
            
            # Post-process
            if raw_out is not None:
                mask_s = np.argmax(raw_out, axis=0) if raw_out.shape[0] > 1 else (raw_out[0] > 0.5).astype(np.uint8)
                mask = np.flipud(cv2.resize(mask_s, (meta['orig_t'], meta['orig_c']), interpolation=cv2.INTER_NEAREST))
            else:
                mask = np.zeros_like(subint_2d, dtype=np.uint8)

            t0 = time.perf_counter()
            if mask_dir:
                # Save categorical mask (class ids) for visualization/overlay.
                # Note: AI_RFI内部 mask 已用于清洗，此处仅修正保存方向，避免可视化上下颠倒。
                mask_to_save = np.flipud(mask).astype(np.uint8, copy=False)
                cv2.imwrite(os.path.join(mask_dir, f"{base_name}_sub{idx}.png"), mask_to_save)
            
            cleaned_2d = replace_masked_pixels(subint_2d, mask)
            try:
                if meta['data_3d'] is not None:
                    c3d = meta['data_3d'].copy(); c3d[:, 0, :] = cleaned_2d.T; final = c3d.flatten()
                else: final = cleaned_2d.flatten()
                
                final_r = final.reshape(1, meta['orig_t'], meta['npol'], meta['orig_c'], 1).astype(meta['raw_dtype'])
                with file_lock:
                    subint_table.write_column('DATA', final_r, firstrow=idx)
                print(f"✅ Subint {idx} fixed.")
            except Exception as e: print(f"❌ Subint {idx} write failed: {e}")
            
            write_times.append(time.perf_counter() - t0)
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
    parser.add_argument("--pipeline", type=int, default=1, help="Pipeline depth multiplier")
    
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(1)

    args = parser.parse_args()
    if os.path.exists(args.fits):
        process_psrfits_example(args.fits, args.engine, args.start, args.ntodo, args.batch, args.nomask, args.batch * args.pipeline)
    else:
        print(f"❌ File not found: {args.fits}")

