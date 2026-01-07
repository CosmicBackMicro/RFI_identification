import fitsio
import numpy as np
# Fix for older TensorRT versions that reference deprecated numpy aliases
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

import os
import time
import cv2
import argparse
import sys
import threading
import gc
from queue import Queue, Empty
from threading import Thread
from concurrent.futures import ProcessPoolExecutor

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
        
        # Compatibility for TRT 8.0
        self.input_names = []
        self.output_names = []
        self.bindings = [None] * self.engine.num_bindings
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError("No input tensors found.")
        self.primary_input = self.input_names[0]
        input_idx = self.engine.get_binding_index(self.primary_input)

        # Pre-allocate buffers
        profile_shape = self.engine.get_profile_shape(input_idx, 0)
        self.target_shape = tuple(profile_shape[1])
        max_input_shape = tuple(profile_shape[2])
        input_dtype = trt.nptype(self.engine.get_binding_dtype(input_idx))
        
        self.h_input = cuda.pagelocked_empty(max_input_shape, input_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.bindings[input_idx] = int(self.d_input)
        
        self.context.set_binding_shape(input_idx, max_input_shape)
        
        self.primary_output = self.output_names[0]
        self.output_host_buffers = {}
        self.output_device_buffers = {}
        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            max_output_shape = tuple(self.context.get_binding_shape(idx))
            output_dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            host_buf = cuda.pagelocked_empty(max_output_shape, output_dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)
            self.output_host_buffers[name] = host_buf
            self.output_device_buffers[name] = device_buf
            self.bindings[idx] = int(device_buf)
        
        self.context.set_binding_shape(input_idx, self.target_shape)
        self.max_batch_size = max_input_shape[0]
        
        print(f"🚀 Engine loaded. Target: {self.target_shape}, Max: {max_input_shape}")
        
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
        
        input_idx = self.engine.get_binding_index(self.primary_input)
        self.context.set_binding_shape(input_idx, (batch_size, 1, self.target_shape[2], self.target_shape[3]))
        
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
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
    
    clean_mask = (mask == 0)
    clean_counts = np.sum(clean_mask, axis=1)
    rows_to_fix = np.where((clean_counts > 0) & (clean_counts > 0.05 * T))[0]
    
    for c in rows_to_fix:
        m_idx = mask[c] > 0
        if not np.any(m_idx): continue
        bg_data = data[c, clean_mask[c]]
        cleaned[c, m_idx] = bg_data[np.random.randint(0, bg_data.size, np.count_nonzero(m_idx))]

    still_masked = (mask > 0) & (np.abs(cleaned - data) < 1e-7)
    clean_counts_col = np.sum(clean_mask, axis=0)
    cols_to_fix = np.where(clean_counts_col > 0)[0]
    
    for t in cols_to_fix:
        col_m_mask = still_masked[:, t]
        if not np.any(col_m_mask): continue
        bg_data = data[clean_mask[:, t], t]
        cleaned[col_m_mask, t] = bg_data[np.random.randint(0, bg_data.size, np.count_nonzero(col_m_mask))]

    return cleaned


def post_process_task(item, nomask, mask_dir, base_name):
    """CPU intensive task: Mask generation + RFI mitigation + Data flattening."""
    idx, subint_2d, raw_out, meta, it = item
    
    # 1. Generate Mask
    if raw_out is not None:
        mask_s = np.argmax(raw_out, axis=0) if raw_out.shape[0] > 1 else (raw_out[0] > 0.5).astype(np.uint8)
        mask = np.flipud(cv2.resize(mask_s, (meta['orig_t'], meta['orig_c']), interpolation=cv2.INTER_NEAREST))
    else:
        mask = np.zeros_like(subint_2d, dtype=np.uint8)

    # 2. RFI Mitigation
    cleaned_2d = replace_masked_pixels(subint_2d, mask)
    
    # 3. Prepare for FITS
    if meta['data_3d'] is not None:
        c3d = meta['data_3d'].copy()
        c3d[:, 0, :] = cleaned_2d.T
        final = c3d.flatten()
    else:
        final = cleaned_2d.flatten()
    
    final_r = final.reshape(1, meta['orig_t'], meta['npol'], meta['orig_c'], 1).astype(meta['raw_dtype'])
    
    # 4. Save Mask (Optional)
    if not nomask and mask_dir:
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}_sub{idx}.png"), (mask * 255).astype(np.uint8))
        
    return idx, final_r, it


def process_psrfits_example(fits_path, engine_path, start_subint=0, ntodo=0, gpu_batch=1, nomask=False,
                            pipeline_depth=4, width=512, height=512):
    """
    4-stage pipeline: 
    1. Reader (Thread) -> 2. GPU Predictor (Thread) -> 3. Post-Process Dispatcher (Thread/ProcessPool) -> 4. Writer (Main Thread)
    """
    file_lock = threading.Lock()
    inference_times, read_times, write_times = [], [], []
    base_name = os.path.basename(fits_path).replace(".fits", "")
    mask_dir = "output/masks" if not nomask else None
    if mask_dir: os.makedirs(mask_dir, exist_ok=True)

    ready_queue = Queue(maxsize=pipeline_depth * gpu_batch)
    gpu_done_queue = Queue(maxsize=pipeline_depth * gpu_batch)
    post_process_queue = Queue(maxsize=pipeline_depth * gpu_batch)

    # 使用 4 个进程进行后处理
    cpu_executor = ProcessPoolExecutor(max_workers=4)

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
        print(f"📊 Processing: {total_subints} subints, Batch={gpu_batch}, CPU Workers=4")

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
                    img_resized = cv2.resize(img_norm, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    meta = {'data_3d': data_3d, 'raw_dtype': raw.dtype, 'orig_c': nchan, 'orig_t': nsblk, 'npol': npol}
                    ready_queue.put((idx, img_resized, subint_2d, meta))
                ready_queue.put(None)
            except Exception as e:
                print(f"❌ Reader Error: {e}"); ready_queue.put(None)

        def gpu_worker():
            """Stage 2: GPU Inference"""
            if cuda is None:
                print("❌ GPU Worker: PyCUDA not available")
                while True:
                    task = ready_queue.get()
                    if task is None: gpu_done_queue.put(None); break
                    gpu_done_queue.put((task[0], task[2], None, task[3], None))
                return

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
                            gpu_done_queue.put((task[0], task[2], outputs[i], task[3], it))
                    if stop: gpu_done_queue.put(None); break
            finally:
                if predictor: del predictor
                gc.collect()
                ctx.pop()

        def post_process_dispatcher():
            """Stage 3: Dispatch to ProcessPool"""
            while True:
                item = gpu_done_queue.get()
                if item is None:
                    post_process_queue.put(None)
                    break
                # Submit to ProcessPool
                future = cpu_executor.submit(post_process_task, item, nomask, mask_dir, base_name)
                post_process_queue.put(future)

        Thread(target=reader_worker, daemon=True).start()
        Thread(target=gpu_worker, daemon=True).start()
        Thread(target=post_process_dispatcher, daemon=True).start()

        processed = 0
        while processed < total_subints:
            future = post_process_queue.get()
            if future is None: break
            
            try:
                idx, final_r, it = future.result()
                if it: inference_times.append(it)
                
                t0 = time.perf_counter()
                with file_lock:
                    subint_table.write_column('DATA', final_r, firstrow=idx)
                write_times.append(time.perf_counter() - t0)
                
                if processed % 10 == 0 or processed == total_subints - 1:
                    print(f"✅ Subint {idx} processed ({processed+1}/{total_subints})")
            except Exception as e:
                print(f"❌ Subint processing failed: {e}")
            
            processed += 1

    cpu_executor.shutdown()
    
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
    parser.add_argument("--width", type=int, default=512, help="Model input width")
    parser.add_argument("--height", type=int, default=512, help="Model input height")
    
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(1)

    args = parser.parse_args()
    if os.path.exists(args.fits):
        process_psrfits_example(args.fits, args.engine, args.start, args.ntodo, args.batch, args.nomask, 
                                args.batch * args.pipeline, args.width, args.height)
    else:
        print(f"❌ File not found: {args.fits}")
