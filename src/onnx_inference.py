#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import fitsio
import cv2
from tqdm import tqdm

# prefer local src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import onnxruntime as ort
except Exception as e:
    print("Error: onnxruntime is required to run this script. Please install onnxruntime-gpu or onnxruntime.")
    raise

def _stat_line(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size}, total={arr.sum():.3f}s, mean={arr.mean():.4f}s, "
        f"median={np.median(arr):.4f}s, min={arr.min():.4f}s, max={arr.max():.4f}s"
    )

def preprocess(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std <= 1e-6:
        return np.full(img.shape, 0.5, dtype=np.float32)
    lo, hi = mean - 5.0 * std, mean + 5.0 * std
    np.clip(img, lo, hi, out=img)
    img -= lo
    img /= (hi - lo)
    return img

def apply_post_processing(mask_s, morph_size=7):
    h_mask = (mask_s == 1).astype(np.uint8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1))
    h_closed = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, h_kernel)
    mask_s[(h_closed == 1) & (mask_s == 0)] = 1

    if morph_size > 0:
        pts_mask = ((mask_s == 3) | (mask_s == 4)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        closed = cv2.morphologyEx(pts_mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = float(w) / h
            if area > 400 and w >= 25 and h >= 25 and 0.2 < aspect_ratio < 5.0:
                mask_s[(labels == i) & ((mask_s == 0) | (mask_s == 3))] = 4
            else:
                mask_s[(labels == i) & (mask_s == 0)] = 3
    return mask_s

class ONNXPredictor:
    def __init__(self, onnx_path: str, use_cuda=True):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        available = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        if use_cuda and 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        print(f"Creating ONNXSession (providers={providers}) for: {onnx_path}")
        sess_options = ort.SessionOptions()
        # enable some optimizations
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        inputs = self.sess.get_inputs()
        if len(inputs) == 0:
            raise RuntimeError("ONNX model has no inputs")
        self.input_name = inputs[0].name
        # note: shape may contain None; we'll rely on runtime resizing
        print(f"ONNX input name: {self.input_name}, model input shape: {inputs[0].shape}")

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        # Expect batch as float32, shape (N,1,H,W) in NCHW
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        feeds = {self.input_name: batch}
        outputs = self.sess.run(None, feeds)
        # assume first output is logits N x C x H x W
        return outputs[0]

def process_single_file(fits_path, predictor, args):
    timings = {"read": [], "preprocess": [], "infer": [], "postprocess": [], "save": [], "rows": []}
    mask_dir = args.maskdir or "results/AI_RFI_samples"
    if not args.nomask:
        os.makedirs(mask_dir, exist_ok=True)

    with fitsio.FITS(fits_path, 'r') as fits:
        table = fits['SUBINT'] if 'SUBINT' in fits else fits[1]
        header = table.read_header()
        nchan = int(header.get('NCHAN', 1792))
        nsblk = int(header.get('NSBLK', 8192))
        npol = int(header.get('NPOL', 1))
        total_rows = table.get_nrows()
        total_rows = min(total_rows, int(args.ntodo)) if int(args.ntodo) > 0 else total_rows

        print(f"Found {total_rows} subints, using nchan={nchan}, nsblk={nsblk}, npol={npol}")

        for idx in tqdm(range(total_rows), desc="ONNX Inference"):
            row_t0 = time.perf_counter()

            read_t0 = time.perf_counter()
            raw = table['DATA'][idx].flatten().astype(np.float32)
            read_elapsed = time.perf_counter() - read_t0

            try:
                if npol > 1:
                    data_3d = raw.reshape(nsblk, npol, nchan)
                    subint_2d = data_3d[:, 0, :].T
                else:
                    subint_2d = raw.reshape(nsblk, nchan).T
            except ValueError as exc:
                raise ValueError(f"Failed to reshape DATA for row {idx}: raw={raw.size}, nchan={nchan}, nsblk={nsblk}, npol={npol}") from exc

            preprocess_t0 = time.perf_counter()
            img_prep = np.flipud(subint_2d)
            img_norm = preprocess(img_prep)

            target_h, target_w = int(args.height), int(args.width)
            interp = cv2.INTER_AREA if (img_norm.shape[0] > target_h or img_norm.shape[1] > target_w) else cv2.INTER_LINEAR
            img_resized = cv2.resize(img_norm, (target_w, target_h), interpolation=interp)
            preprocess_elapsed = time.perf_counter() - preprocess_t0

            infer_t0 = time.perf_counter()
            batch_input = img_resized[np.newaxis, np.newaxis, ...].astype(np.float32)
            raw_out = predictor.predict_batch(batch_input)
            infer_elapsed = time.perf_counter() - infer_t0

            post_t0 = time.perf_counter()
            if raw_out.ndim == 4:
                # N C H W
                out0 = raw_out[0]
            elif raw_out.ndim == 3:
                # C H W
                out0 = raw_out
            else:
                raise RuntimeError(f"Unexpected ONNX output shape: {raw_out.shape}")

            if out0.shape[0] > 1:
                mask_s = np.argmax(out0, axis=0).astype(np.uint8)
            else:
                mask_s = (out0[0] > 0.5).astype(np.uint8)

            mask_s = apply_post_processing(mask_s, args.morph)
            mask = np.flipud(cv2.resize(mask_s, (nsblk, nchan), interpolation=cv2.INTER_NEAREST)).astype(np.uint8)
            post_elapsed = time.perf_counter() - post_t0

            save_elapsed = 0.0
            if not args.nomask:
                save_t0 = time.perf_counter()
                mask_name = f"{os.path.basename(fits_path).replace('.fits', '')}_subint{idx:04d}.png"
                cv2.imwrite(os.path.join(mask_dir, mask_name), mask)
                save_elapsed = time.perf_counter() - save_t0

            row_total = time.perf_counter() - row_t0
            timings['rows'].append(row_total)
            timings['read'].append(read_elapsed)
            timings['preprocess'].append(preprocess_elapsed)
            timings['infer'].append(infer_elapsed)
            timings['postprocess'].append(post_elapsed)
            timings['save'].append(save_elapsed)

    return True, timings, total_rows

def write_timing_report(report_path, args, total_rows, timings, total_elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ONNX inference timing report\n")
        f.write(f"fits={args.fits}\n")
        f.write(f"onnx={args.onnx}\n")
        f.write(f"ntodo={args.ntodo}\n")
        f.write(f"processed_subints={total_rows}\n")
        f.write(f"input_size={args.height}x{args.width}\n")
        f.write(f"morph={args.morph}\n")
        f.write(f"nomask={args.nomask}\n")
        f.write(f"total_elapsed={total_elapsed:.3f}s\n")
        f.write(f"avg_per_subint={(total_elapsed / max(1, total_rows)):.4f}s\n")
        f.write(f"read:        {_stat_line(timings['read'])}\n")
        f.write(f"preprocess:  {_stat_line(timings['preprocess'])}\n")
        f.write(f"infer:       {_stat_line(timings['infer'])}\n")
        f.write(f"postprocess: {_stat_line(timings['postprocess'])}\n")
        f.write(f"save:        {_stat_line(timings['save'])}\n")
        f.write(f"row_total:   {_stat_line(timings['rows'])}\n")

def main():
    parser = argparse.ArgumentParser(description="Single-file ONNX RFI inference for PSRFITS")
    parser.add_argument("--fits", required=True, help="Path to a single PSRFITS file")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model file (.onnx)")
    parser.add_argument("--morph", type=int, default=7, help="Morphological size")
    parser.add_argument("--ntodo", type=int, default=0, help="Maximum number of subints to process (0 means all)")
    parser.add_argument("--width", type=int, default=1024, help="Inference input width")
    parser.add_argument("--height", type=int, default=1792, help="Inference input height")
    parser.add_argument("--maskdir", type=str, default=None, help="Output directory for masks")
    parser.add_argument("--timing-out", type=str, default=None, help="Path to timing report file (default: results/onnx_timing.txt)")
    parser.add_argument("--nomask", action="store_true", help="Don't save mask PNGs")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU provider for ONNX Runtime")

    args = parser.parse_args()

    if not os.path.isfile(args.fits):
        print(f"Error: FITS file not found: {args.fits}")
        sys.exit(1)
    if not os.path.isfile(args.onnx):
        print(f"Error: ONNX file not found: {args.onnx}")
        sys.exit(1)

    use_cuda = not args.no_cuda
    predictor = ONNXPredictor(args.onnx, use_cuda=use_cuda)

    run_t0 = time.perf_counter()
    ok, timings, total_rows = process_single_file(args.fits, predictor, args)
    total_elapsed = time.perf_counter() - run_t0

    if ok:
        print("Done. Mask files generated successfully.")
        if timings is None:
            timings = {"read": [], "preprocess": [], "infer": [], "postprocess": [], "save": [], "rows": []}
        print(f"Timing: processed={total_rows}, total={total_elapsed:.3f}s, avg={(total_elapsed / max(1, total_rows)):.4f}s/subint")
        print(f"Timing(read):        {_stat_line(timings['read'])}")
        print(f"Timing(preprocess):  {_stat_line(timings['preprocess'])}")
        print(f"Timing(infer):       {_stat_line(timings['infer'])}")
        print(f"Timing(postprocess): {_stat_line(timings['postprocess'])}")
        print(f"Timing(save):        {_stat_line(timings['save'])}")
        print(f"Timing(row_total):   {_stat_line(timings['rows'])}")

        timing_out = args.timing_out or "results/onnx_timing.txt"
        write_timing_report(timing_out, args, total_rows, timings, total_elapsed)
        print(f"Timing report saved to: {timing_out}")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
