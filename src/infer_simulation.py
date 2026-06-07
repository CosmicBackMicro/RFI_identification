#!/usr/bin/env python3
"""Infer first N subints from a PSRFITS file using a TensorRT engine or fallback to PyTorch.

Saves predicted masks as PNG files named `simulation_psrfits_block{idx:04d}.png` in output dir.
"""
import os
import argparse
import numpy as np
import cv2
from PIL import Image

def normalize_image_mean_std(image: np.ndarray, k: float = 5.0) -> np.ndarray:
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

def preprocess(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # Flip, normalize, resize and add channel
    img = np.flipud(img)
    img = normalize_image_mean_std(img)
    img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if img_resized.ndim == 2:
        img_resized = img_resized[np.newaxis, ...]
    return img_resized.astype(np.float32)

def save_mask(mask_arr: np.ndarray, out_path: str):
    # mask_arr should be 2D of ints (0..C-1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im = Image.fromarray(mask_arr.astype(np.uint8))
    im.save(out_path)

def infer_trt(engine_path, fits_path, out_dir, num_subint=500, height=896, width=1024):
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except Exception as e:
        print(f"TensorRT/pycuda not available: {e}")
        return False

    # Helper: load engine
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("Failed to deserialize engine")
        return False

    context = engine.create_execution_context()

    # Input/Output binding info
    bindings = [None] * engine.num_bindings
    for b in range(engine.num_bindings):
        name = engine.get_binding_name(b)
        shape = engine.get_binding_shape(b)
        dtype = trt.nptype(engine.get_binding_dtype(b))
        is_input = engine.binding_is_input(b)
        bindings[b] = {'index': b, 'name': name, 'shape': shape, 'dtype': dtype, 'is_input': is_input}

    # Determine input binding and prob output
    input_binding = next((b for b in bindings if b['is_input']), None)
    output_binding = next((b for b in bindings if not b['is_input'] and 'probs' in b['name'].lower()), None)
    if input_binding is None or output_binding is None:
        output_binding = next((b for b in bindings if not b['is_input']), None)

    # Allocate host/device buffers
    import numpy as np
    h_input = None
    d_input = None
    d_output = None

    # Open fits and iterate
    try:
        import fitsio
        f = fitsio.FITS(fits_path)
        subint_hdu = f['SUBINT']
        hdr = subint_hdu.read_header()
        nrows = subint_hdu.get_nrows()
        max_rows = min(num_subint, nrows)

        for idx in range(max_rows):
            row = subint_hdu[idx]
            try:
                raw = row['DATA'].astype(np.float32)
                scl = row['DAT_SCL']
                offs = row['DAT_OFFS']
            except Exception:
                # fallback: try reading first non-empty image hdu
                with fitsio.FITS(fits_path) as ff:
                    img = ff[0].read().astype(np.float32)
                raw = None

            if raw is not None:
                # need nsblk, npol, nchan from header
                nsblk = int(hdr.get('NSBLK', 1))
                nchan = int(hdr.get('NCHAN', raw.size))
                npol = int(hdr.get('NPOL', 1))
                try:
                    data_3d = raw.reshape(nsblk, npol, nchan)
                    img = (data_3d[:, 0, :] * scl) + offs
                    img = img.T
                except Exception:
                    try:
                        img = raw.reshape(nchan, nsblk)
                    except Exception:
                        print(f"Could not reshape raw data for subint {idx}")
                        continue
            # preprocess
            proc = preprocess(img, out_h=height, out_w=width)
            # TRT expects NCHW
            inp = np.expand_dims(proc, axis=0)  # (1, C, H, W)

            # Allocate buffers per iteration to handle dynamic shapes
            # Host
            h_input = cuda.pagelocked_empty(inp.nbytes, np.uint8)
            np.copyto(np.frombuffer(h_input, dtype=inp.dtype).reshape(inp.shape), inp)
            d_input = cuda.mem_alloc(inp.nbytes)

            # Prepare output shape from binding
            out_shape = tuple(context.get_binding_shape(output_binding['index']))
            # If dynamic (-1), replace with actual
            out_shape = tuple([s if s != -1 else dim for s, dim in zip(output_binding['shape'], (1, inp.shape[1], inp.shape[2], inp.shape[3]))])
            out_count = int(np.prod(out_shape))
            h_output = cuda.pagelocked_empty(out_count, np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)

            # Execute
            cuda.memcpy_htod(d_input, np.frombuffer(h_input, dtype=inp.dtype))
            bindings_ptr = [int(d_input), int(d_output)] if engine.num_bindings == 2 else [int(d_input), int(d_output)]
            context.execute_v2(bindings_ptr)
            cuda.memcpy_dtoh(h_output, d_output)

            # reshape output to (C, H, W) or (1,C,H,W)
            out_np = np.array(h_output, dtype=np.float32).reshape(out_shape)
            # find probs channel axis: assume channels first
            if out_np.ndim == 4:
                probs = out_np[0]
            elif out_np.ndim == 3:
                probs = out_np
            else:
                probs = out_np

            preds = np.argmax(probs, axis=0).astype(np.uint8)
            out_name = os.path.join(out_dir, f"simulation_psrfits_block{idx:04d}.png")
            save_mask(preds, out_name)
            print(f"Saved {out_name}")

        f.close()
        return True
    except Exception as e:
        print(f"TRT inference error: {e}")
        return False

def infer_fallback_torch(checkpoint_path, fits_path, out_dir, num_subint=500, height=896, width=1024, arch='segformer', device=None):
    # Load model from checkpoint. Arch chooses the LightningModule source.
    try:
        import torch
    except Exception as e:
        print(f"Torch not available: {e}")
        return False

    # Default device selection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        if arch == 'unet':
            from UNet import UNetLightningModule
        else:
            # segformer
            from SegFormer_StrategyAltered import UNetLightningModule
    except Exception as e:
        print(f"Failed to import Lightning module for arch={arch}: {e}")
        return False

    model = UNetLightningModule.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model.to(device)

    import fitsio
    f = fitsio.FITS(fits_path)
    subint_hdu = f['SUBINT']
    hdr = subint_hdu.read_header()
    nrows = subint_hdu.get_nrows()
    max_rows = min(num_subint, nrows)

    with torch.no_grad():
        for idx in range(max_rows):
            row = subint_hdu[idx]
            try:
                raw = row['DATA'].astype(np.float32)
                scl = row['DAT_SCL']
                offs = row['DAT_OFFS']
            except Exception:
                with fitsio.FITS(fits_path) as ff:
                    img = ff[0].read().astype(np.float32)
                raw = None

            if raw is not None:
                nsblk = int(hdr.get('NSBLK', 1))
                nchan = int(hdr.get('NCHAN', raw.size))
                npol = int(hdr.get('NPOL', 1))
                try:
                    data_3d = raw.reshape(nsblk, npol, nchan)
                    img = (data_3d[:, 0, :] * scl) + offs
                    img = img.T
                except Exception:
                    try:
                        img = raw.reshape(nchan, nsblk)
                    except Exception:
                        print(f"Could not reshape raw data for subint {idx}")
                        continue

            proc = preprocess(img, out_h=height, out_w=width)
            inp = torch.from_numpy(np.expand_dims(proc, axis=0)).to(device)
            out = model(inp)
            if isinstance(out, tuple) or isinstance(out, list):
                logits = out[0]
            else:
                logits = out
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            out_name = os.path.join(out_dir, f"simulation_psrfits_block{idx:04d}.png")
            save_mask(preds, out_name)
            print(f"Saved {out_name}")
    f.close()
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='segformer', choices=['segformer', 'unet'], help='Model architecture to load for torch fallback')
    parser.add_argument('--engine', type=str, help='Path to TensorRT engine (optional)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for torch fallback')
    parser.add_argument('--fits', type=str, required=True, help='Path to PSRFITS file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save masks')
    parser.add_argument('--num_subint', type=int, default=500, help='Number of subints to process')
    parser.add_argument('--height', type=int, default=896, help='Engine/input height')
    parser.add_argument('--width', type=int, default=1024, help='Engine/input width')
    args = parser.parse_args()

    success = False
    if args.engine and os.path.exists(args.engine):
        print(f"Attempting TensorRT inference with engine: {args.engine}")
        success = infer_trt(args.engine, args.fits, args.out_dir, num_subint=args.num_subint, height=args.height, width=args.width)

    if not success and args.checkpoint and os.path.exists(args.checkpoint):
        print("Falling back to torch inference using checkpoint.")
        success = infer_fallback_torch(args.checkpoint, args.fits, args.out_dir, num_subint=args.num_subint, height=args.height, width=args.width, arch=args.arch)

    if not success:
        print("Inference failed (no working backend). Exiting.")

if __name__ == '__main__':
    main()
