#!/usr/bin/env python3
"""
Export Lightning UNet model to ONNX and TensorRT.

Features:
- Load model from Lightning .ckpt.
- Export with two outputs: 'logits' and 'probs' (after Softmax).
- Supports dynamic axes (batch, height, width) by default.
- Default opset 17 for performance and LayerNorm support.
- Automatic TensorRT engine build from the exported ONNX.

Usage:
python src/exportModel.py \
  --checkpoint path/to/checkpoint/checkpointxx.ckpt \
  --output checkpoints/tensorrt/model.engine \
  --batch 1 --height 512 --width 512 \
  --fp16
"""
import argparse
import os
import torch
import sys
import numpy as np
import cv2
import glob
import random

try:
    import tensorrt as trt
except ImportError:
    trt = None

# Removed hardcoded import: from UNet import UNetLightningModule

class IInt8Calibrator(trt.IInt8EntropyCalibrator2 if trt else object):
    """
    TensorRT INT8 Calibrator using random samples from a directory.
    """
    def __init__(self, calib_dir, batch_size, input_shape, cache_file, max_samples=500):
        if trt:
            trt.IInt8EntropyCalibrator2.__init__(self)
            
        self.calib_dir = calib_dir
        self.batch_size = batch_size
        self.input_shape = input_shape  # (B, C, H, W)
        self.cache_file = cache_file
        self.max_samples = max_samples
        
        # List files
        self.img_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.npy", "*.fits", "*.fit"]:
            self.img_paths.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
        
        if not self.img_paths:
            raise ValueError(f"No valid calibration files found in {calib_dir}")
            
        random.shuffle(self.img_paths)
        self.img_paths = self.img_paths[:max_samples]
        print(f"[Calib] Using {len(self.img_paths)} samples for INT8 calibration.")
        
        self.current_index = 0
        self.device_input = None

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

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.img_paths):
            return None
            
        batch_data = []
        while len(batch_data) < self.batch_size and self.current_index < len(self.img_paths):
            path = self.img_paths[self.current_index]
            self.current_index += 1
            img = None
            try:
                if path.endswith('.npy'):
                    img = np.load(path).astype(np.float32)
                elif path.endswith(('.fits', '.fit')):
                    try:
                        import fitsio
                        with fitsio.FITS(path) as f:
                            if 'SUBINT' in f:
                                # PSRFITS structure as in AI_RFI.py
                                # Use first row for calibration
                                row_data = f['SUBINT'][0] 
                                header = f['SUBINT'].read_header()
                                nchan = header.get('NCHAN', 1)
                                nsblk = header.get('NSBLK', 1)
                                npol = header.get('NPOL', 1)
                                
                                # Extract and scale data: (raw * scl) + offs
                                raw = row_data['DATA'].astype(np.float32)
                                scl = row_data['DAT_SCL']
                                offs = row_data['DAT_OFFS']
                                
                                try:
                                    data_3d = raw.reshape(nsblk, npol, nchan)
                                    # Use first polarization, apply scaling
                                    img = (data_3d[:, 0, :] * scl) + offs
                                    img = img.T # (NCHAN, NSBLK)
                                except:
                                    img = raw.reshape(nchan, nsblk)
                            else:
                                # Plain Image FITS
                                img = f[0].read().astype(np.float32)
                    except (ImportError, Exception):
                        # Fallback to astropy if fitsio fails or is missing
                        from astropy.io import fits
                        with fits.open(path) as hdul:
                            for hdu in hdul:
                                if hdu.data is not None:
                                    img = hdu.data.astype(np.float32)
                                    break
                else:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                
                if img is not None:
                    # 1. Flip (Important: match AI_RFI)
                    img_prep = np.flipud(img)
                    
                    # 2. Normalize (Important: match AI_RFI normalize_image_mean_std)
                    img_norm = self.normalize_image_mean_std(img_prep)
                    
                    # 3. Resize to target
                    img_resized = cv2.resize(img_norm, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)
                    
                    if img_resized.ndim == 2:
                        img_resized = img_resized[np.newaxis, ...]
                    batch_data.append(img_resized)
            except Exception as e:
                print(f"[Calib] Skipping corrupted file {path}: {e}")
                continue

        if not batch_data:
            return None
            
        batch_tensor = torch.from_numpy(np.stack(batch_data)).to('cuda')
        self.device_input = batch_tensor.data_ptr()
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Calib] Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[Calib] Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_tensorrt_engine(onnx_path, engine_path, fp16=True, int8=False, calib_dir=None, input_shape=(1, 1, 512, 512), is_static=False, calib_count=500):
    """
    Build TensorRT engine from ONNX file.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[Error] tensorrt python package not found. Cannot build engine.")
        return

    print(f"[TRT] Starting build for {onnx_path} -> {engine_path}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # EXPLICIT_BATCH is required for ONNX models
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Set memory pool limit for workspace (5GB: balances performance and WSL2 stability)
    memory_pool_type = getattr(trt, "MemoryPoolType", None)
    workspace_size = 5120 * 1024 * 1024 # 5GB
    if hasattr(config, "set_memory_pool_limit") and memory_pool_type:
        config.set_memory_pool_limit(getattr(memory_pool_type, "WORKSPACE"), workspace_size)
        print(f"[TRT] Workspace limit set to {workspace_size/(1024**2):.0f}MB.")
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_size
        print(f"[TRT] Workspace limit set to {workspace_size/(1024**2):.0f}MB.")

    # Tactic Sources: Enable all available sources for maximum performance.
    # Only restrict if you encounter specific library version mismatch crashes.
    tactic_source_enum = getattr(trt, "TacticSource", None)
    if tactic_source_enum and hasattr(config, "set_tactic_sources"):
        try:
            # Use all default sources including CUBLAS_LT which is highly optimized for deep learning
            # Calling set_tactic_sources with a full mask or just not calling it achieves this.
            # Here we explicitly ensure CUBLAS_LT is included if we are setting it.
            print("[TRT] Tactic Sources: Using all available sources (CUBLAS, CUBLAS_LT, CUDNN, etc.) for peak performance.")
        except Exception as e:
            print(f"[TRT] Could not set Tactic Sources: {e}")

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("[TRT] Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Enable FP16 if requested and supported
    if fp16:
        if builder.platform_has_fast_fp16:
            # Use getattr to avoid linter errors on different TRT versions
            builder_flag = getattr(trt, "BuilderFlag", None)
            if builder_flag and hasattr(config, "set_flag"):
                config.set_flag(getattr(builder_flag, "FP16"))
                print("[TRT] FP16 mode enabled (via set_flag).")
            elif hasattr(builder, "fp16_mode"):
                builder.fp16_mode = True
                print("[TRT] FP16 mode enabled (via fp16_mode).")
            else:
                print("[TRT] Could not enable FP16 (API mismatch).")
        else:
            print("[TRT] FP16 requested but not supported by platform. Falling back to FP32.")

    # Optimization Profile
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # Check if input has dynamic shapes (-1)
    has_dynamic = any(d == -1 or d is None for d in input_tensor.shape)
    
    if has_dynamic:
        print(f"[TRT] Dynamic ONNX detected. Creating optimization profile for {input_name}...")
        profile = builder.create_optimization_profile()
        b, c, h, w = input_shape
        
        if is_static:
            # Even if ONNX is dynamic, we can force TRT to be static for better optimization
            min_shape = (b, c, h, w)
            opt_shape = (b, c, h, w)
            max_shape = (b, c, h, w)
            print(f"[TRT] Forcing static shapes in TRT for efficiency: {opt_shape}")
        else:
            # Dynamic axes: allow some range
            min_shape = (1, c, h // 2, w // 2)
            opt_shape = (b, c, h, w)
            max_shape = (b, c, h, w)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    else:
        print(f"[TRT] Static ONNX detected (shape={input_tensor.shape}).")
        if is_static:
            print("[TRT] Static mode: TensorRT will perform maximum optimization for this fixed shape.")

    # Build engine
    # Enable INT8 if requested
    if int8:
        if builder.platform_has_fast_int8:
            if calib_dir and os.path.exists(calib_dir):
                config.set_flag(trt.BuilderFlag.INT8)
                cache_file = os.path.splitext(engine_path)[0] + ".cache"
                config.int8_calibrator = IInt8Calibrator(calib_dir, 1, input_shape, cache_file, max_samples=calib_count)
                print(f"[TRT] INT8 mode enabled (using calibrator on {calib_dir}).")
            else:
                print("[TRT] INT8 requested but --cal directory missing or invalid. Falling back.")
        else:
            print("[TRT] INT8 requested but not supported by platform. Falling back.")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[TRT] Engine build failed.")
        return

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"[TRT] Engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Lightning UNet/SegFormer model to ONNX/TensorRT")
    parser.add_argument('--checkpoint', required=True, help='Path to Lightning .ckpt file')
    parser.add_argument('--arch', type=str, default='unet', choices=['unet', 'segformer'], help='Model architecture type (unet or segformer)')
    
    # Get project root (one level up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(project_root, 'checkpoints', 'tensorrt', 'model.engine')
    
    parser.add_argument('--output', required=False, default=default_output, help=f'Output TensorRT engine path (default: {default_output}). ONNX will be saved with same basename.')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (default: 17 for stability and performance)')
    parser.add_argument('--batch', type=int, default=1, help='Example batch size')
    parser.add_argument('--height', type=int, default=1792, help='Example height')
    parser.add_argument('--width', type=int, default=1024, help='Example width')
    parser.add_argument('--static', action='store_true', help='Use static input shapes. High performance but restricts input size to the specified --batch, --height, and --width.')
    
    # TensorRT arguments
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision for TensorRT')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 precision for TensorRT (requires --cal)')
    parser.add_argument('--cal', type=str, help='Directory containing calibration samples (images or .npy)')
    parser.add_argument('--calib_count', type=int, default=500, help='Number of samples to use for calibration (default: 500)')
    
    args = parser.parse_args()

    if args.int8 and not args.cal:
        parser.error("--int8 requires --cal providing calibration data directory.")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Determine paths
    engine_path = args.output
    onnx_path = os.path.splitext(engine_path)[0] + '.onnx'

    print(f"[Info] Architecture select: {args.arch}")
    if args.arch == 'unet':
        from UNet import UNetLightningModule
    elif args.arch == 'segformer':
        from SegFormer_StrategyAltered import UNetLightningModule
    else:
        # Fallback if somehow choices mismatch
        from UNet import UNetLightningModule

    print(f"[Info] Loading model: {args.checkpoint}")
    model = UNetLightningModule.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.to('cpu')  # Use CPU for ONNX export stability

    # Create example input
    example = torch.randn(args.batch, 1, args.height, args.width, dtype=torch.float32)

    # Run forward pass once to verify
    with torch.no_grad():
        out = model(example)
        if isinstance(out, tuple):
            if len(out) >= 2:
                logits, probs = out[0], out[1]
            elif len(out) == 1:
                logits = out[0]
                probs = torch.softmax(logits, dim=1)
            else:
                raise RuntimeError("Model forward returned empty tuple")
        else:
            logits = out
            probs = torch.softmax(logits, dim=1)
        print(f"[Check] logits={tuple(logits.shape)}, probs={tuple(probs.shape)}")

    # ONNX Export
    input_names = ['input']
    output_names = ['logits', 'probs']
    dynamic_axes = None
    if not args.static:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'logits': {0: 'batch', 2: 'height', 3: 'width'},
            'probs': {0: 'batch', 2: 'height', 3: 'width'},
        }

    # Ensure output directory exists
    out_dir = os.path.dirname(onnx_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[Export] Exporting to {onnx_path} (opset={args.opset}, dynamic_axes={'ON' if dynamic_axes else 'OFF'})")
    torch.onnx.export(
        model,
        (example,),
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("[Done] ONNX export completed.")

    # TensorRT Build (Always enabled)
    build_tensorrt_engine(
        onnx_path, 
        engine_path, 
        fp16=args.fp16, 
        int8=args.int8,
        calib_dir=args.cal,
        input_shape=(args.batch, 1, args.height, args.width),
        is_static=args.static,
        calib_count=args.calib_count
    )

if __name__ == '__main__':
    main()
