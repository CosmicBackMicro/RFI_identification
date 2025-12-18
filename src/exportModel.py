#!/usr/bin/env python3
"""
Export Lightning SegFormer model to ONNX and optionally TensorRT.

Features:
- Load model from Lightning .ckpt.
- Export with two outputs: logits and probs (after Softmax).
- Default dynamic axes (batch, height, width).
- Default opset 17.
- Optional TensorRT engine build.

Usage:
python src/export_onnx.py \
  --checkpoint checkpoints/best_model.ckpt \
  --output model.engine \
  --height 512 --width 512 --batch 1 \
  --fp16
"""
import argparse
import os
import torch
import sys

from UNet import UNetLightningModule

def build_tensorrt_engine(onnx_path, engine_path, fp16=True, input_shape=(1, 1, 512, 512)):
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
    
    # EXPLICIT_BATCH is required for ONNX models with dynamic shapes
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

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
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 mode enabled.")
        else:
            print("[TRT] FP16 requested but not supported by platform. Falling back to FP32.")

    # Optimization Profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    b, c, h, w = input_shape
    # Define min, opt, max shapes
    # Min: 1x1x(h/2)x(w/2) - heuristic
    min_shape = (1, c, h // 2, w // 2)
    opt_shape = (b, c, h, w)
    # Max: Set equal to opt_shape to save memory on Laptop GPUs (avoid OOM)
    # If you need larger inputs, increase --height/--width or manually adjust here
    max_shape = (b, c, h, w)
    
    print(f"[TRT] Optimization Profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[TRT] Engine build failed.")
        return

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"[TRT] Engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Lightning SegFormer model to ONNX/TensorRT")
    parser.add_argument('--checkpoint', required=True, help='Path to Lightning .ckpt file')
    default_output = os.path.join('checkpoints', 'tensorrt', 'model.engine')
    parser.add_argument('--output', required=False, default=default_output, help=f'Output TensorRT engine path (default: {default_output}). ONNX will be saved with same basename.')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--batch', type=int, default=1, help='Example batch size')
    parser.add_argument('--height', type=int, default=512, help='Example height')
    parser.add_argument('--width', type=int, default=512, help='Example width')
    parser.add_argument('--static', action='store_true', help='Use static shape (disable dynamic axes)')
    
    # TensorRT arguments
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision for TensorRT')
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Determine paths
    engine_path = args.output
    onnx_path = os.path.splitext(engine_path)[0] + '.onnx'

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
        input_shape=(args.batch, 1, args.height, args.width)
    )

if __name__ == '__main__':
    main()
