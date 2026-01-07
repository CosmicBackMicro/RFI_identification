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

def build_tensorrt_engine(onnx_path, engine_path, fp16=True, input_shape=(1, 1, 512, 512), is_static=False):
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

    # Set memory pool limit for workspace (e.g., 4GB for large models)
    memory_pool_type = getattr(trt, "MemoryPoolType", None)
    workspace_size = 4096 * 1024 * 1024 # 4GB
    if hasattr(config, "set_memory_pool_limit") and memory_pool_type:
        config.set_memory_pool_limit(getattr(memory_pool_type, "WORKSPACE"), workspace_size)
        print(f"[TRT] Workspace limit set to {workspace_size/(1024**2):.0f}MB.")
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_size
        print(f"[TRT] Workspace limit set to {workspace_size/(1024**2):.0f}MB.")

    # Tactic Sources: Try to disable CUBLAS_LT if it causes crashes due to library mismatch
    tactic_source_enum = getattr(trt, "TacticSource", None)
    if tactic_source_enum and hasattr(config, "set_tactic_sources"):
        try:
            # Use getattr to safely get enum values
            cublas = getattr(tactic_source_enum, "CUBLAS")
            cudnn = getattr(tactic_source_enum, "CUDNN")
            mask = (1 << int(cublas)) | (1 << int(cudnn))
            config.set_tactic_sources(mask)
            print("[TRT] Tactic Sources: Restricted to CUBLAS and CUDNN (disabled CUBLAS_LT for stability).")
        except Exception as e:
            print(f"[TRT] Could not restrict Tactic Sources: {e}")

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
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version (default: 11 for compatibility)')
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
        input_shape=(args.batch, 1, args.height, args.width),
        is_static=args.static
    )

if __name__ == '__main__':
    main()
