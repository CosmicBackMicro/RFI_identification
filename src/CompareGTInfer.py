#!/usr/bin/env python3
"""
TensorRT Inference and Visualization Script

This script performs inference on a FITS dataset using a TensorRT engine 
and generates 4-panel comparison plots (Image, Ground Truth, Prediction, Overlay).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from UNet import FITSDataset

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("❌ Error: tensorrt or pycuda not installed. This script requires a TensorRT environment.")
    sys.exit(1)

def get_class_names(dataset, num_classes_hint=None):
    """Retrieve class names from dataset or defaults."""
    try:
        names = list(getattr(dataset, 'classes', []))
    except Exception:
        names = []
    
    if not names:
        names = FITSDataset.CLASSES
        
    if num_classes_hint is not None and len(names) < int(num_classes_hint):
        names = names + [f'class_{i}' for i in range(len(names), int(num_classes_hint))]
    
    return names if num_classes_hint is None else names[:int(num_classes_hint)]

def calculate_metrics(pred, target, num_classes):
    """Calculate per-sample IoU and Accuracy for comparison."""
    # Accuracy
    correct = np.sum(pred == target)
    acc = correct / target.size
    
    all_ious = []
    fg_ious = []
    for cls in range(num_classes):
        inter = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        
        # If class exists in either pred or target, calculate IoU
        if union > 0:
            iou = inter / union
            all_ious.append(iou)
            if cls > 0:  # Foreground categories (RFI classes)
                fg_ious.append(iou)
    
    miou = np.mean(all_ious) if all_ious else 0.0
    fg_miou = np.mean(fg_ious) if fg_ious else 0.0
    
    return acc, miou, fg_miou

def save_visual_plot(i, image_np, mask_np, pred_np, class_names, save_dir, prefix="trt_"):
    """
    Generate and save a 1x4 comparison subplot.
    Layout: [Input Data] [Ground Truth] [AI Prediction] [Detection Overlay]
    """
    os.makedirs(save_dir, exist_ok=True)
    nc = len(class_names)
    
    # Ensure prediction and mask have same shape for metrics calculation
    if pred_np.shape != mask_np.shape:
        mask_np = cv2.resize(mask_np, (pred_np.shape[1], pred_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        image_np = cv2.resize(image_np, (pred_np.shape[1], pred_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Calculate performance metrics for this specific sample
    acc, miou, fg_miou = calculate_metrics(pred_np, mask_np, nc)
    
    # Performance-aware color limits
    mask_vmax = max(nc - 1, int(np.max(mask_np)) if mask_np.size > 0 else 0)
    pred_vmax = max(nc - 1, int(np.max(pred_np)) if pred_np.size > 0 else 0)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.suptitle(f"Sample #{i:03d} | Comparison: GT vs Prediction\nMetrics: Acc={acc:.2%}, mIoU(All)={miou:.4f}", 
                 fontsize=14, fontweight='bold')
    
    # Panel 0: Original Data
    axes[0].imshow(image_np, cmap='gist_heat')
    axes[0].set_title('Normalized Input Data', fontsize=10)
    axes[0].set_axis_off()
    
    # Panel 1: Training GT
    axes[1].imshow(mask_np, cmap='viridis', vmin=0, vmax=mask_vmax)
    axes[1].set_title('Training Ground Truth (GT)', fontsize=10)
    axes[1].set_axis_off()
    
    # Panel 2: AI Prediction
    axes[2].imshow(pred_np, cmap='viridis', vmin=0, vmax=pred_vmax)
    axes[2].set_title('AI Inference Prediction', fontsize=10)
    axes[2].set_axis_off()
    
    # Panel 3: Overlay
    axes[3].imshow(image_np, cmap='gist_heat')
    axes[3].imshow(pred_np, cmap='viridis', alpha=0.4, vmin=0, vmax=pred_vmax)
    axes[3].set_title('Detection Overlay', fontsize=10)
    axes[3].set_axis_off()
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    output_path = os.path.join(save_dir, f"{prefix}comp_{i:03d}_miou{miou:.3f}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def run_trt_inference(engine_path, dataset, num_samples=32, save_dir="results/inference_plots", batch_size=1):
    """Load TensorRT engine and process dataset samples."""
    print(f"🚀 Loading TensorRT engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")

    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    # Identify all input and output tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    output_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    
    if not input_names or not output_names:
        raise RuntimeError("Engine must have at least one input and one output.")
    
    primary_input = input_names[0]
    # We'll use the first output for predictions, but bind all
    primary_output = output_names[0]

    # Pre-allocate buffers for ALL tensors to satisfy API requirements
    # and improve performance by avoiding per-iteration allocation
    buffers = {}
    for name in tensor_names:
        # Get max shape from profile if dynamic, or fixed shape
        # For simplicity in this script, we assume the provided batch_size and 512x512
        # fits within the engine's profile.
        context.set_input_shape(primary_input, (batch_size, 1, 512, 512))
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        
        host_mem = cuda.pagelocked_empty(tuple(shape), dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        buffers[name] = {
            "host": host_mem,
            "device": device_mem,
            "nbytes": host_mem.nbytes
        }
        # Bind address
        context.set_tensor_address(name, int(device_mem))

    print("🔥 Warming up engine...")
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    print("✅ Warmup complete.")

    class_names = get_class_names(dataset)
    num_to_process = min(num_samples, len(dataset))
    
    for start_idx in range(0, num_to_process, batch_size):
        end_idx = min(start_idx + batch_size, num_to_process)
        current_batch = end_idx - start_idx
        
        batch_imgs, batch_masks = [], []
        for i in range(start_idx, end_idx):
            img, mask = dataset[i]
            # Standardize resolution for TensorRT engine
            img_res = cv2.resize(np.array(img, dtype=np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
            batch_imgs.append(np.expand_dims(img_res, axis=0))
            batch_masks.append(mask)
        
        x = np.ascontiguousarray(np.stack(batch_imgs, axis=0))
        
        # 1. Update input shape if batch is smaller than max
        if current_batch != batch_size:
            context.set_input_shape(primary_input, (current_batch, 1, 512, 512))
            # Rebind addresses because shape change might affect some internal state in some TRT versions
            for name in tensor_names:
                context.set_tensor_address(name, int(buffers[name]["device"]))

        # 2. Upload Input
        cuda.memcpy_htod_async(buffers[primary_input]["device"], x, stream)

        # 3. Execute
        t0 = time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        
        # 4. Download Outputs (All of them, to be safe)
        for name in output_names:
            cuda.memcpy_dtoh_async(buffers[name]["host"], buffers[name]["device"], stream)
        
        stream.synchronize()
        infer_time = time.time() - t0
        
        # 5. Extract results from primary output
        # Get actual output shape for the current batch
        out_shape = context.get_tensor_shape(primary_output)
        output = buffers[primary_output]["host"][:np.prod(out_shape)].reshape(out_shape)
        
        # Argmax over class dimension
        preds = np.argmax(output, axis=1)
        
        for bi, global_idx in enumerate(range(start_idx, end_idx)):
            img_np = batch_imgs[bi].squeeze(0)
            target_mask = batch_masks[bi]
            pred_mask = preds[bi]
            
            saved_path = save_visual_plot(global_idx + 1, img_np, target_mask, pred_mask, class_names, save_dir)
            print(f"📸 Saved: {saved_path} (Infer: {infer_time/current_batch:.4f}s/sample)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified TensorRT Inference & Visualization")
    parser.add_argument('--dataset', required=True, help='Top-level dataset directory')
    parser.add_argument('--model', required=True, help='Path to .engine or .plan file')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--save-dir', type=str, default='results/inference_plots', help='Output directory')
    
    args = parser.parse_args()

    val_img_dir = os.path.join(args.dataset, "image", "val")
    val_mask_dir = os.path.join(args.dataset, "mask", "val")

    if not os.path.exists(val_img_dir):
        print(f"❌ Error: {val_img_dir} does not exist.")
        sys.exit(1)

    dataset = FITSDataset(val_img_dir, val_mask_dir)
    print(f"🔍 Processing {args.num_samples} samples...")
    
    run_trt_inference(
        engine_path=args.model,
        dataset=dataset,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        batch_size=args.batch_size
    )
    print(f"🎉 Results saved to: {args.save_dir}")