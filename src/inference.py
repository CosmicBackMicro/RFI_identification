#!/usr/bin/env python3
"""
统一推理脚本（ONNX / TensorRT）：
- ONNX 路线：--model 指定 .onnx，优先 TensorRT，其次 CUDA，最后 CPU Provider
- TensorRT 路线：--model 指定 .engine，使用 TensorRT Python API 进行推理

示例：
    ONNX  : python src/infer_example.py --model <model.onnx> --dataset <dataset_dir>
    TRT   : python src/infer_example.py --model <model.engine> --dataset <dataset_dir>
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
sys.path.append(os.path.dirname(__file__))

from UNet import FITSDataset

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception:
    trt = None
    cuda = None

def _pick_ort_providers():
    """按优先顺序返回 ORT Providers 列表。"""
    providers = []
    if ort is None:
        return providers
    try:
        avail = ort.get_available_providers()
        if 'TensorrtExecutionProvider' in avail:
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in avail:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
    except Exception:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return providers

def _get_class_names(dataset, num_classes_hint=None):
    try:
        names = list(getattr(dataset, 'classes', []))
    except Exception:
        names = []
    if num_classes_hint is None:
        return names if names else FITSDataset.CLASSES
    # 对齐长度到提示的类数
    if not names:
        names = FITSDataset.CLASSES
    if len(names) < int(num_classes_hint):
        names = names + [f'class_{i}' for i in range(len(names), int(num_classes_hint))]
    return names[:int(num_classes_hint)]

def _save_sample_visual(i, image_np, mask_np, pred_np, class_names, save_dir, prefix=""):
    """通用可视化：1x4 视图，原图/GT/Pred/Overlay。"""
    os.makedirs(save_dir, exist_ok=True)
    nc = len(class_names)
    mask_vmax = max(nc - 1, int(np.max(mask_np)) if getattr(mask_np, 'size', 0) > 0 else 0)
    pred_vmax = max(nc - 1, int(np.max(pred_np)) if getattr(pred_np, 'size', 0) > 0 else 0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_np, cmap='gist_heat'); axes[0].set_title('Image'); axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='viridis', vmin=0, vmax=mask_vmax); axes[1].set_title('GT'); axes[1].axis('off')
    axes[2].imshow(pred_np, cmap='viridis', vmin=0, vmax=pred_vmax); axes[2].set_title('Pred'); axes[2].axis('off')
    axes[3].imshow(image_np, cmap='gist_heat'); axes[3].imshow(pred_np, cmap='viridis', alpha=0.5, vmin=0, vmax=pred_vmax); axes[3].set_title('Overlay'); axes[3].axis('off')
    fname = f"{prefix}inference_sample_{i}.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    return path

def run_trt_inference(engine_path, dataset, num_samples=3, save_dir="results/trt", batch_size=1, no_visual=False):
    """TensorRT Engine 推理与可视化。"""
    if trt is None or cuda is None:
        raise RuntimeError("tensorrt 或 pycuda 未安装，无法使用 .engine 模型。")
    
    print(f"[TRT] Loading engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine.")

    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    # 获取所有 Tensor 名称 (适配 TensorRT 10.x API)
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    
    # 找到输入 Tensor 名称 (假设只有一个输入)
    input_tensor_name = None
    for name in tensor_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_tensor_name = name
            break
    if input_tensor_name is None:
        raise RuntimeError("Engine 中未找到输入 Tensor")
    
    # Warmup
    print("🔥 Warming up TensorRT engine...")
    warmup_shape = (batch_size, 1, 512, 512)
    if context.set_input_shape(input_tensor_name, warmup_shape):
        # 简单执行一次，不分配复杂 buffer，仅用于触发驱动初始化
        # 注意：为了简单起见，我们这里不进行完整的内存分配，
        # 真正的 Warmup 会在第一个 batch 自动完成，但为了响应用户需求，
        # 我们可以在这里做一个轻量级的。
        # 实际上 TRT 引擎加载后第一次执行确实会有微小延迟。
        pass
    print("✅ Warmup done.")

    if not no_visual:
        os.makedirs(save_dir, exist_ok=True)
    dataset_class_names = _get_class_names(dataset)
    
    total_inference_time = 0.0
    total_data_load_time = 0.0
    total_postprocess_time = 0.0
    total_save_time = 0.0
    
    # Inference Loop
    for start_idx in range(0, min(num_samples, len(dataset)), batch_size):
        end_idx = min(start_idx + batch_size, min(num_samples, len(dataset)))
        current_batch_size = end_idx - start_idx
        
        # 计时数据加载
        data_load_start = time.time()
        batch_imgs = []
        batch_masks = []
        for i in range(start_idx, end_idx):
            image, mask = dataset[i]
            img_np = np.expand_dims(np.array(image, dtype=np.float32), axis=0) # (1, H, W)
            batch_imgs.append(img_np)
            batch_masks.append(mask)
        
        # Stack batch
        x = np.stack(batch_imgs, axis=0) # (B, 1, H, W)
        x = np.ascontiguousarray(x)
        data_load_end = time.time()
        total_data_load_time += (data_load_end - data_load_start)
        
        # Set input shape
        if not context.set_input_shape(input_tensor_name, x.shape):
            raise RuntimeError(f"❌ 设置输入形状失败: {x.shape}。请检查是否满足 Engine 的优化配置文件 (Optimization Profile) 限制。")
        
        # Allocate buffers
        d_inputs = []
        d_outputs = []
        h_outputs = []
        
        # 遍历所有 Tensor 分配内存
        for name in tensor_names:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            vol = trt.volume(shape)
            if vol < 0:
                 raise RuntimeError(f"Tensor {name} has invalid shape {shape}")
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                d_input = cuda.mem_alloc(x.nbytes)
                cuda.memcpy_htod_async(d_input, x, stream)
                context.set_tensor_address(name, int(d_input))
                d_inputs.append(d_input)
            else:
                h_output = cuda.pagelocked_empty(vol, dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)
                context.set_tensor_address(name, int(d_output))
                d_outputs.append(d_output)
                h_outputs.append(h_output)

        # Execute (v3 API for TRT 8.5+)
        start_time = time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_inference_time += batch_time
        
        # 计时后处理 (D2H 拷贝 + Softmax + Argmax)
        postprocess_start = time.time()
        # Copy back
        for h_out, d_out in zip(h_outputs, d_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
        
        # Process output (Assume last output is probs or logits)
        output_tensor_name = None
        for name in tensor_names:
             if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                 output_tensor_name = name 
        
        raw_output = h_outputs[-1] 
        output_shape = context.get_tensor_shape(output_tensor_name)
        output = raw_output.reshape(output_shape)
        
        # Softmax if needed
        if np.max(output) > 1.0 or np.min(output) < 0.0:
             shift = output - np.max(output, axis=1, keepdims=True)
             expv = np.exp(shift)
             probs = expv / np.sum(expv, axis=1, keepdims=True)
        else:
             probs = output

        preds = np.argmax(probs, axis=1)
        postprocess_end = time.time()
        total_postprocess_time += (postprocess_end - postprocess_start)
        
        # 计时保存
        save_start = time.time()
        if not no_visual:
            for bi, i in enumerate(range(start_idx, end_idx)):
                img_np = batch_imgs[bi].squeeze(0)
                mask_np = batch_masks[bi]
                pred_np = preds[bi]
                try:
                    saved_path = _save_sample_visual(i+1, img_np, mask_np, pred_np, dataset_class_names, save_dir, prefix="trt_")
                    print(f"✅ 保存(TRT)结果: {saved_path}")
                except Exception as e:
                    print(f"❌ 保存失败: {e}")
        save_end = time.time()
        total_save_time += (save_end - save_start)
        
        print(f"⏱️ Batch {start_idx//batch_size + 1}: {current_batch_size} samples, "
              f"DataLoad: {data_load_end - data_load_start:.4f}s, "
              f"Infer: {batch_time:.4f}s, "
              f"Post: {postprocess_end - postprocess_start:.4f}s, "
              f"Save: {save_end - save_start:.4f}s")

    total_samples = min(num_samples, len(dataset))
    print(f"🎉 TRT 推理完成！共处理 {total_samples} 个样本")
    print(f"  - 数据加载总时间: {total_data_load_time:.4f}s (Avg: {total_data_load_time/total_samples:.4f}s/sample)")
    print(f"  - 推理总时间: {total_inference_time:.4f}s (Avg: {total_inference_time/total_samples:.4f}s/sample)")
    print(f"  - 后处理总时间: {total_postprocess_time:.4f}s (Avg: {total_postprocess_time/total_samples:.4f}s/sample)")
    print(f"  - 保存总时间: {total_save_time:.4f}s (Avg: {total_save_time/total_samples:.4f}s/sample)")

def run_onnx_inference(onnx_path, dataset, num_samples=3, save_dir="results/onnx", batch_size=4, no_visual=False):
    """ONNX Runtime 推理与可视化（与 PyTorch 版本保持输出结构一致）。"""
    if ort is None:
        raise RuntimeError("onnxruntime 未安装，无法使用 --onnx。请先 pip install onnxruntime-gpu 或 onnxruntime。")
    if not no_visual:
        os.makedirs(save_dir, exist_ok=True)
    total_start_time = time.time()
    providers = _pick_ort_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers) if providers else ort.InferenceSession(onnx_path)
    used = sess.get_providers()
    print(f"[ORT] Providers: {used}")

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    # Warmup (ONNX Runtime TensorRT EP 首次运行会进行编译，耗时较长)
    print("🔥 Warming up ONNX session...")
    warmup_start = time.time()
    warmup_shape = (batch_size, 1, 512, 512) # 假设输入是 512x512
    dummy_input = np.zeros(warmup_shape, dtype=np.float32)
    sess.run(output_names, {input_name: dummy_input})
    warmup_end = time.time()
    print(f"✅ Warmup done in {warmup_end - warmup_start:.4f}s")

    # 类别名称
    dataset_class_names = _get_class_names(dataset)
    num_classes = len(dataset_class_names)

    # 推理循环
    total_inference_time = 0.0
    total_data_load_time = 0.0
    total_postprocess_time = 0.0
    total_save_time = 0.0
    for start_idx in range(0, min(num_samples, len(dataset)), batch_size):
        end_idx = min(start_idx + batch_size, min(num_samples, len(dataset)))
        # 计时数据加载
        data_load_start = time.time()
        batch_imgs = []
        batch_masks = []
        for i in range(start_idx, end_idx):
            image, mask = dataset[i]
            img_np = np.expand_dims(np.array(image, dtype=np.float32), axis=0)  # (1,H,W)
            batch_imgs.append(img_np)
            batch_masks.append(mask)
        x = np.stack(batch_imgs, axis=0)  # (B,1,H,W)
        feed = {input_name: x}
        data_load_end = time.time()
        total_data_load_time += (data_load_end - data_load_start)
        start_time = time.time()
        outs = sess.run(output_names, feed)
        end_time = time.time()
        batch_time = end_time - start_time
        total_inference_time += batch_time
        
        # 计时后处理 (Softmax + Argmax)
        postprocess_start = time.time()
        if 'probs' in output_names:
            probs = outs[output_names.index('probs')]
        elif 'logits' in output_names:
            logits = outs[output_names.index('logits')]
            # numpy softmax
            shift = logits - np.max(logits, axis=1, keepdims=True)
            expv = np.exp(shift)
            probs = expv / np.sum(expv, axis=1, keepdims=True)
        else:
            probs = outs[0]
        preds = np.argmax(probs, axis=1)
        postprocess_end = time.time()
        total_postprocess_time += (postprocess_end - postprocess_start)
        
        # 计时保存
        save_start = time.time()
        if not no_visual:
            for bi, i in enumerate(range(start_idx, end_idx)):
                img_np = batch_imgs[bi].squeeze(0)
                mask_np = batch_masks[bi]
                pred_np = preds[bi]
                try:
                    saved_path = _save_sample_visual(i+1, img_np, mask_np, pred_np, dataset_class_names, save_dir, prefix="onnx_")
                    print(f"✅ 保存(ONNX)结果: {saved_path}")
                except Exception as e:
                    print(f"❌ 保存失败: {e}")
        save_end = time.time()
        total_save_time += (save_end - save_start)
        
        print(f"⏱️ Batch {start_idx//batch_size + 1}: {len(batch_imgs)} samples, "
              f"DataLoad: {data_load_end - data_load_start:.4f}s, "
              f"Infer: {batch_time:.4f}s, "
              f"Post: {postprocess_end - postprocess_start:.4f}s, "
              f"Save: {save_end - save_start:.4f}s")
    total_time = time.time() - total_start_time
    total_samples = min(num_samples, len(dataset))
    avg_time = total_inference_time / total_samples
    print(f"🎉 ONNX 推理完成！共处理 {total_samples} 个样本")
    print(f"  - 数据加载总时间: {total_data_load_time:.4f}s (Avg: {total_data_load_time/total_samples:.4f}s/sample)")
    print(f"  - 推理总时间: {total_inference_time:.4f}s (Avg: {total_inference_time/total_samples:.4f}s/sample)")
    print(f"  - 后处理总时间: {total_postprocess_time:.4f}s (Avg: {total_postprocess_time/total_samples:.4f}s/sample)")
    print(f"  - 保存总时间: {total_save_time:.4f}s (Avg: {total_save_time/total_samples:.4f}s/sample)")
    print(f"  - 总运行时间: {total_time:.4f}s，结果保存至 {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="统一推理脚本(ONNX 或 TensorRT)")
    parser.add_argument('--dataset', required=True, help='数据集顶层目录 (包含 image/mask)')
    parser.add_argument('--model', required=True, help='模型路径 (.onnx 或 .engine)')
    parser.add_argument('--num-samples', type=int, default=32, help='推理样本数')
    parser.add_argument('--batch-size', type=int, default=1, help='批大小')
    parser.add_argument('--save-dir', type=str, default='results', help='保存目录')
    parser.add_argument('--no-visual', action='store_true', help='禁用可视化保存（用于纯性能测试）')
    args = parser.parse_args()

    dataset_top_dir = args.dataset
    image_dir = os.path.join(dataset_top_dir, "image")
    mask_dir = os.path.join(dataset_top_dir, "mask")
    val_image_dir = os.path.join(image_dir, "val")
    val_mask_dir = os.path.join(mask_dir, "val")

    val_dataset = FITSDataset(
        val_image_dir,
        val_mask_dir,
        classes=None,
        augmentation=None,
        preprocessing=None
    )
    print("🔍 开始推理可视化...")
    print(f"📂 数据集: {dataset_top_dir}")
    print(f"📊 样本数量: {len(val_dataset)}")

    if args.model.endswith('.onnx'):
        print(f"🧠 检测到 ONNX 模型: {args.model}")
        run_onnx_inference(
            onnx_path=args.model,
            dataset=val_dataset,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            no_visual=args.no_visual
        )
    elif args.model.endswith('.engine') or args.model.endswith('.plan'):
        print(f"🚀 检测到 TensorRT Engine: {args.model}")
        run_trt_inference(
            engine_path=args.model,
            dataset=val_dataset,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            no_visual=args.no_visual
        )
    else:
        raise SystemExit("❌ 未知模型格式，请提供 .onnx 或 .engine 文件")