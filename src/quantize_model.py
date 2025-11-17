#!/usr/bin/env python3

import os
import re
import argparse
import time
import torch
import numpy as np
import cv2

from UNet import UNetLightningModule, FITSDataset


def load_lightning_module(checkpoint: str) -> UNetLightningModule:
    model = UNetLightningModule.load_from_checkpoint(checkpoint, strict=False)
    model.eval()
    # 默认用 CPU 做量化前的准备
    model.to('cpu')
    return model


def dynamic_quantize(model: UNetLightningModule) -> UNetLightningModule:
    """对内部的 core SegFormer 模型执行动态量化, 仅线性层。"""
    if not hasattr(model, 'model'):
        raise ValueError("LightningModule 缺少内部 'model' 属性。")
    core = model.model  # SegformerForSemanticSegmentation
    # 只量化 nn.Linear 层; fp16 权重会被先转换为 fp32 再量化
    quantized_core = torch.quantization.quantize_dynamic(
        core,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )
    # 替换回 lightning module
    model.model = quantized_core
    return model


def export_torchscript(model: UNetLightningModule, example_shape=(1,1,512,512), output_path="quantized_model_ts.pt") -> str:
    """尝试导出 TorchScript 版本, 失败则抛出异常。"""
    example = torch.randn(*example_shape, dtype=torch.float32)
    model.cpu()
    with torch.no_grad():
        # 由于 forward 返回 (logits, probs), 用 scripting 保留结构
        scripted = torch.jit.script(model)
        # Ensure parent directory exists before saving
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        scripted.save(output_path)
    return output_path


def save_fallback(model: UNetLightningModule, output_path="quantized_model_fallback.pt") -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


def benchmark(model_before: UNetLightningModule, model_after: UNetLightningModule, batch_size=4, n_warmup=2, n_runs=5, image_shape=(1,512,512)):
    """简单 CPU 基准: 比较量化前后推理耗时。"""
    device = torch.device('cpu')
    model_before.to(device)
    model_after.to(device)
    # 构造输入
    x = torch.randn(batch_size, *image_shape, dtype=torch.float32, device=device)

    def run(m):
        with torch.no_grad():
            logits, probs = m(x)
            return probs  # 仅检查成功返回

    # 预热
    for _ in range(n_warmup):
        _ = run(model_before)
        _ = run(model_after)

    def time_model(m):
        start = time.time()
        for _ in range(n_runs):
            _ = run(m)
        end = time.time()
        return (end - start) / n_runs

    t_before = time_model(model_before)
    t_after = time_model(model_after)

    print(f"[Benchmark] batch_size={batch_size}, image_shape={image_shape}")
    print(f"  未量化平均耗时: {t_before*1000:.2f} ms/forward")
    print(f"  动态量化平均耗时: {t_after*1000:.2f} ms/forward")
    speedup = t_before / t_after if t_after > 0 else float('inf')
    print(f"  推理加速倍率: {speedup:.2f}x")


def build_tensorrt_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    calib_dir: str | None = None,
    calib_size: int = 1024,
    batch_size: int = 8,
    input_shape=(1, 512, 512),
    max_workspace=1 << 30,
    use_fp16: bool = False,
    force_rebuild: bool = False,
):
    """
    使用 TensorRT 将 ONNX 转为 serialized engine，并在 INT8 模式下使用校准数据进行量化。

    说明:
    - 依赖: tensorrt, pycuda
    - calib_dir: 指向训练集中用于校准的 image 目录（FITS 文件），脚本会选取其中一部分样本用于校准。
    - 如果已存在 engine_path 且 force_rebuild=False，则直接返回该路径。
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except Exception as e:
        raise RuntimeError("TensorRT / pycuda 未安装或无法导入: %s" % e)

    # Ensure target directory exists for engine and related artifacts
    engine_parent = os.path.dirname(engine_path)
    if engine_parent:
        os.makedirs(engine_parent, exist_ok=True)

    if os.path.exists(engine_path) and not force_rebuild:
        print(f"[TRT] 已存在 engine: {engine_path}, 跳过重建")
        return engine_path

    # Use VERBOSE logger when doing INT8 calibration to capture more detail for missing-scale warnings
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if calib_dir is not None else trt.Logger.INFO)

    # 简单的校准数据流：从 FITS 目录读取图像，归一化并 resize 到 input_shape
    def iter_calib_batches(calib_dir, batch_size, max_samples):
        # 使用随机抽样的代表性校准集，并跳过无法读取的文件以提高健壮性
        import random
        # 尝试使用 FITSDataset 的读取函数以保证与训练一致的预处理
        files = []
        if calib_dir and os.path.isdir(calib_dir):
            for f in os.listdir(calib_dir):
                if f.lower().endswith('.fits'):
                    files.append(os.path.join(calib_dir, f))
        if len(files) == 0:
            raise FileNotFoundError(f"未找到校准 FITS 文件于: {calib_dir}")

        # 随机抽样 max_samples 个文件（或少于总数）以避免偏序列
        n = min(len(files), max_samples)
        sampled = random.sample(files, n)

        batch = []
        for path in sampled:
            try:
                img = FITSDataset.load_fits_image(path)
                img = FITSDataset.normalize_image_mean_std(img, k=5.0)
                # resize to expected input (input_shape is (C,H,W) or (1,H,W))
                h = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
                w = input_shape[2] if len(input_shape) >= 3 else input_shape[1]
                img_resized = cv2.resize(img, (w, h))
                # model expects (1,H,W)
                if img_resized.ndim == 2:
                    img_resized = np.expand_dims(img_resized, axis=0)
                img_resized = img_resized.astype(np.float32)
                batch.append(img_resized)
                if len(batch) == batch_size:
                    arr = np.stack(batch, axis=0)
                    yield arr
                    batch = []
            except Exception as e:
                print(f"[TRT][Calib] 跳过校准样本 {path} 因为读取失败: {e}")
                continue

        if len(batch) > 0:
            arr = np.stack(batch, axis=0)
            yield arr

    # 实现 TensorRT 的 Python 校准器
    class NumpyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, calib_iter, cache_file="trt_calib.cache"):
            super().__init__()
            self.cache_file = cache_file
            self._calib_iter = iter(calib_iter)
            self.device_input = None

        def get_batch_size(self):
            # 由外部传入 batch_size，通过绑定时确定
            return batch_size

        def get_batch(self, names):
            try:
                batch = next(self._calib_iter)
            except StopIteration:
                return None

            # batch shape: (B, C, H, W)
            # 确保 contiguous
            batch = np.ascontiguousarray(batch)
            # 在第一次调用时分配 GPU buffer
            import pycuda.driver as cuda
            if self.device_input is None:
                self.device_input = cuda.mem_alloc(batch.nbytes)
            cuda.memcpy_htod(self.device_input, batch.tobytes())
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            cache_parent = os.path.dirname(self.cache_file)
            if cache_parent:
                os.makedirs(cache_parent, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

    # 读取 ONNX 并构建 engine
    with open(onnx_path, 'rb') as f:
        onnx_data = f.read()

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_data):
        error_msgs = []
        for i in range(parser.num_errors):
            error_msgs.append(str(parser.get_error(i)))
        raise RuntimeError("ONNX 解析失败: " + "\n".join(error_msgs))

    config = builder.create_builder_config()
    # Set workspace size: different TRT Python bindings expose different APIs.
    if hasattr(config, 'max_workspace_size'):
        try:
            config.max_workspace_size = max_workspace
        except Exception:
            print("[TRT] Warning: 无法设置 config.max_workspace_size (continuing with defaults)")
    elif hasattr(config, 'set_memory_pool_limit') and hasattr(trt, 'MemoryPoolType'):
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace)
        except Exception:
            print("[TRT] Warning: 无法通过 set_memory_pool_limit 设置 workspace 大小 (continuing with defaults)")
    elif hasattr(builder, 'max_workspace_size'):
        try:
            builder.max_workspace_size = max_workspace
        except Exception:
            print("[TRT] Warning: 无法设置 builder.max_workspace_size (continuing with defaults)")
    else:
        print("[TRT] Warning: 未找到可设置 workspace size 的 API，继续使用默认 workspace")
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 处理动态形状：如果 ONNX 模型含有动态维度，需要在 builder/config 中添加 optimization profile
    try:
        # 仅当 network 包含动态维度时才尝试添加 profile
        need_profile = False
        for i in range(network.num_inputs):
            dims = network.get_input(i).shape
            if any(d <= 0 for d in dims):
                need_profile = True
                break
        if need_profile and hasattr(builder, 'create_optimization_profile'):
            profile = builder.create_optimization_profile()
            # 对每个输入构造 min/opt/max 三元组；input_shape 参数被视为 (C,H,W) 或 (1,H,W)
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                name = inp.name
                # Compose shapes with explicit batch dim for EXPLICIT_BATCH
                try:
                    # Ensure input_shape is a sequence of length 3 (C,H,W)
                    base = tuple(input_shape) if len(input_shape) == 3 else tuple(input_shape)
                except Exception:
                    base = tuple(input_shape)
                min_shape = (1,) + base
                opt_shape = (max(1, batch_size),) + base
                max_shape = (max(1, batch_size * 2),) + base
                try:
                    profile.set_shape(name, min_shape, opt_shape, max_shape)
                except Exception:
                    # some bindings expect trt.Dims objects or other APIs; fall back to tuple
                    profile.set_shape(name, list(min_shape), list(opt_shape), list(max_shape))
            try:
                config.add_optimization_profile(profile)
            except Exception:
                # older bindings may require adding profile via builder, ignore and continue
                print('[TRT] Warning: 无法通过 config.add_optimization_profile 添加 optimization profile (binding 不同，继续尝试)')
    except Exception:
        # 如果任何步骤失败，继续构建并让 TensorRT 报出更具体的错误
        pass

    # 设置 INT8 校准
    if calib_dir is not None:
        calib_iter = iter_calib_batches(calib_dir, batch_size=batch_size, max_samples=calib_size)
        calibrator = NumpyEntropyCalibrator(calib_iter, cache_file=engine_path + '.calib')
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    print(f"[TRT] 开始构建 engine (FP16={use_fp16}, INT8={calib_dir is not None}) ...")
    engine = None
    # Different TensorRT Python bindings expose different builder APIs.
    # Try builder.build_engine first, then fallback to build_serialized_network + runtime.deserialize_cuda_engine.
    try:
        # Capture TRT stderr output so we can analyze "Missing scale" warnings
        import io, contextlib
        log_buf = io.StringIO()
        with contextlib.redirect_stderr(log_buf):
            if hasattr(builder, 'build_engine'):
                engine = builder.build_engine(network, config)
            else:
                raise AttributeError('build_engine not available')
    except Exception:
        try:
            # build_serialized_network returns serialized engine bytes in newer bindings
            if hasattr(builder, 'build_serialized_network'):
                import io, contextlib
                log_buf = io.StringIO()
                with contextlib.redirect_stderr(log_buf):
                    serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError('builder.build_serialized_network returned None')
                runtime = trt.Runtime(TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(serialized)
            else:
                raise RuntimeError('No supported builder API (build_engine/build_serialized_network) available')
        except Exception as e:
            # 保存构建日志（若有）以便排查
            try:
                log_text = log_buf.getvalue() if 'log_buf' in locals() else ''
                open(engine_path + '.trtlog', 'w', encoding='utf-8').write(log_text)
            except Exception:
                pass
            raise RuntimeError("TensorRT 构建 engine 失败: %s" % e)
    if engine is None:
        raise RuntimeError("TensorRT 构建 engine 失败 (engine is None)")

    # 若存在构建日志，保存并统计 Missing scale 警告条数，帮助决策是否使用 FP16 回退
    try:
        log_text = log_buf.getvalue() if 'log_buf' in locals() else ''
        if log_text:
            open(engine_path + '.trtlog', 'w', encoding='utf-8').write(log_text)
            missing_count = log_text.count('Missing scale') + log_text.count('Missing scale and zero-point')
            if missing_count > 0:
                print(f"[TRT] 构建日志已保存: {engine_path}.trtlog (Missing scale warnings: {missing_count})")
                if missing_count > 50:
                    print("[TRT] 注意: 检测到大量 Missing scale 警告，建议使用 FP16 或进行 QAT 来提高可量化层覆盖率。")
    except Exception:
        pass

    # 序列化并保存
    # Ensure parent dir exists (defensive, created earlier but be safe)
    eng_parent = os.path.dirname(engine_path)
    if eng_parent:
        os.makedirs(eng_parent, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"[TRT] Engine 已保存: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="对 Lightning SegFormer 模型执行动态量化与 TensorRT INT8 导出")
    # 使 checkpoint 可选：如果用户只想从 ONNX 构建 TensorRT engine（--int8），则无需提供 checkpoint
    parser.add_argument('--checkpoint', required=False, default=None, help='Lightning .ckpt 文件路径（可选，若进行动态量化/导出/TorchScript 则需要）')
    parser.add_argument('--output', default='quantized_segformer.pt', help='输出 TorchScript 文件名')
    parser.add_argument('--fallback-output', default='quantized_segformer_fallback.pt', help='备用 state_dict 文件名')
    parser.add_argument('--no-script', action='store_true', help='仅保存 state_dict, 不导出 TorchScript')
    parser.add_argument('--benchmark', action='store_true', help='执行简单 CPU 推理基准对比')
    parser.add_argument('--batch-size', type=int, default=4, help='基准的 batch size')
    # ONNX -> TensorRT 相关参数
    parser.add_argument('--onnx', default='model_segformer.onnx', help='已导出的 ONNX 模型路径 (用于 TensorRT 转换)')
    parser.add_argument('--trt-engine', default='model_trt_int8.engine', help='输出的 TensorRT engine 文件路径')
    parser.add_argument('--int8', action='store_true', help='启用 TensorRT INT8 构建（需要 --onnx）')
    parser.add_argument('--calib-dir', default='/home/cbm/deRFI/Datasets/SynthesizedDataset/image/train', help='用于 INT8 校准的 FITS 训练集目录（只会使用部分样本）')
    parser.add_argument('--calib-size', type=int, default=512, help='用于校准的样本数上限（从 calib-dir 中选择）')
    parser.add_argument('--trt-batch-size', type=int, default=8, help='校准时使用的 batch size (TensorRT)')
    parser.add_argument('--trt-fp16', action='store_true', help='同时启用 FP16 优化（若硬件支持）')
    parser.add_argument('--force-rebuild', action='store_true', help='若已存在 engine 则强制重建')
    args = parser.parse_args()

    ckpt = args.checkpoint

    # 当用户没有提供 checkpoint 且仅做 ONNX->TensorRT 时，跳过加载/量化步骤
    model_before = None
    model_after = None
    if ckpt is None:
        if not args.int8:
            parser.error("--checkpoint 为必需，除非使用 --int8 从已有 ONNX 构建 TensorRT engine。\n" \
                         "若你只想用 ONNX 构建 engine，请使用 --int8 并确保 --onnx 指向有效文件。")
        else:
            print("[Info] 未提供 checkpoint；跳过动态量化/导出流程，仅执行 ONNX -> TensorRT 构建（若指定 --int8）。")
    else:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"未找到 checkpoint: {ckpt}")

        print(f"[Info] 加载模型: {ckpt}")
        model_before = load_lightning_module(ckpt)

        print("[Info] 执行动态量化 (Linear -> INT8)...")
        model_after = dynamic_quantize(load_lightning_module(ckpt))

    # 保存 TorchScript
    saved_paths = []
    if model_after is not None:
        if not args.no_script:
            try:
                ts_path = export_torchscript(model_after, output_path=args.output)
                print(f"[Save] TorchScript 已保存: {ts_path}")
                saved_paths.append(ts_path)
            except Exception as e:
                print(f"[Warn] TorchScript 导出失败: {e}")
        # 始终保存一个 fallback state_dict
        fallback_path = save_fallback(model_after, output_path=args.fallback_output)
        print(f"[Save] 备用 state_dict 已保存: {fallback_path}")
        saved_paths.append(fallback_path)
    else:
        # 当用户只请求 ONNX->TensorRT (--int8) 时，不输出未执行量化的额外信息以减少噪声
        if not args.int8:
            print("[Info] 未执行动态量化/导出（因为未提供 checkpoint）; 未生成 TorchScript 或 fallback state_dict。")

    # 基准对比
    if args.benchmark:
        if model_before is None or model_after is None:
            parser.error("要运行 --benchmark，必须提供 --checkpoint 并能加载模型以进行量化对比。")
        print("[Benchmark] 开始...")
        benchmark(model_before, model_after, batch_size=args.batch_size)

    # TensorRT INT8 构建（可选）
    if args.int8:
        onnx_path = args.onnx
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"未找到 ONNX 模型: {onnx_path}")
        requested_path = args.trt_engine
        # Ensure clear filenames: create explicit int8 and fp16 variant names
        def ensure_tag(path, tag):
            base, ext = os.path.splitext(path)
            if base.endswith(f"_{tag}"):
                return path
            # remove any existing _int8/_fp16 tag to avoid duplication
            base = re.sub(r"_(int8|fp16)$", "", base)
            if ext == '':
                return f"{base}_{tag}.engine"
            return f"{base}_{tag}{ext}"

        # compute explicit engine paths
        engine_path_int8 = ensure_tag(requested_path, 'int8')
        engine_path_fp16 = ensure_tag(requested_path, 'fp16')

        print(f"[TRT] 使用 ONNX={onnx_path} 进行 TensorRT INT8 构建，输出={engine_path_int8}")
        try:
            build_tensorrt_engine_from_onnx(
                onnx_path=onnx_path,
                engine_path=engine_path_int8,
                calib_dir=args.calib_dir,
                calib_size=args.calib_size,
                batch_size=args.trt_batch_size,
                input_shape=(1, 512, 512),
                use_fp16=args.trt_fp16,
                force_rebuild=args.force_rebuild,
            )
        except Exception as e:
            print(f"[TRT] 构建失败: {e}")

        # 同时为对比构建一个 FP16-only 引擎（不做 INT8 校准），并保存为独立文件
        try:
            def make_variant_path(base_path, tag):
                base, ext = os.path.splitext(base_path)
                if ext == '':
                    return f"{base}_{tag}.engine"
                return f"{base}_{tag}{ext}"

            fp16_engine = engine_path_fp16
            # 如果用户不强制重建且文件已存在，则跳过
            if not os.path.exists(fp16_engine) or args.force_rebuild:
                print(f"[TRT] 也构建 FP16 引擎: {fp16_engine}")
                build_tensorrt_engine_from_onnx(
                    onnx_path=onnx_path,
                    engine_path=fp16_engine,
                    calib_dir=None,
                    calib_size=0,
                    batch_size=args.trt_batch_size,
                    input_shape=(1, 512, 512),
                    use_fp16=True,
                    force_rebuild=args.force_rebuild,
                )
            else:
                print(f"[TRT] FP16 引擎已存在且未强制重建，跳过: {fp16_engine}")
            # 把引擎路径加入输出列表
            saved_paths.append(engine_path_int8)
            saved_paths.append(fp16_engine)
        except Exception as e:
            print(f"[TRT] 同时构建 FP16 引擎时发生错误: {e}")

    print("[Done] 量化完成。输出文件:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == '__main__':
    main()
