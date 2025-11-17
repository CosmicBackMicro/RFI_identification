#!/usr/bin/env python3
"""
导出 Lightning SegFormer 模型为 ONNX，便于在 ONNX Runtime / TensorRT 中进行 GPU 高性能推理。

特性:
- 从 Lightning .ckpt 加载模型，使用其 forward 返回 (logits, probs)。
- 导出包含两个输出: logits 与 probs（Softmax 后概率）。
- 默认启用动态轴 (batch, height, width)。
- opset 默认 17。

用法:
python src/export_onnx.py \
  --checkpoint /home/cbm/deRFI/checkpoints/best_model-epoch=09-fgF1_val_fg_macro_f1=0.7595.ckpt \
  --output /home/cbm/deRFI/model_segformer.onnx \
  --height 512 --width 512 --batch 1

"""
import argparse
import os
import torch

from UNet import UNetLightningModule


def main():
    parser = argparse.ArgumentParser(description="导出 Lightning SegFormer 模型为 ONNX")
    parser.add_argument('--checkpoint', required=True, help='Lightning .ckpt 文件路径')
    default_output = os.path.join('checkpoints', 'onnx', 'model_segformer.onnx')
    parser.add_argument('--output', required=False, default=default_output, help=f'输出 ONNX 文件路径 (default: {default_output})')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset 版本')
    parser.add_argument('--batch', type=int, default=1, help='示例 batch 大小')
    parser.add_argument('--height', type=int, default=512, help='示例高度')
    parser.add_argument('--width', type=int, default=512, help='示例宽度')
    parser.add_argument('--static', action='store_true', help='使用静态形状(不设置动态轴)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"未找到 checkpoint: {args.checkpoint}")

    print(f"[Info] 加载模型: {args.checkpoint}")
    model = UNetLightningModule.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.to('cpu')  # ONNX 导出使用 CPU 更稳妥

    # 构造示例输入
    example = torch.randn(args.batch, 1, args.height, args.width, dtype=torch.float32)

    # 先跑一次，确保前向可用
    with torch.no_grad():
        out = model(example)
        if isinstance(out, tuple):
            if len(out) >= 2:
                logits, probs = out[0], out[1]
            elif len(out) == 1:
                logits = out[0]
                probs = torch.softmax(logits, dim=1)
            else:
                raise RuntimeError("模型前向返回空 tuple，无法导出")
        else:
            logits = out
            probs = torch.softmax(logits, dim=1)
        print(f"[Check] logits={tuple(logits.shape)}, probs={tuple(probs.shape)}")

    # ONNX 导出
    input_names = ['input']
    output_names = ['logits', 'probs']
    dynamic_axes = None
    if not args.static:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'logits': {0: 'batch', 2: 'height', 3: 'width'},
            'probs': {0: 'batch', 2: 'height', 3: 'width'},
        }

    # 确保输出目录存在（默认写入 checkpoints/onnx）
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[Export] 导出到 {args.output} (opset={args.opset}, dynamic_axes={'ON' if dynamic_axes else 'OFF'})")
    torch.onnx.export(
        model,
        (example,),
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("[Done] ONNX 导出完成。")


if __name__ == '__main__':
    main()
