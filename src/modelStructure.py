import os
import sys
import torch
from torchinfo import summary

# 将 src 目录添加到模块搜索路径，以便导入 UNet
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from SegFormer_StrategyAltered import UNetLightningModule
except ImportError:
    print("错误: 无法导入 src/UNet.py。请确保脚本在项目根目录下运行。")
    sys.exit(1)

def main():
    # 1. 配置参数（根据 UNet.py 中的默认值或实际需求）
    encoder_name = "mit_b2"
    num_classes = 6 # 脚本中通常处理 6 类
    input_size = (1, 1, 512, 512) # (Batch, Channel, Height, Width)

    print(f"正在初始化模型: Backbone={encoder_name}, Classes={num_classes}...")
    
    # 2. 实例化 LightningModule
    module = UNetLightningModule(
        encoder_name=encoder_name,
        classes=num_classes
    )
    
    # 我们主要查看内部封装的真正模型结构 (smp.Unet)
    model = module.model
    depth = 3

    print("\n" + "="*50)
    print(f"模型分层结构概览 (Depth = {depth}):")
    print("="*50)
    
    # 3. 输出概要信息
    # depth=1: 只看顶级模块 (encoder, decoder, head)
    # depth=2: 能看到 Transformer 的 Stage 和 Unet 的 DecoderBlock
    # depth=3: 可以看到 Block 内部细节
    summary(
        model, 
        input_size=input_size, 
        device="cpu", 
        depth=depth, 
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"]
    )

if __name__ == "__main__":
    main()
