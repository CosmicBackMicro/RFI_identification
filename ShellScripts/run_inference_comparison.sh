#!/bin/bash

# ==============================================================================
# AI RFI 推理对比实验脚本
# 该脚本用于对比 SegFormer 和 UNet 两个模型对同一 PSRFITS 文件的处理效果。
# ==============================================================================

# --- 配置区 ---
SRC_FITS="/mnt/c/FASTData/FITSFiles/G30.00+6.44_20240120_snapshot-M09-P4-c2048b1.fits"
# 从文件名提取第一个下划线前的部分作为 Cover 名 (例如 G31.37+2.20_... -> G31.37+2.20)
COVER_NAME=$(basename "$SRC_FITS" | cut -d'_' -f1)
NTODO=300

# 模型引擎路径（无需改动）
SF_ENGINE="checkpoints/tensorrt/SegFormer-B2_40000_Actual+Sim+Pulsar/BEST_SegFormer-B2_epoch28_valFGMacroF1_0.8019_batch1_w1024h896_static_fp16.engine"
UN_ENGINE="checkpoints/tensorrt/UNet+MiT-B2_40000_Actual+Sim+Pulsar/BEST_UNet+MiT-B2_epoch27_val-FG-MacroF1_0.7767_batch1_w1024h896_static_fp16.engine"

# 自动生成推理文件路径 (basename 之后加上模型名)
DIR_NAME=$(dirname "$SRC_FITS")
BASE_NAME=$(basename "$SRC_FITS" .fits)

SF_FITS="${DIR_NAME}/${BASE_NAME}_SegFormer.fits"
UN_FITS="${DIR_NAME}/${BASE_NAME}_UNet.fits"

# 自动生成结果目录 (results/AI_RFI_模型名_Cover名)
SF_RESULTS="results/AI_RFI_SegFormer_${COVER_NAME}"
UN_RESULTS="results/AI_RFI_UNet_${COVER_NAME}"

# --- 执行区 ---

echo "🚀 开始实验准备..."

# 1. 清理旧文件 (使用 -f 忽略文件不存在的情况)
rm -f "$SF_FITS" "$UN_FITS"
echo "✅ 已清理旧推理 FITS 文件。"

# 2. 准备源文件拷贝
echo "📂 正在复制原始 FITS 文件..."
cp "$SRC_FITS" "$SF_FITS"
cp "$SRC_FITS" "$UN_FITS"
echo "✅ 文件拷贝完成。"

# 3. 运行 SegFormer 推理
echo "🧠 正在运行 SegFormer 推理 (NTODO=$NTODO)..."
python src/AI_RFI.py --fits "$SF_FITS" --engine "$SF_ENGINE" --ntodo "$NTODO"

if [ $? -eq 0 ]; then
    echo "📦 正在整理 SegFormer 结果..."
    rm -rf "$SF_RESULTS"
    mkdir -p "$SF_RESULTS"
    # 检查是否有结果生成，避免 mv 报错
    if [ "$(ls -A results/AI_RFI 2>/dev/null)" ]; then
        mv results/AI_RFI/* "$SF_RESULTS/"
        echo "✅ SegFormer 推理完成，结果及掩模存放在: $SF_RESULTS"
    else
        echo "⚠️  推理已完成但未发现生成的掩模文件。"
    fi
else
    echo "❌ SegFormer 推理失败，跳过整理步骤。"
fi

# 4. 运行 UNet 推理
echo "🧠 正在运行 UNet 推理 (NTODO=$NTODO)..."
python src/AI_RFI.py --fits "$UN_FITS" --engine "$UN_ENGINE" --ntodo "$NTODO"

if [ $? -eq 0 ]; then
    echo "📦 正在整理 UNet 结果..."
    rm -rf "$UN_RESULTS"
    mkdir -p "$UN_RESULTS"
    # 检查是否有结果生成，避免 mv 报错
    if [ "$(ls -A results/AI_RFI 2>/dev/null)" ]; then
        mv results/AI_RFI/* "$UN_RESULTS/"
        echo "✅ UNet 推理完成，结果及掩模存放在: $UN_RESULTS"
    else
        echo "⚠️  推理已完成但未发现生成的掩模文件。"
    fi
else
    echo "❌ UNet 推理失败，跳过整理步骤。"
fi

echo "==========================================================="
echo "🎉 所有实验任务执行结束。"
echo "您可以运行以下命令对比结果："
echo "python src/visualize_fits.py --psrfits $SF_FITS --mask $SF_RESULTS"
echo "python src/visualize_fits.py --psrfits $UN_FITS --mask $UN_RESULTS"
echo "==========================================================="
