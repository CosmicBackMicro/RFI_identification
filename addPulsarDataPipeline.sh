#!/bin/bash
# 确保在任何命令出错时立即停止脚本
set -e
set -o pipefail

echo "🚀 开始执行脉冲星数据处理流水线..."

# --- B0355+54 系列 ---
echo "📂 正在处理 B0355+54 (downsamp1)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B0355+54_20191110_tracking_1-M01-P1-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=57 --pulseP0=0.15638 --pulseT0=0.1057 --pulseWidth=0.02
for i in output/B0355+54_20191110_block*; do mv "$i" "${i/_block/_downsamp1_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B0355+54_20191110_5classes_3.0_3.0_downsamp1 --val_ratio 0.2

echo "📂 正在处理 B0355+54 (downsamp2)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B0355+54_20191110_tracking_1-M01-P1-c2048b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=57 --pulseP0=0.15638 --pulseT0=0.1057 --pulseWidth=0.02
for i in output/B0355+54_20191110_block*; do mv "$i" "${i/_block/_downsamp2_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B0355+54_20191110_5classes_3.0_3.0_downsamp2 --val_ratio 0.2

echo "📂 正在处理 B0355+54 (downsamp4)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B0355+54_20191110_tracking_1-M01-P1-c2048b1.fits --blocksPerRead=4 --binFactorTime=4 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=57 --pulseP0=0.15638 --pulseT0=0.1057 --pulseWidth=0.02
for i in output/B0355+54_20191110_block*; do mv "$i" "${i/_block/_downsamp4_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B0355+54_20191110_5classes_3.0_3.0_downsamp4 --val_ratio 0.2

echo "📂 正在处理 B0355+54 (downsamp8)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B0355+54_20191110_tracking_1-M01-P1-c2048b1.fits --blocksPerRead=8 --binFactorTime=8 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=57 --pulseP0=0.15638 --pulseT0=0.1057 --pulseWidth=0.02
for i in output/B0355+54_20191110_block*; do mv "$i" "${i/_block/_downsamp8_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B0355+54_20191110_5classes_3.0_3.0_downsamp8 --val_ratio 0.2

# --- B1929+10 系列 ---
echo "📂 正在处理 B1929+10 (downsamp2)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B1929+10_20210106_tracking-M01-P1-c1024b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=3.183 --pulseP0=0.226525 --pulseT0=0.1514 --pulseWidth=0.017 --interpulse --interpulset0=0.0434 --interpulseWidth=0.004 --noBlock
for i in output/B1929+10_20210106_block*; do mv "$i" "${i/_block/_downsamp2_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B1929+10_20210106_5classes_3.0_3.0_downsamp2 --val_ratio 0.2

echo "📂 正在处理 B1929+10 (downsamp4)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B1929+10_20210106_tracking-M01-P1-c1024b1.fits --blocksPerRead=4 --binFactorTime=4 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=3.183 --pulseP0=0.226525 --pulseT0=0.1514 --pulseWidth=0.017 --interpulse --interpulset0=0.0434 --interpulseWidth=0.004 --noBlock
for i in output/B1929+10_20210106_block*; do mv "$i" "${i/_block/_downsamp4_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B1929+10_20210106_5classes_3.0_3.0_downsamp4 --val_ratio 0.2

echo "📂 正在处理 B1929+10 (downsamp8)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/B1929+10_20210106_tracking-M01-P1-c1024b1.fits --blocksPerRead=8 --binFactorTime=8 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=3.183 --pulseP0=0.226525 --pulseT0=0.1514 --pulseWidth=0.017 --interpulse --interpulset0=0.0434 --interpulseWidth=0.004 --noBlock
for i in output/B1929+10_20210106_block*; do mv "$i" "${i/_block/_downsamp8_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_B1929+10_20210106_5classes_3.0_3.0_downsamp8 --val_ratio 0.2

# --- J195401+292434 系列 ---
echo "📂 正在处理 J195401+292434 (downsamp8)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/J195401+292434_20210402_tracking-M02-P1-c2048b1.fits --blocksPerRead=8 --binFactorTime=8 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=5 --pulseP0=0.426665 --pulseT0=0.065 --pulseWidth=0.07 --pulsehifreq=1250 --noVertical
for i in output/J195401+292434_20210402_block*; do mv "$i" "${i/_block/_downsamp8_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_J195401+292434_20210402_5classes_3.0_3.0_downsamp8 --val_ratio 0.2

echo "📂 正在处理 J195401+292434 (downsamp16)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/J195401+292434_20210402_tracking-M02-P1-c2048b1.fits --blocksPerRead=16 --binFactorTime=16 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=5 --pulseP0=0.426665 --pulseT0=0.065 --pulseWidth=0.07 --pulsehifreq=1250 --noVertical
for i in output/J195401+292434_20210402_block*; do mv "$i" "${i/_block/_downsamp16_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_J195401+292434_20210402_5classes_3.0_3.0_downsamp16 --val_ratio 0.2

echo "📂 正在处理 J195401+292434 (downsamp32)..."
./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/J195401+292434_20210402_tracking-M02-P1-c2048b1.fits --blocksPerRead=32 --binFactorTime=32 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=0 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --hasPulse --pulseDM=5 --pulseP0=0.426665 --pulseT0=0.065 --pulseWidth=0.07 --pulsehifreq=1250 --noVertical
for i in output/J195401+292434_20210402_block*; do mv "$i" "${i/_block/_downsamp32_block}"; done
python src/split_dataset.py --new_name Datasets/Dataset_J195401+292434_20210402_5classes_3.0_3.0_downsamp32 --val_ratio 0.2

echo "✅ 所有任务执行完毕！"
