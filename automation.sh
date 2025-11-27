#!/bin/bash

# 定义默认参数（可以被命令行选项覆盖）
BLOCKS_PER_READ=1
BIN_FACTOR_TIME=1
BIN_FACTOR_FREQ=1
PLOT=0
SAVE_PLOT=1
WRITE=1
WRITE_BACK=0
WRITE_MASKS=1
DO_SUBSTITUTION=1
DO_SUM_THRESHOLD=1
START_TIME=0.0
GENERATE_MASKS=1
DATASET_PATH="/home/bmcao/deRFI/output"
ENABLE_CUDA=0
IN_CHAN_NSIGMA=3.0
OUT_CHAN_NSIGMA=3.0
NCPUS=50

# 解析命令行参数
FILENAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --blocksPerRead=*)
            BLOCKS_PER_READ="${1#*=}"
            ;;
        --binFactorTime=*)
            BIN_FACTOR_TIME="${1#*=}"
            ;;
        --binFactorFreq=*)
            BIN_FACTOR_FREQ="${1#*=}"
            ;;
        --plot=*)
            PLOT="${1#*=}"
            ;;
        --savePlot=*)
            SAVE_PLOT="${1#*=}"
            ;;
        --write=*)
            WRITE="${1#*=}"
            ;;
        --writeBack=*)
            WRITE_BACK="${1#*=}"
            ;;
        --writeMasks=*)
            WRITE_MASKS="${1#*=}"
            ;;
        --doSubstitution=*)
            DO_SUBSTITUTION="${1#*=}"
            ;;
        --doSumThreshold=*)
            DO_SUM_THRESHOLD="${1#*=}"
            ;;
        --startTime=*)
            START_TIME="${1#*=}"
            ;;
        --generateMasks=*)
            GENERATE_MASKS="${1#*=}"
            ;;
        --datasetPath=*)
            DATASET_PATH="${1#*=}"
            ;;
        --enableCuda=*)
            ENABLE_CUDA="${1#*=}"
            ;;
        --inChanNSigma=*)
            IN_CHAN_NSIGMA="${1#*=}"
            ;;
        --outChanNSigma=*)
            OUT_CHAN_NSIGMA="${1#*=}"
            ;;
        --ncpus=*)
            NCPUS="${1#*=}"
            ;;
        *)
            if [[ -z "$FILENAME" ]]; then
                FILENAME="$1"
            else
                echo "错误：只接受一个 FITS 文件路径"
                echo "用法: $0 <fits文件路径> [--选项=值 ...]"
                exit 1
            fi
            ;;
    esac
    shift
done

# 检查是否提供了文件路径
if [[ -z "$FILENAME" ]]; then
    echo "错误：必须提供 FITS 文件路径"
    echo "用法: $0 <fits文件路径> [--选项=值 ...]"
    exit 1
fi

# 创建 output 目录
mkdir -p output

# 解析 filename 以构建 new_name
# 提取文件名部分（去掉路径）
BASENAME=$(basename "$FILENAME")
# 提取 "G开头坐标_yyyymmdd" 部分（假设格式为 Gxx.xx+xx.xx_yyyymmdd_...）
# 使用正则表达式匹配 G 开头，后跟坐标和日期
if [[ $BASENAME =~ ^(G[0-9]+\.[0-9]+\+[0-9]+\.[0-9]+_[0-9]{8}) ]]; then
    EXTRACTED="${BASH_REMATCH[1]}"
else
    echo "无法解析 $FILENAME 中的坐标和日期部分"
    exit 1
fi

# 构建 new_name
NEW_NAME="Datasets/Dataset_${EXTRACTED}_4classes_${IN_CHAN_NSIGMA}_${OUT_CHAN_NSIGMA}_Downsamp${BIN_FACTOR_TIME}"

# 执行第一个命令
./build/ReadFASTData \
  --filename="$FILENAME" \
  --blocksPerRead="$BLOCKS_PER_READ" \
  --binFactorTime="$BIN_FACTOR_TIME" \
  --binFactorFreq="$BIN_FACTOR_FREQ" \
  --plot="$PLOT" \
  --savePlot="$SAVE_PLOT" \
  --write="$WRITE" \
  --writeBack="$WRITE_BACK" \
  --writeMasks="$WRITE_MASKS" \
  --doSubstitution="$DO_SUBSTITUTION" \
  --doSumThreshold="$DO_SUM_THRESHOLD" \
  --startTime="$START_TIME" \
  --generateMasks="$GENERATE_MASKS" \
  --datasetPath="$DATASET_PATH" \
  --enableCuda="$ENABLE_CUDA" \
  --inChanNSigma="$IN_CHAN_NSIGMA" \
  --outChanNSigma="$OUT_CHAN_NSIGMA" \
  --ncpus "$NCPUS"

# 执行第三个命令
python ./src/split_dataset.py --new_name="$NEW_NAME"