# !/bin/bash

INPUT_FILE="J195137+283648_20240726_tracking-M01-P1-c512b1.fits"
# INPUT_FILE="G57.49-0.17_20190321_snapshot-M01-P1-c2048b1.fits"
# INPUT_FILE="G200.14+2.80_20250409_snapshot-M05-P1-c1024b1.fits"
# INPUT_FILE="B0355+54_20191110_tracking_1-M01-P1-c2048b1.fits"
# INPUT_FILE="G57.49-0.17_20211221_snapshot-M03-P2-c2048b1.fits"
# INPUT_FILE="B1929+10_20210106_tracking-M01-P1-c1024b1.fits"
# INPUT_FILE="G65.07+1.10_20230604_snapshot-M12-P4-c2048b1.fits"

"${WORKSPACE:-.}/build/ReadFASTData" \
    --filename="/mnt/d/FASTData/FITSFiles/$INPUT_FILE" \
    --plot=1 --write=1 --savePlot=1 \
    --doSubstitution=1 --doSumThreshold=1 \
    --blocksPerRead=8 --startTime=0.0 \
    --generateMasks=1 \
    --datasetPath="/home/cbm/deRFI/src/dataset"