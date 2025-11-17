./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G27.99-1.27_20251108_snapshot30-M07-P3-c2048b1.fits --blocksPerRead=4 --binFactorTime=4 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G27.99-1.27_20251108_4classes_3.0_3.0_Downsamp4

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G28.58+3.81_20220914_snapshot-M06-P2-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G28.58+3.81_20220914_4classes_3.0_3.0_Downsamp1

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G30.00+6.44_20240120_snapshot-M09-P4-c2048b1.fits --blocksPerRead=64 --binFactorTime=64 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.3
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G30.00+6.44_20240120_4classes_3.0_3.0_Downsamp64

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G30.54+4.66_20190917_snapshot-M05-P4-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.5
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G30.54+4.66_20190917_4classes_3.0_2.5

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G31.37+2.20_20201111_snapshot-M07-P1-c2048b1.fits --blocksPerRead=4 --binFactorTime=4 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.2
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G31.37+2.20_20201111_4classes_3.0_3.0_Downsamp4

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G33.18-1.44_20220711_snapshot-M13-P2-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G33.18-1.44_20220711_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G44.14+6.52_20210628_snapshot-M06-P3-c2048b1.fits --blocksPerRead=16 --binFactorTime=16 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G44.14+6.52_20210628_4classes_3.0_3.0_Downsamp16

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G45.56+2.71_20230825_snapshot-M07-P3-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G45.56+2.71_20230825_4classes_3.0_2.7

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G50.94+1.02_20210822_snapshot-M02-P1-c2048b1.fits --blocksPerRead=8 --binFactorTime=8 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G50.94+1.02_20210822_4classes_3.0_3.0_Downsamp8

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G57.49-0.17_20190321_snapshot-M01-P1-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G57.49-0.17_20190321_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G57.49-0.17_20211221_snapshot-M03-P2-c2048b1.fits --blocksPerRead=32 --binFactorTime=32 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7 --fallbackMeanNSigma=1.8
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G57.49-0.17_20211221_4classes_3.0_2.7_Downsamp32

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G57.54+0.25_20201102_snapshot-M01-P1-c2048b1.fits --blocksPerRead=32 --binFactorTime=32 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.4
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G57.54+0.25_20201102_4classes_3.0_3.0_Downsamp32

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G65.07+1.10_20230604_snapshot-M12-P4-c2048b1.fits --blocksPerRead=8 --binFactorTime=8 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.7
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G65.07+1.10_20230604_4classes_3.0_3.0_Downsamp8

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G69.04+0.00_20190320_snapshot-M06-P2-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7 --fallbackMeanNSigma=1.5
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G69.04+0.00_20190320_4classes_3.0_2.7

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G70.89+0.00_20200219_snapshot-M06-P3-c2048b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G70.89+0.00_20200219_4classes_3.0_3.0_Downsamp2

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G72.95-1.53_20251108_snapshot-M14-P1-c2048b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.5
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G72.95-1.53_20251108_4classes_3.0_3.0_Downsamp2

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G73.29-1.78_20251108_snapshot-M06-P1-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G73.29-1.78_20251108_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G80.92-0.08_20210226_snapshot-M05-P4-c2048b1.fits --blocksPerRead=64 --binFactorTime=64 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.5
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G80.92-0.08_20210226_4classes_3.0_2.5_Downsamp64

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G82.54-2.20_20240120_snapshotzcal-M07-P2-c2048b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G82.54-2.20_20240120_4classes_3.0_3.0_Downsamp2

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G90.22-0.08_20220808_snapshot-M19-P4-c2048b1.fits --blocksPerRead=16 --binFactorTime=16 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G90.22-0.08_20220808_4classes_3.0_2.7_Downsamp16

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G94.18-1.19_20251011_snapshot-M03-P4-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G94.18-1.19_20251011_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G120.82-21.31_20240812_snapshotdec-M16-P1-c1024b1.fits --blocksPerRead=4 --binFactorTime=4 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G120.82-21.31_20240812_4classes_3.0_3.0_Downsamp4

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G135.96-8.90_20220225_snapshot-M10-P3-c1024b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G135.96-8.90_20220225_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G136.25-9.57_20210524_snapshot-M05-P4-c1024b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G136.25-9.57_20210524_4classes_3.0_2.7_Downsamp2

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G150.00+0.00_20230220_snapshot-M13-P3-c1024b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.5
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G150.00+0.00_20230220_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G152.25-3.05_20190509_snapshot-M04-P1-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G152.25-3.05_20190509_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G152.35-2.20_20200104_snapshot-M05-P3-c2048b1.fits --blocksPerRead=16 --binFactorTime=16 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G152.35-2.20_20200104_4classes_3.0_3.0_Downsamp16

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G166.34-0.17_20221108_snapshot-M13-P2-c1024b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G166.34-0.17_20221108_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G167.71+2.03_20241127_snapshot-M04-P4-c1024b1.fits --blocksPerRead=8 --binFactorTime=8 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G167.71+2.03_20241127_4classes_3.0_2.7_Downsamp8

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G174.90-0.08_20200109_snapshot-M03-P3-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.8 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G174.90-0.08_20200109_4classes_3.0_2.8

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G183.17+0.68_20230216_snapshot-M18-P3-c1024b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G183.17+0.68_20230216_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G191.19+2.54_20250408_snapshot-M05-P2-c1024b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G191.19+2.54_20250408_4classes_3.0_3.0

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G200.14+2.80_20250409_snapshot-M05-P1-c1024b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.3
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G200.14+2.80_20250409_4classes_3.0_2.3

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G200.48+2.54_20250409_snapshot-M01-P1-c1024b1.fits --blocksPerRead=4 --binFactorTime=4 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.3
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G200.48+2.54_20250409_4classes_3.0_2.3_Downsamp4

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G200.53+2.97_20250409_snapshot-M01-P2-c1024b1.fits --blocksPerRead=2 --binFactorTime=2 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=2.7 --fallbackMeanNSigma=1.7
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G200.53+2.97_20250409_4classes_3.0_2.7_Downsamp2

./build/ReadFASTData --filename=/mnt/d/FASTData/FITSFiles/G213.40+1.69_20191223_snapshot-M05-P3-c2048b1.fits --blocksPerRead=1 --binFactorTime=1 --binFactorFreq=1 --plot=0 --savePlot=1 --write=1 --writeBack=0 --writeMasks=1 --doSubstitution=1 --doSumThreshold=1 --startTime=0.0 --generateMasks=1 --datasetPath=/home/cbm/deRFI/output --enableCuda=0 --inChanNSigma=3.0 --outChanNSigma=3.0 --fallbackMeanNSigma=1.0
mkdir output
python ./src/split_dataset.py --new_name=Datasets/Dataset_G213.40+1.69_20191223_4classes_3.0_3.0