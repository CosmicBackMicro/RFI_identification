#pragma once
#include <fitsio.h>
#include <stdbool.h>
typedef struct {
    /* From Command Line */
    char *filename;
    float startTime;
    int plot, write, writeBack, writeMasks;
    int savePlot;
    int binFactorTime, binFactorFreq;
    float timeDuration;
    int generateMasks;
    char *datasetPath;
    int doSubstitution;
    int doSumThreshold;
    int enableCuda;  // 0 = disable CUDA, 1 = enable CUDA (default)
    int cudaReady;   // 0 = CUDA not available/initialized, 1 = CUDA ready to use

    /* From FITS Header */
    int nchan;
    double chan_bw;
    int nsblk;
    int naxis2;
    double tbin;
    int npol;
    int colnumData;
    int colnumFreq;

    /* Calculated after reading from cmdline and header */
    int nsamp;
    int nsampBinned, nchanBinned;
    float tbinBinned, chan_bwBinned;
    int blocksPerRead, blockSize, binnedBlockSize;
    /* Calculated after reading frequency array */
    float lofreq, hifreq; // Invariant to downsampling
} Metadata;

char *extractSourceName(const char *absolutePath);

void getProfile(float *array, int nsamp, int nchan, float *freqProfile, float *timeProfile, bool *mask);

void getProfileStd(float *array, int nsamp, int nchan, float *freqProfile, float *timeProfile, bool *mask);

void downsamp2D(float *array, int nsamp, int nchan, 
    float *binnedArray, int binFactorTime, int binFactorFreq, int isTranspose);

void downsamp1D(float *array, int inputSize, int binFactor, float *binnedArray);

void upsampleMask2D(int *binnedMask, int nsampBinned, int nchanBinned,
                    int *originalMask, int nsampOriginal, int nchanOriginal,
                    int binFactorTime, int binFactorFreq, int isTranspose);

void readRawBlock(fitsfile *fptr, int blockIndex, int blocksPerRead, int nchan, int blockSize,
                  float *scale, float *offset, float *scaleRows, float *offsetRows,
                  unsigned char *outRawData, int *fits_status);

