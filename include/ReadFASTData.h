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
    int noBlock;     // 0 = enable block RFI detection (default), 1 = disable
    int noVertical;  // 0 = enable vertical RFI detection (default), 1 = disable
    int enableCuda;  // 0 = disable CUDA, 1 = enable CUDA (default)
    int cudaReady;   // 0 = CUDA not available/initialized, 1 = CUDA ready to use
    int enableIQRM;  // 0 = disable IQRM, 1 = enable IQRM
    int enableCLFD;  // 0 = disable CLFD, 1 = enable CLFD
    int ncpus;       // Number of CPU threads to use (CLI --ncpus)
    // NSigma thresholds (configurable via CLI), defaults set in parseCommandLineArguments
    float NSigmaInChan;   // iterative in-channel outlier threshold (sigma)
    float NSigmaOutChan;  // cross-channel (out-of-channel) threshold (sigma)
    float FallbackMeanNSigma; // fallback mean-based channel sigma clip threshold (default 2.0)

     /* Pulse mask parameters (optional; used to generate an auxiliary pulse mask)
         hasPulse=0 disables identPulse entirely. All time values are in seconds.
         T0_local is window-local time where t=0 is the beginning of current block.
     */
     int hasPulse;
     float pulseDM;
     float pulseP0;
     float pulseWidth;
     float pulseT0Local;
     int interpulse;
     float interpulseWidth;
     float interpulseT0;  // New: independent interpulse T0
     float pulselofreq;   // Min frequency for pulse masking (MHz)
     float pulsehifreq;   // Max frequency for pulse masking (MHz)

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

