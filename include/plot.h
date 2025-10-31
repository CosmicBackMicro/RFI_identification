#pragma once
#include "ReadFASTData.h"
#include "identification.h"

// Ensure a PGPLOT device is opened once.
// If device is NULL, choose default: 
//   - "/xs" when DISPLAY is set (interactive)
//   - "output/plot.ps/PS" otherwise (headless)
// Returns 1 on success, 0 on failure. Safe to call multiple times.
int ensure_pgplot_device(const char *device);

void plotDownsampLongTimeAbs(
    Metadata *m, 
    int numReads, 
    float *dsDataT, 
    float *dsFreqArray, 
    float startTime, 
    int currentBlock);

void plotTimeFreqSED(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, 
    int currentBlock, float *baseline, int topPanelMode, int rightPanelMode, bool *mask, int *flaggedChans);

void plotAllMasks(Metadata *m, int blocksPerRead, float *outDataT, float *dsFreqArray, int startTime, int numiter, IdentNSigmaMasks *maskSet, int *flaggedChans);

void plotIndexMask(
    float fmin, 
    int nchanPlot, 
    float chan_bwPlot,
    float tmin, 
    int nsampPlot, 
    float tbinPlot,
    bool *mask,
    int plotStartChan,
    int plotEndChan
);

void plot8bitHist(int *hist, float lowerBound, float upperBound, float mean, float sigma, int maxBin);