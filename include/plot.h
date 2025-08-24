#pragma once
#include "ReadFASTData.h"

void plotDownsampLongTimeAbs(
    Metadata *m, 
    int numReads, 
    float *dsDataT, 
    float *dsFreqArray, 
    float startTime, 
    int currentBlock);

void plotTimeFreqSED(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, float *baseline, int topPanelMode, int rightPanelMode, int *mask);

void plotIndexMask(
    float fmin, 
    int nchanPlot, 
    float chan_bwPlot,
    float tmin, 
    int nsampPlot, 
    float tbinPlot,
    int *mask,
    int plotStartChan,
    int plotEndChan
);

void plot8bitHist(int *hist, float lowerBound, float upperBound, float mean, float sigma, int maxBin);