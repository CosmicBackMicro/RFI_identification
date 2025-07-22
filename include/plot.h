#pragma once
#include "ReadFASTData.h"

void plotDownsampLongTimeAbs(
    Metadata *m, 
    int numReads, 
    float *dsDataT, 
    float *dsFreqArray, 
    float startTime, 
    int currentBlock);

void plotDownsampSED(
    Metadata *m, 
    int numReads, 
    float *dsDataT, 
    float *dsFreqArray, 
    float startTime, 
    int currentBlock, 
    float *baseline);

void plotDownsampSEDStd(Metadata *m, int numReads, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, float *baseline);
void plotDataAndMaskStd(Metadata *m, int numBuffs, float *dsDataT, float *dsFreqArray, float startTime, int currentBlock, int *mask);

void plotDataAndMask(
    Metadata *m, 
    int numBuffs, 
    float *dsDataT, 
    float *dsFreqArray, 
    float startTime, 
    int currentBlock, 
    int *mask);

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