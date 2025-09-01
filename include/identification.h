#pragma once

void sumthreshold_2d(
    const float *dataT, 
    int nsamp, 
    int nchan,
    int *mask_chanRFI,
    int *mask,
    float timesOfSigma,
    int M_len);

float ksigma_1d(float *data, int n, int bins, float *hist, float *x_val, float *median_temp);

float ksigma_2d(const float *dataT, const int *mask, int nsamp, int nchan);

void writeIndexMaskPNG(int *mask, int nsamp, int nchan, char *filename);

void mergeMask2D(int *masks[], int nmasks, int nsamp, int nchan, int *result);

void substPixels2D(float *data, int nsamp, int nchan, int *mask);
void substPixels(float *data, int size, int *mask, int *goodSamps, int *randIdx);
void binarySIR(int *mask, int nsamp, int nchan, int win_samp, int win_chan, float thrup, float thrdown);

void flagChannelsByMeanOutliers(float *data, int nsamp, int nchan, int *horizontalMask,
                               float *channel_means, float *channel_means_temp);
void flagChannelsByStdOutliers(float *data, int nsamp, int nchan, int *horizontalMask,
                              float *channel_stds, float *channel_stds_temp);

void normalizeChannelData(float *data, int nsamp, int nchan, 
                         float *finalMedian, float *finalStd, float *median_temp);

void subtractChannelMedians(float *data, int nsamp, int nchan);

void visualizeChannelMAD(float *data, int nsamp, int nchan, int plot);

void visualizeChannelStd(float *data, int nsamp, int nchan, int plot);

void printThresholdStatistics(const float *channel_values, int nchan, 
                             const float *thresh_values, const char **threshold_names, 
                             int num_thresholds, const char *metric_name);

void drawUnifiedThresholdLines(const float *thresh_values, const char **threshold_labels,
                              const int *threshold_colors, const float *threshold_y_positions,
                              const int *threshold_enabled, int num_thresholds,
                              float max_count, float x_min, float x_max);

void applyKillThreshAndSubstitution(float *data, int *globalMask, int nsamp, int nchan, 
                                   float killThresh, int flaggedBefore,
                                   int *killedChannels, int *localRFISkippedPtr, 
                                   int *totalFlaggedAfter, int *pixelsSubstituted);

void identSubstNSigma(
    float *data, int nsamp, int nchan, 
    float Nsigma, int iterationIndex, int plot,
    int *horizontalMask, int *verticalMask, int *globalMask,
    float *finalMedian, float *finalStd, int cudaReady);
