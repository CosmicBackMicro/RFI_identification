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

void substPixels2D(float *data, int nsamp, int nchan, int *mask);

// Function to randomly replace flagged pixels with unflagged pixels from the same time sample
// Replace pixels for channels flagged in channelMask using values from unflagged channels at the same time
// If pointMask is non-NULL, only unmasked pixels are used as source
void randomReplaceRFIPixels(float *data, const int *channelMask, const int *pointMask, int nsamp, int nchan);
void substPixels(float *data, int size, int *mask, int *goodSamps, int *randIdx);
void binarySIR(int *mask, int nsamp, int nchan, int win_samp, int win_chan, float thrup, float thrdown);

void flagChannelsByMeanOutliers(float *data, int nsamp, int nchan, int *horizontalMask,
                               float *channel_means, float *channel_means_temp);
void outChanDetection(float *data, int nsamp, int nchan, int *channelFlagged,
                              float *channel_stds, float *channel_stds_temp, float channel_std_threshold, float nsigma_in);

void normalizeChannelData(float *data, int nsamp, int nchan, 
                         float *finalMedian, float *finalStd, float *median_temp);

void subtractChannelMedians(float *data, int nsamp, int nchan);

void printThresholdStatistics(const float *channel_values, int nchan, 
                             const float *thresh_values, const char **threshold_names, 
                             int num_thresholds, const char *metric_name);

void drawUnifiedThresholdLines(const float *thresh_values, const char **threshold_labels,
                              const int *threshold_colors, const float *threshold_y_positions,
                              const int *threshold_enabled, int num_thresholds,
                              float max_count, float x_min, float x_max);

typedef struct IdentNSigmaMasks {
    int *horizontalMask;
    int *verticalMask;
    int *globalMask;
    int *pointMask;
    int *chanBrightMask;
    int *chanDarkMask;
} IdentNSigmaMasks;

void identSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan, int iterationIndex, int plot,
    IdentNSigmaMasks *masks,
    float *finalMedian, float *finalStd, int cudaReady, int *flaggedChans);

// Histogram functions
void calculateHistogram(float *data, int n, int nbins, 
                       float *hist_data, float *bin_min, float *bin_max, 
                       float *bin_width, float *max_count);
void drawHistogramFromData(float *hist_data, int nbins, 
                          float bin_min, float bin_max, float bin_width, float max_count,
                          const char *xlabel, const char *ylabel, const char *title,
                          int draw_median, float median_value);
void drawSimpleChannelSTDHist(float *data, int nsamp, int nchan, int plot);
void drawSimpleChannelSTDHistWithMask(float *channel_stds, int nchan, int *channel_mask, int plot);

// OutChannel comparison histogram function
void drawOutChannelComparisonHist(float *initial_stats, float *final_stats, int nchan, 
                                  int use_mad, int initial_flagged_count, int final_flagged_count,
                                  int iterations, float nsigma_out, float nsigma_in);
