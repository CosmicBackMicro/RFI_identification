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

float ksigma_2d(const float *dataT, const int *mask, int nsamp, int nchan,
                float *unmasked_buf, float *median_temp_buf);

// Function to randomly replace flagged pixels with unflagged pixels from the same time sample
// Replace pixels for channels flagged in channelMask using values from unflagged channels at the same time
// If pointMask is non-NULL, only unmasked pixels are used as source
void outChanSubstitution(float *data, const int *channelMask, const int *pointMask, int nsamp, int nchan);

// Substitute in-channel outliers using local statistics
void inChanSubstitution(float *data, int *globalMask, int nsamp, int nchan, int *pixelsSubstituted);
void substPixels(float *data, int size, int *mask, int *goodSamps, int *randIdx);
void binarySIR(int *mask, int nsamp, int nchan, int win_samp, int win_chan, float thrup, float thrdown);

void outChanDetection(float *data, int nsamp, int nchan, int *channelFlagged,
                              float *channel_stds, float *channel_stds_temp, float channel_std_threshold, float nsigma_in, int plot);

int meanOutlierDetection(float *data, int nsamp, int nchan, int *channelFlagged);


void subtractChannelMedians(float *data, int nsamp, int nchan);

typedef struct IdentNSigmaMasks {
    int *horizontalMask;
    int *verticalMask;
    int *globalMask;
    int *pointMask;
    int *chanBrightMask;
    int *chanDarkMask;
    int *chanComplexMask;
} IdentNSigmaMasks;

void identSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan, int iterationIndex, int plot,
    IdentNSigmaMasks *masks,
    float *finalMedian, float *finalStd, int cudaReady, int *flaggedChans);

// Histogram functions
// OutChannel comparison histogram function
void drawOutChannelComparisonHist(float *initial_stats, float *final_stats, int nchan, 
                                  int use_mad, int initial_flagged_count, int final_flagged_count,
                                  int iterations, float nsigma_out, float nsigma_in);

// New consistent-shape version (keeps initial histogram shape fixed)
void drawOutChannelComparisonHist_new(float *initial_stats, float *final_stats, int nchan,
                                      int use_mad, int initial_flagged_count, int final_flagged_count,
                                      int iterations, float nsigma_out, float nsigma_in);
