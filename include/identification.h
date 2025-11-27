#pragma once
#include <stdbool.h>
#include <stddef.h>
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
// 2D variant: substitute masked pixels per-channel using unmasked samples from same channel
void substPixels2D(float *data, int nsamp, int nchan, int *mask);
void binarySIR(int *mask, int nsamp, int nchan, int win_samp, int win_chan, float thrup, float thrdown);

void outChanDetection(float *data, int nsamp, int nchan, int *channelFlagged,
                              float *channel_stds, float *channel_stds_temp, float channel_std_threshold, float nsigma_in, int plot);

/* 设置线程局部的保底均值缓冲区（用于 outChanDetection 的 ±kσ 均值兜底检测，避免并行循环内重复 malloc）*/
void setFallbackChannelMeansBuffer(float *buf, size_t size);
/* 设置保底均值检测的 σ 倍数（默认 2.0，可通过命令行参数配置）*/
void setFallbackMeanNSigma(float v);

int meanOutlierDetection(float *data, int nsamp, int nchan, int *channelFlagged);

int inChanDetection(float *data, int nsamp, int nchan, float Nsigma,
    bool *horizontalMask, int *channel_fully_flagged,
    float *scratch, size_t scratch_count);

void subChanMed(float *data, int nsamp, int nchan, float *channel_medians, float *temp_data);

typedef struct IdentNSigmaMasks {
    bool *horizontalMask;
    bool *verticalMask;
    bool *blockMask;  // New: block RFI mask
    bool *periodicMask; // New: periodic point RFI mask (subset of pointMask)
    bool *globalMask;
    bool *pointMask;
    bool *chanBrightMask;
    bool *chanDarkMask;
    bool *chanComplexMask;
} IdentNSigmaMasks;

void identSubstNSigma(
     float *data, int nsamp, int nchan,
     float NSigmaInChan, float NSigmaOutChan, int iterationIndex, int plot, int doSubstitute,
     IdentNSigmaMasks *masks,
     float *finalMedian, float *finalStd, int cudaReady, int *flaggedChans,
     int *identSubst_goodSamps, int *identSubst_randIdxs, float *identSubst_medTemp,
     float *inChanScratch, size_t inChanScratchCount,
     /* Caller-provided CLFD mask buffer (int array of size nsamp*nchan, per-thread slice)
         This avoids allocating inside CLFD hot paths.
     */
     int *clfd_mask_buf,
     /* Vertical-stripe detection scratch buffers (allocated by caller) */
     float *vs_time_means_buf, unsigned char *vs_flag_time_buf);

// Histogram functions
// OutChannel comparison histogram function
void drawOutChannelComparisonHist(float *initial_stats, float *final_stats, int nchan, 
                                  int initial_flagged_count, int final_flagged_count,
                                  int iterations, float nsigma_out, float nsigma_in);

// Detect broadband vertical stripes (all or most channels active at the same time index)
// using peaks on the time-mean and/or time-std sequences.
// verticalMask is set to 1 for all channels at flagged time indices.
void detectVerticalStripesByTimeProfiles(
    const float *data,
    int nsamp,
    int nchan,
    const bool *pointMask,      // exclude: per-pixel point-level RFI
    const bool *horizontalMask, // exclude: fully-flagged channels expanded to 2D
    bool *verticalMask,
    float nsigma_mean,
    int min_run,
    int plot,
    /* Scratch buffers provided by caller to avoid internal allocations */
    float *time_means_buf,
    unsigned char *flag_time_buf);

// Detect vertical "string" interference caused by duplicated time samples:
// identifies runs where adjacent time columns are nearly identical across channels.
// For each time index t>0, compute an error metric err[t] = mean(|col(t)-col(t-1)|) over unmasked pixels.
// Flag times with err[t] <= max(abs_epsilon, rel_sigma*std(err)) and group runs with length >= min_run.
// The function ORs results into verticalMask (does not clear it).
void detectVerticalRepeatedColumns(
    const float *data,
    int nsamp,
    int nchan,
    const bool *pointMask,
    const bool *horizontalMask,
    bool *verticalMask,
    float abs_epsilon,     // absolute floor for near-zero detection (e.g., 1e-6)
    float rel_sigma,       // relative threshold vs std(err), e.g., 0.25
    int min_run,           // minimum consecutive times to accept
    int plot,              // reserved for future visualization
    float *err_buf,        // length >= nsamp (reused scratch buffer)
    unsigned char *flag_time_buf // length >= nsamp (0/1 flags)
);

// Simple block RFI detection using connected components
void detectBlockRFI(
    const bool *binaryMask, // input binary mask (e.g., pointMask). Will NOT be modified
    int nsamp, int nchan,
    bool *blockMask, // output block RFI mask
    int min_area, float min_density, // simple thresholds
    int dilate_radius, int dilate_iterations // extra dilation applied internally on a copy
);

// Detect periodic point RFI strictly within pointMask.
// For each channel independently, search period in [min_period, max_period] using a simple
// autocorrelation-on-binary approach, then mark only those pointMask pixels that participate
// in the best period as periodic. The output periodicMask is a subset of pointMask.
void detectPeriodicPointRFI(
    const bool *pointMask,
    int nsamp, int nchan,
    bool *periodicMask,
    int min_period, int max_period,
    int min_pairs,          // minimum matched pairs x[t] & x[t+T]
    float min_align_frac    // s[T]/sum(x) threshold to accept periodicity
);

// If a channel has > ratio_thresh of its samples flagged in pointMask,
// consider it point-dominated and clear its horizontalMask; also clear flaggedChans[ch].
// Returns the number of channels whose horizontalMask was canceled.
int cancelHorizontalMaskForPointDominantChannels(
    const bool *pointMask,
    int nsamp,
    int nchan,
    float ratio_thresh,      // e.g., 0.30f
    bool *horizontalMask,
    int *flaggedChans);

/* Global toggles for alternative channel-detection algorithms (set from caller) */
void setUseIQRM(int v);
void setUseCLFD(int v);