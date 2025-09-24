#ifndef CUDA_ACCELERATION_H
#define CUDA_ACCELERATION_H

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUDA Accelerated Functions
// ============================================================================

/**
 * Initialize CUDA and check device capabilities
 * @return 0 on success, -1 on failure
 */
int cuda_init(void);

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup(void);

/**
 * Check if CUDA is available and functional
 * @return 1 if CUDA is available, 0 otherwise
 */
int cuda_isAvailable(void);

/**
 * CUDA-accelerated channel median subtraction
 * Subtracts median value from each frequency channel
 * 
 * @param data Input/output data array (nsamp * nchan elements)
 * @param channel_medians Array of median values for each channel
 * @param nsamp Number of time samples per channel
 * @param nchan Number of frequency channels
 */
void cuda_subtractChannelMedians(float *data, const float *channel_medians, 
                                int nsamp, int nchan);

/**
 * CUDA-accelerated channel statistics calculation
 * Calculates mean and standard deviation for each channel
 * 
 * @param data Input data array (nsamp * nchan elements)
 * @param means Output array for channel means (nchan elements)
 * @param stds Output array for channel standard deviations (nchan elements)
 * @param nsamp Number of time samples per channel
 * @param nchan Number of frequency channels
 */
void cuda_calculateChannelStats(const float *data, float *means, float *stds, 
                               int nsamp, int nchan);

/**
 * CUDA-accelerated matrix transpose
 * Transposes a 2D matrix using shared memory optimization
 * 
 * @param input Input matrix (rows * cols elements)
 * @param output Output transposed matrix (cols * rows elements)
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 */
void cuda_transpose(const float *input, float *output, int rows, int cols);

/**
 * CUDA-accelerated 2D downsampling
 * Downsamples a 2D array by averaging over bins
 * 
 * @param input Input array (nsamp * nchan elements)
 * @param output Output downsampled array ((nsamp/binFactorTime) * (nchan/binFactorFreq) elements)
 * @param nsamp Number of time samples in input
 * @param nchan Number of frequency channels in input
 * @param binFactorTime Downsampling factor in time dimension
 * @param binFactorFreq Downsampling factor in frequency dimension
 */
void cuda_downsample2D(const float *input, float *output, 
                      int nsamp, int nchan,
                      int binFactorTime, int binFactorFreq);

/**
 * CUDA-accelerated binary morphological filtering (binarySIR)
 * Applies structural indexing reduction with configurable window size and thresholds
 * 
 * @param mask Input/output binary mask array (nsamp * nchan elements, 0 or 1)
 * @param nsamp Number of time samples per channel
 * @param nchan Number of frequency channels
 * @param win_samp Window size in time dimension (must be odd)
 * @param win_chan Window size in frequency dimension (must be odd)
 * @param thr_up Upper threshold for density ratio (0.0-1.0)
 * @param thr_down Lower threshold for density ratio (0.0-1.0)
 */
void cuda_binarySIR(int *mask, int nsamp, int nchan,
                   int win_samp, int win_chan, 
                   float thr_up, float thr_down);

/**
 * CUDA-accelerated pipeline (skeleton) for identSubstNSigma
 * Approximates point-level outlier detection on GPU and produces masks.
 * This is a framework function intended to be extended.
 *
 * @param data            In/Out data array (nsamp * nchan)
 * @param nsamp           Number of samples per channel
 * @param nchan           Number of channels
 * @param Nsigma          Sigma threshold for point-level detection
 * @param channel_std_threshold  Threshold for channel-level (reserved, not used in skeleton)
 * @param iterationIndex  Iteration index (reserved for logging)
 * @param plot            Whether to plot (not used in CUDA path)
 * @param horizontalMask  Output mask (nsamp * nchan), 1 = flagged
 * @param verticalMask    Output mask (nsamp * nchan), reserved/zeroed in skeleton
 * @param globalMask      Output mask (nsamp * nchan), copy of horizontalMask in skeleton
 * @param finalMedian     Output overall median estimate (approx; skeleton writes mean)
 * @param finalStd        Output overall std estimate (approx; skeleton writes std)
 * @param channel_fully_flagged Output per-channel full-flag indicator (nchan)
 */
void cuda_identSubstNSigma(float *data, int nsamp, int nchan,
                           float Nsigma, float channel_std_threshold,
                           int iterationIndex, int plot,
                           int *horizontalMask, int *verticalMask, int *globalMask,
                           float *finalMedian, float *finalStd,
                           int *channel_fully_flagged);

// ============================================================================
// Utility Macros
// ============================================================================

/**
 * Macro to conditionally use CUDA functions if available
 * Falls back to CPU implementation if CUDA is not available
 */
#define USE_CUDA_IF_AVAILABLE(cuda_func, cpu_func, ...) \
    do { \
        if (cuda_isAvailable()) { \
            cuda_func(__VA_ARGS__); \
        } else { \
            cpu_func(__VA_ARGS__); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif // CUDA_ACCELERATION_H
