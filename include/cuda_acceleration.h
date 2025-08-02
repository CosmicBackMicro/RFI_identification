/*
 * CUDA Acceleration Header for deRFI
 * 
 * This header declares CUDA-accelerated functions for the RFI detection pipeline.
 * Functions provide GPU acceleration for compute-intensive operations.
 * 
 * Author: GitHub Copilot
 * Date: 2025-08-02
 */

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
