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
 * CUDA-accelerated RFI detection using N-sigma thresholding
 * Processes the complete identSubstNSigma pipeline on GPU
 * 
 * @param data Input data array (nsamp * nchan elements, will be modified)
 * @param nsamp Number of time samples
 * @param nchan Number of channels
 * @param NSigmaInChan N-sigma threshold for in-channel (pixel-level) detection
 * @param NSigmaOutChan N-sigma threshold for out-channel (channel-level) detection
 * @param iterationIndex Current iteration index for logging
 * @param plot Whether to generate plots (currently not supported in CUDA version)
 * @param masks Structure containing all mask arrays
 * @param finalMedian Output: final median of processed data
 * @param finalStd Output: final standard deviation of processed data
 * @param flaggedChans Output: array indicating which channels are flagged
 * @return 0 on success, -1 on failure
 */
int cuda_identSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan,
    int iterationIndex, int plot, int doSubstitute,
    void *masks,
    float *finalMedian, float *finalStd, int *flaggedChans);

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
