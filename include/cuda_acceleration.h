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
