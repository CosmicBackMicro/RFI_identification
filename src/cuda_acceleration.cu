#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>
#include <omp.h>

// Include header for CUDA functions
#include "cuda_acceleration.h"
// Include identification header for struct definitions
#include "identification.h"

// Ensure C linkage for interface functions
extern "C" {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Runtime Management
// ============================================================================

// Global state for CUDA availability
static int cuda_availability_checked = 0;
static int cuda_is_available = 0;

// Function pointers for dynamically loaded CUDA functions
static void* cuda_lib_handle = NULL;

// Check CUDA availability at runtime
static int check_cuda_availability(void) {
    if (cuda_availability_checked) {
        return cuda_is_available;
    }
    
    cuda_availability_checked = 1;
    
    // Try to load CUDA runtime library
    cuda_lib_handle = dlopen("libcudart.so", RTLD_LAZY);
    if (!cuda_lib_handle) {
        // Try alternative CUDA library paths
        cuda_lib_handle = dlopen("libcudart.so.11", RTLD_LAZY);
        if (!cuda_lib_handle) {
            cuda_lib_handle = dlopen("libcudart.so.12", RTLD_LAZY);
        }
    }
    
    if (!cuda_lib_handle) {
        cuda_is_available = 0;
        return 0;
    }
    
    // Check if we can load basic CUDA runtime functions
    void* cudaGetDeviceCount = dlsym(cuda_lib_handle, "cudaGetDeviceCount");
    if (!cudaGetDeviceCount) {
        dlclose(cuda_lib_handle);
        cuda_lib_handle = NULL;
        cuda_is_available = 0;
        return 0;
    }
    
    // Try to get device count to verify CUDA is functional
    int (*getDeviceCount)(int*) = (int(*)(int*))cudaGetDeviceCount;
    int deviceCount = 0;
    int result = getDeviceCount(&deviceCount);
    
    if (result != 0 || deviceCount == 0) {
        dlclose(cuda_lib_handle);
        cuda_lib_handle = NULL;
        cuda_is_available = 0;
        return 0;
    }
    
    cuda_is_available = 1;
    return 1;
}

// ============================================================================
// Public CUDA Interface Functions
// ============================================================================

int cuda_isAvailable(void) {
    return check_cuda_availability();
}

int cuda_init(void) {
    if (!cuda_isAvailable()) {
        return -1;
    }
    
    // Try to initialize the first CUDA device
    void* cudaSetDevice = dlsym(cuda_lib_handle, "cudaSetDevice");
    void* cudaGetDeviceProperties = dlsym(cuda_lib_handle, "cudaGetDeviceProperties");
    
    if (!cudaSetDevice || !cudaGetDeviceProperties) {
        return -1;
    }
    
    // Set device 0
    int (*setDevice)(int) = (int(*)(int))cudaSetDevice;
    int result = setDevice(0);
    if (result != 0) {
        return -1;
    }
    
    printf("CUDA initialization successful.\n");
    return 0;
}

void cuda_cleanup(void) {
    if (cuda_lib_handle) {
        dlclose(cuda_lib_handle);
        cuda_lib_handle = NULL;
    }
    cuda_is_available = 0;
    cuda_availability_checked = 0;
}

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

/**
 * CUDA kernel for transpose operation
 * Uses shared memory tile for coalesced memory access
 */
#define TILE_SIZE 32

__global__ void transposeKernel(const float *input, float *output, 
                              int rows, int cols)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read from input matrix into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write from shared memory to output matrix (transposed)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ---------------------------------------------------------------------------
// Transpose Kernel-Wrapper Pair
// ---------------------------------------------------------------------------

/**
 * CUDA-accelerated matrix transpose
 */
void cudaTranspose(const float *input, float *output, int rows, int cols)
{
    // Device memory allocation
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel with 2D grid
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    transposeKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

/**
 * CUDA transpose wrapper with availability checking
 */
void cuda_transpose(const float *input, float *output, int rows, int cols) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA transpose requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }
    
    if (cudaTranspose) {
        cudaTranspose(input, output, rows, cols);
    } else {
        fprintf(stderr, "Error: CUDA transpose function not linked\n");
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Host Interface Functions
// ============================================================================

// CUDA implementation functions are now called via weak symbols from wrapper
// The wrapper handles availability checking and error reporting

// ============================================================================
// CUDA RFI Detection Kernels and Functions
// ============================================================================

// Helper macros for CUDA memory management
#define CUDA_MALLOC(ptr, size, label) \
    do { \
        cudaStatus = cudaMalloc(ptr, size); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "cudaMalloc failed for " label ": %s\n", cudaGetErrorString(cudaStatus)); \
            result = -1; \
            goto cleanup; \
        } \
    } while(0)

#define CUDA_MEMCPY(dst, src, size, kind, label) \
    do { \
        cudaStatus = cudaMemcpy(dst, src, size, kind); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "cudaMemcpy failed for " label ": %s\n", cudaGetErrorString(cudaStatus)); \
            result = -1; \
            goto cleanup; \
        } \
    } while(0)

#define CUDA_FREE(ptr) \
    do { \
        if (ptr) cudaFree(ptr); \
    } while(0)

#define CUDA_KERNEL_CHECK(kernel_name) \
    do { \
        cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, kernel_name " failed: %s\n", cudaGetErrorString(cudaStatus)); \
            result = -1; \
            goto cleanup; \
        } \
    } while(0)

/**
 * CUDA kernel for in-channel (pixel-level) outlier detection
 * Each thread processes one pixel, computing local statistics
 */
__global__ void inChanDetectionKernel(const float *data, int *pointMask, 
                                    int nsamp, int nchan, float nSigma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = nsamp * nchan;
    
    if (idx >= total_pixels) return;
    
    // TODO: Implement pixel-level outlier detection
    // This would require computing local median/std for each pixel's neighborhood
    // For now, this is a placeholder
    
    pointMask[idx] = 0; // Placeholder: no pixels flagged
}

/**
 * CUDA kernel to compute median for each channel
 * Uses a simple selection algorithm (not optimal but functional)
 */
__global__ void computeChannelMedianKernel(const float *data, const int *mask, 
                                         float *medians, int nsamp, int nchan)
{
    int chan = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chan >= nchan) return;
    
    // Collect valid (unmasked) samples for this channel
    float valid_samples[4096]; // Assume max nsamp <= 4096
    int valid_count = 0;
    
    for (int s = 0; s < nsamp; s++) {
        int idx = chan * nsamp + s;
        if (mask[idx] == 0) { // Unmasked
            if (valid_count < 4096) {
                valid_samples[valid_count] = data[idx];
                valid_count++;
            }
        }
    }
    
    // Simple median calculation using selection
    if (valid_count > 0) {
        // Sort valid samples (simple bubble sort for small arrays)
        for (int i = 0; i < valid_count - 1; i++) {
            for (int j = 0; j < valid_count - i - 1; j++) {
                if (valid_samples[j] > valid_samples[j + 1]) {
                    float temp = valid_samples[j];
                    valid_samples[j] = valid_samples[j + 1];
                    valid_samples[j + 1] = temp;
                }
            }
        }
        
        // Get median
        if (valid_count % 2 == 0) {
            medians[chan] = (valid_samples[valid_count/2 - 1] + valid_samples[valid_count/2]) / 2.0f;
        } else {
            medians[chan] = valid_samples[valid_count/2];
        }
    } else {
        medians[chan] = 0.0f;
    }
}

/**
 * CUDA kernel to compute standard deviation from median for each channel
 */
__global__ void computeChannelStdFromMedianKernel(const float *data, const int *mask, 
                                                const float *medians, float *stds, 
                                                int nsamp, int nchan)
{
    int chan = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chan >= nchan) return;
    
    float median_val = medians[chan];
    float sum_squared_dev = 0.0f;
    int valid_count = 0;
    
    // Calculate sum of squared deviations from median
    for (int s = 0; s < nsamp; s++) {
        int idx = chan * nsamp + s;
        if (mask[idx] == 0) { // Unmasked
            float deviation = data[idx] - median_val;
            sum_squared_dev += deviation * deviation;
            valid_count++;
        }
    }
    
    // Calculate standard deviation
    if (valid_count > 0) {
        float mean_squared_dev = sum_squared_dev / valid_count;
        stds[chan] = sqrtf(mean_squared_dev);
    } else {
        stds[chan] = 0.0f;
    }
}

/**
 * CUDA kernel to flag outliers based on median and std bounds
 */
__global__ void flagOutliersKernel(const float *data, int *mask, 
                                 const float *medians, const float *stds,
                                 int nsamp, int nchan, float nSigma)
{
    int chan = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chan >= nchan) return;
    
    float median_val = medians[chan];
    float std_val = stds[chan];
    float upper_bound = median_val + nSigma * std_val;
    float lower_bound = median_val - nSigma * std_val;
    
    // Flag outliers
    for (int s = 0; s < nsamp; s++) {
        int idx = chan * nsamp + s;
        if (mask[idx] == 0) { // Only check currently unmasked pixels
            float val = data[idx];
            if (val > upper_bound || val < lower_bound) {
                mask[idx] = 1; // Flag as outlier
            }
        }
    }
}

/**
 * CUDA kernel for logical OR operation between two masks
 */
__global__ void logicalORKernel(const int *mask1, const int *mask2, int *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        result[idx] = mask1[idx] | mask2[idx];
    }
}

/**
 * CUDA kernel for channel-wise statistics computation
 * Excludes pixels marked in the mask from statistics calculation
 */
__global__ void channelStatsKernel(const float *data, const int *mask, 
                                 float *channel_means, float *channel_stds, 
                                 int nsamp, int nchan)
{
    int chan = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chan >= nchan) return;
    
    // Compute mean and std for this channel, excluding masked pixels
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int valid_count = 0;
    
    for (int samp = 0; samp < nsamp; samp++) {
        int idx = chan * nsamp + samp;
        if (mask[idx] == 0) {  // Only include unmasked pixels
            float val = data[idx];
            sum += val;
            sum_sq += val * val;
            valid_count++;
        }
    }
    
    float mean = 0.0f;
    float std = 0.0f;
    
    if (valid_count > 0) {
        mean = sum / valid_count;
        float variance = (sum_sq / valid_count) - (mean * mean);
        std = sqrtf(fmaxf(variance, 0.0f));
    }
    
    channel_means[chan] = mean;
    channel_stds[chan] = std;
}

/**
 * CUDA kernel to compute median of channel statistics (stds)
 * Uses a simple selection algorithm (not optimal but functional)
 */
__global__ void computeChannelStatsMedianKernel(const float *channel_stds, const int *channel_mask, 
                                               float *median_result, int nchan)
{
    // Only one thread does the work since we need a single median
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Collect valid (unmasked) channel stds
    float valid_stds[4096]; // Assume max nchan <= 4096
    int valid_count = 0;
    
    for (int c = 0; c < nchan; c++) {
        if (channel_mask[c] == 0) { // Valid channel
            if (valid_count < 4096) {
                valid_stds[valid_count] = channel_stds[c];
                valid_count++;
            }
        }
    }
    
    // Simple median calculation using selection
    if (valid_count > 0) {
        // Sort valid stds (simple bubble sort for small arrays)
        for (int i = 0; i < valid_count - 1; i++) {
            for (int j = 0; j < valid_count - i - 1; j++) {
                if (valid_stds[j] > valid_stds[j + 1]) {
                    float temp = valid_stds[j];
                    valid_stds[j] = valid_stds[j + 1];
                    valid_stds[j + 1] = temp;
                }
            }
        }
        
        // Get median
        if (valid_count % 2 == 0) {
            *median_result = (valid_stds[valid_count/2 - 1] + valid_stds[valid_count/2]) / 2.0f;
        } else {
            *median_result = valid_stds[valid_count/2];
        }
    } else {
        *median_result = 0.0f;
    }
}

/**
 * CUDA kernel to compute standard deviation from median of channel statistics
 */
__global__ void computeChannelStatsStdFromMedianKernel(const float *channel_stds, const int *channel_mask, 
                                                      const float *median_val, float *std_result, int nchan)
{
    // Only one thread does the work
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    float med = *median_val;
    float sum_squared_dev = 0.0f;
    int valid_count = 0;
    
    // Calculate sum of squared deviations from median
    for (int c = 0; c < nchan; c++) {
        if (channel_mask[c] == 0) { // Valid channel
            float deviation = channel_stds[c] - med;
            sum_squared_dev += deviation * deviation;
            valid_count++;
        }
    }
    
    // Calculate standard deviation
    if (valid_count > 0) {
        float mean_squared_dev = sum_squared_dev / valid_count;
        *std_result = sqrtf(mean_squared_dev);
    } else {
        *std_result = 0.0f;
    }
}

/**
 * CUDA kernel to flag outlier channels based on median and std bounds
 */
__global__ void flagOutlierChannelsKernel(const float *channel_stds, int *channel_mask, int *channel_flagged,
                                        const float *median_val, const float *std_val, 
                                        int nchan, float nSigma)
{
    int chan = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chan >= nchan) return;
    
    // Only process currently valid channels
    if (channel_mask[chan] != 0) return;
    
    float med = *median_val;
    float std = *std_val;
    float upper_bound = med + nSigma * std;
    float lower_bound = med - nSigma * std;
    float chan_std = channel_stds[chan];
    
    // Flag as outlier if outside bounds
    if (chan_std > upper_bound || chan_std < lower_bound) {
        channel_mask[chan] = 1;    // Mark as invalid for future iterations
        channel_flagged[chan] = 1; // Mark as flagged
    }
}

/**
 * CUDA kernel to expand 1D channel flags to 2D mask
 * If a channel is flagged, set all pixels in that channel to 1 in the mask
 */
__global__ void expandChannelMaskKernel(const int *channelFlagged, int *mask2D, 
                                       int nsamp, int nchan)
{
    int chan = blockIdx.x;
    if (chan >= nchan) return;
    
    // Check if this channel is flagged
    if (channelFlagged[chan]) {
        // Set all pixels in this channel to 1
        int base = chan * nsamp;
        for (int samp = threadIdx.x; samp < nsamp; samp += blockDim.x) {
            mask2D[base + samp] = 1;
        }
    }
}

/**
 * CUDA kernel to build lists of good (unmasked) sample indices for each channel
 * Output: good_samples_count[chan] = number of good samples in channel
 *         good_samples_indices[chan][0..count-1] = indices of good samples
 */
__global__ void buildGoodSamplesKernel(const int *mask, int *good_samples_count, 
                                     int *good_samples_indices, int nsamp, int nchan, 
                                     int max_good_per_chan)
{
    int chan = blockIdx.x;
    if (chan >= nchan) return;
    
    int base = chan * nsamp;
    int indices_base = chan * max_good_per_chan;
    
    // Initialize count for this channel
    if (threadIdx.x == 0) {
        good_samples_count[chan] = 0;
    }
    __syncthreads();
    
    // Each thread processes a subset of samples in the channel
    for (int s = threadIdx.x; s < nsamp; s += blockDim.x) {
        if (mask[base + s] == 0) {  // Good sample
            // Atomically increment count and add index
            int idx = atomicAdd(&good_samples_count[chan], 1);
            if (idx < max_good_per_chan) {
                good_samples_indices[indices_base + idx] = s;
            }
        }
    }
}

/**
 * CUDA kernel for pixel substitution using pre-built good samples lists
 */
__global__ void pixelSubstitutionKernel(float *data, const int *mask, 
                                      const int *good_samples_count, 
                                      const int *good_samples_indices,
                                      int nsamp, int nchan, int max_good_per_chan)
{
    int chan = blockIdx.x;
    if (chan >= nchan) return;
    
    int samp = threadIdx.x;
    if (samp >= nsamp) return;
    
    int idx = chan * nsamp + samp;
    
    if (mask[idx]) {
        // This pixel needs substitution
        int count = good_samples_count[chan];
        if (count > 0) {
            // Use deterministic pseudo-random selection
            unsigned int seed = chan * 12345 + samp * 6789 + 42;
            int selected = seed % count;
            
            int indices_base = chan * max_good_per_chan;
            int source_samp = good_samples_indices[indices_base + selected];
            int source_idx = chan * nsamp + source_samp;
            
            // Replace with the selected good pixel
            data[idx] = data[source_idx];
        }
    }
}

/**
 * CUDA kernel for cross-channel substitution
 * Replaces entire flagged channels with random samples from unflagged channels at each time sample
 */
__global__ void crossChannelSubstitutionKernel(float *data, const int *channelMask, 
                                             const int *pointMask, int nsamp, int nchan)
{
    int samp = blockIdx.x * blockDim.x + threadIdx.x;
    if (samp >= nsamp) return;
    
    // Shared memory for collecting source values from unflagged channels
    extern __shared__ float shared_source_values[];
    int *shared_source_count = (int*)&shared_source_values[nchan];
    
    // Initialize shared count
    if (threadIdx.x == 0) {
        *shared_source_count = 0;
    }
    __syncthreads();
    
    // Each thread in the block processes one channel for this time sample
    int chan = threadIdx.x;
    if (chan < nchan) {
        int idx = chan * nsamp + samp;
        
        // If channel is not flagged and pixel is not masked, add to source values
        if (channelMask[chan] == 0 && (!pointMask || pointMask[idx] == 0)) {
            int source_idx = atomicAdd(shared_source_count, 1);
            if (source_idx < nchan) {
                shared_source_values[source_idx] = data[idx];
            }
        }
    }
    __syncthreads();
    
    // Now replace flagged channels with random selection from source values
    if (chan < nchan) {
        int idx = chan * nsamp + samp;
        
        if (channelMask[chan] != 0) {
            // This channel needs cross-channel substitution
            int source_count = *shared_source_count;
            
            if (source_count > 0) {
                // Use deterministic pseudo-random selection
                unsigned int seed = samp * 1315423911u + chan * 12345u + 42;
                int random_idx = seed % source_count;
                data[idx] = shared_source_values[random_idx];
            } else {
                // No source channels available, set to zero
                data[idx] = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA RFI Detection Main Function
// ---------------------------------------------------------------------------

/**
 * CUDA-accelerated RFI detection using N-sigma thresholding
 * This is a framework implementation - many components are placeholders
 */
int cudaIdentSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan,
    int iterationIndex, int plot,
    IdentNSigmaMasks *masks,
    float *finalMedian, float *finalStd, int *flaggedChans)
{
    // Device memory pointers
    float *d_data = NULL;
    int *d_pointMask = NULL;
    int *d_horizontalMask = NULL;
    int *d_globalMask = NULL;
    int *d_flaggedChans = NULL;
    float *d_channel_means = NULL;
    float *d_channel_stds = NULL;
    float *d_channel_medians = NULL;  // For in-channel detection
    float *d_channel_std_from_median = NULL;  // For in-channel detection
    int *d_good_samples_count = NULL;
    int *d_good_samples_indices = NULL;
    
    cudaError_t cudaStatus;
    int result = 0;
    
    // Calculate sizes
    size_t data_size = (size_t)nsamp * nchan * sizeof(float);
    size_t mask_size = (size_t)nsamp * nchan * sizeof(int);
    size_t chan_size = (size_t)nchan * sizeof(int);
    size_t chan_stats_size = (size_t)nchan * sizeof(float);
    
    // Calculate good samples list sizes
    int max_good_per_chan = nsamp;  // Conservative estimate
    size_t good_count_size = (size_t)nchan * sizeof(int);
    size_t good_indices_size = (size_t)nchan * max_good_per_chan * sizeof(int);
    
    printf("=== CUDA RFI Detection (Iteration %d) ===\n", iterationIndex);
    printf("Data size: %d x %d = %zu elements\n", nsamp, nchan, (size_t)nsamp * nchan);
    
    // Allocate device memory
    CUDA_MALLOC(&d_data, data_size, "data");
    CUDA_MALLOC(&d_pointMask, mask_size, "pointMask");
    CUDA_MALLOC(&d_horizontalMask, mask_size, "horizontalMask");
    CUDA_MALLOC(&d_globalMask, mask_size, "globalMask");
    CUDA_MALLOC(&d_flaggedChans, chan_size, "flaggedChans");
    CUDA_MALLOC(&d_channel_means, chan_stats_size, "channel_means");
    CUDA_MALLOC(&d_channel_stds, chan_stats_size, "channel_stds");
    CUDA_MALLOC(&d_channel_medians, chan_stats_size, "channel_medians");
    CUDA_MALLOC(&d_channel_std_from_median, chan_stats_size, "channel_std_from_median");
    CUDA_MALLOC(&d_good_samples_count, good_count_size, "good_samples_count");
    CUDA_MALLOC(&d_good_samples_indices, good_indices_size, "good_samples_indices");
    
    // Copy input data to device
    CUDA_MEMCPY(d_data, data, data_size, cudaMemcpyHostToDevice, "input data");
    
    // Initialize masks on device
    cudaStatus = cudaMemset(d_pointMask, 0, mask_size);
    cudaStatus = cudaMemset(d_horizontalMask, 0, mask_size);
    cudaStatus = cudaMemset(d_globalMask, 0, mask_size);
    cudaStatus = cudaMemset(d_flaggedChans, 0, chan_size);
    
    // === 1. inChannel Detection ===
    printf("=== CUDA inChannel Detection ===\n");
    {
        const int MAX_ITERATIONS = 15;
        const float STD_CHANGE_THRESHOLD = 0.0001f;
        const float MEDIAN_CHANGE_THRESHOLD = 1e-6f;
        
        int total_pixel_outliers = 0;
        
        // Initialize previous values for convergence checking
        float *h_prev_medians = (float *)malloc(chan_stats_size);
        float *h_prev_stds = (float *)malloc(chan_stats_size);
        memset(h_prev_medians, 0, chan_stats_size);
        memset(h_prev_stds, 0, chan_stats_size);
        
        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            printf("  Iteration %d:\n", iter + 1);
            
            // Step 1: Compute median for each channel
            {
                int blockSize = 256;
                int gridSize = (nchan + blockSize - 1) / blockSize;
                
                computeChannelMedianKernel<<<gridSize, blockSize>>>(
                    d_data, d_pointMask, d_channel_medians, nsamp, nchan);
                CUDA_KERNEL_CHECK("computeChannelMedianKernel");
            }
            
            // Step 2: Compute std from median for each channel
            {
                int blockSize = 256;
                int gridSize = (nchan + blockSize - 1) / blockSize;
                
                computeChannelStdFromMedianKernel<<<gridSize, blockSize>>>(
                    d_data, d_pointMask, d_channel_medians, d_channel_std_from_median, 
                    nsamp, nchan);
                CUDA_KERNEL_CHECK("computeChannelStdFromMedianKernel");
            }
            
            // Copy current statistics to host for convergence checking
            float *h_current_medians = (float *)malloc(chan_stats_size);
            float *h_current_stds = (float *)malloc(chan_stats_size);
            cudaMemcpy(h_current_medians, d_channel_medians, chan_stats_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_current_stds, d_channel_std_from_median, chan_stats_size, cudaMemcpyDeviceToHost);
            
            // Step 3: Flag outliers based on current statistics
            int iter_outliers = 0;
            {
                int blockSize = 256;
                int gridSize = (nchan + blockSize - 1) / blockSize;
                
                flagOutliersKernel<<<gridSize, blockSize>>>(
                    d_data, d_pointMask, d_channel_medians, d_channel_std_from_median,
                    nsamp, nchan, NSigmaInChan);
                CUDA_KERNEL_CHECK("flagOutliersKernel");
                
                // Count newly flagged pixels
                int *h_mask = (int *)malloc(mask_size);
                cudaMemcpy(h_mask, d_pointMask, mask_size, cudaMemcpyDeviceToHost);
                
                for (int i = 0; i < nsamp * nchan; i++) {
                    if (h_mask[i]) iter_outliers++;
                }
                free(h_mask);
            }
            
            // Check convergence (after first iteration)
            int converged = 0;
            if (iter > 0) {
                float max_median_change = 0.0f;
                float max_std_change_rate = 0.0f;
                
                for (int c = 0; c < nchan; c++) {
                    float median_change = fabsf(h_current_medians[c] - h_prev_medians[c]);
                    max_median_change = fmaxf(max_median_change, median_change);
                    
                    if (h_prev_stds[c] > 0) {
                        float std_change_rate = fabsf(h_current_stds[c] - h_prev_stds[c]) / h_prev_stds[c];
                        max_std_change_rate = fmaxf(max_std_change_rate, std_change_rate);
                    }
                }
                
                if (max_median_change < MEDIAN_CHANGE_THRESHOLD && max_std_change_rate < STD_CHANGE_THRESHOLD) {
                    converged = 1;
                }
                
                printf("    Convergence: median_change=%.8f, std_change_rate=%.6f, converged=%d\n",
                       max_median_change, max_std_change_rate, converged);
            }
            
            printf("    Flagged %d pixels in this iteration\n", iter_outliers);
            total_pixel_outliers += iter_outliers;
            
            // Update previous values
            memcpy(h_prev_medians, h_current_medians, chan_stats_size);
            memcpy(h_prev_stds, h_current_stds, chan_stats_size);
            
            free(h_current_medians);
            free(h_current_stds);
            
            // Check stopping conditions
            if (converged || iter_outliers == 0) {
                printf("    Converged after %d iterations\n", iter + 1);
                break;
            }
        }
        
        free(h_prev_medians);
        free(h_prev_stds);
        
        printf("CUDA inChannel detection: flagged %d outlier pixels total\n", total_pixel_outliers);
    }
    
    // === 2. Pixel Substitution ===
    printf("=== CUDA Pixel Substitution ===\n");
    
    // First, build good samples lists for each channel
    printf("  - Building good samples lists...\n");
    {
        dim3 blockSize(256);  // Each block processes one channel
        dim3 gridSize(nchan);
        
        // Initialize good samples count to 0
        cudaStatus = cudaMemset(d_good_samples_count, 0, good_count_size);
        
        buildGoodSamplesKernel<<<gridSize, blockSize>>>(d_pointMask, d_good_samples_count,
                                                       d_good_samples_indices, nsamp, nchan, 
                                                       max_good_per_chan);
        CUDA_KERNEL_CHECK("buildGoodSamplesKernel");
    }
    
    // Then perform pixel substitution using the pre-built lists
    {
        dim3 blockSize(nsamp);  // One thread per sample in channel
        dim3 gridSize(nchan);   // One block per channel
        
        pixelSubstitutionKernel<<<gridSize, blockSize>>>(d_data, d_pointMask, 
                                                        d_good_samples_count, d_good_samples_indices,
                                                        nsamp, nchan, max_good_per_chan);
        CUDA_KERNEL_CHECK("pixelSubstitutionKernel");
        
        // Count substituted pixels (simple implementation)
        int *h_mask = (int *)malloc(mask_size);
        cudaMemcpy(h_mask, d_pointMask, mask_size, cudaMemcpyDeviceToHost);
        
        int substitutedPixels = 0;
        for (int i = 0; i < nsamp * nchan; i++) {
            if (h_mask[i]) substitutedPixels++;
        }
        
        printf("CUDA pixel substitution: replaced %d pixels\n", substitutedPixels);
        free(h_mask);
    }
    
    // === 3. Logical OR: pointMask -> globalMask ===
    {
        int blockSize = 256;
        int gridSize = (nsamp * nchan + blockSize - 1) / blockSize;
        
        logicalORKernel<<<gridSize, blockSize>>>(d_globalMask, d_pointMask, 
                                                d_globalMask, nsamp * nchan);
    }
    
    // === 4. Channel Statistics Computation ===
    printf("=== CUDA Channel Statistics ===\n");
    {
        int blockSize = 256;
        int gridSize = (nchan + blockSize - 1) / blockSize;
        
        channelStatsKernel<<<gridSize, blockSize>>>(d_data, d_pointMask, d_channel_means, 
                                                   d_channel_stds, nsamp, nchan);
        CUDA_KERNEL_CHECK("channelStatsKernel");
    }
    
    // === 5. outChannel Detection ===
    printf("=== CUDA outChannel Detection ===\n");
    {
        const int MAX_ITERATIONS = 100;
        const float STD_CHANGE_THRESHOLD = 0.0001f;
        const float MEDIAN_CHANGE_THRESHOLD = 1e-6f;
        
        // Device memory for channel mask (0=valid, 1=flagged)
        int *d_channel_mask = NULL;
        float *d_current_median = NULL;
        float *d_current_std = NULL;
        
        CUDA_MALLOC(&d_channel_mask, chan_size, "channel_mask");
        CUDA_MALLOC(&d_current_median, sizeof(float), "current_median");
        CUDA_MALLOC(&d_current_std, sizeof(float), "current_std");
        
        // Initialize channel mask to all valid (0)
        cudaStatus = cudaMemset(d_channel_mask, 0, chan_size);
        
        int total_flagged = 0;
        int valid_count = nchan; // Start with all channels valid
        
        // Initialize previous values for convergence checking
        float h_prev_median = 0.0f;
        float h_prev_std = 0.0f;
        
        for (int iter = 0; iter < MAX_ITERATIONS && valid_count >= 3; iter++) {
            printf("  Iteration %d: ", iter + 1);
            
            // Step 1: Compute median of current valid channel stds
            {
                dim3 blockSize(1);
                dim3 gridSize(1);
                
                computeChannelStatsMedianKernel<<<gridSize, blockSize>>>(
                    d_channel_stds, d_channel_mask, d_current_median, nchan);
                CUDA_KERNEL_CHECK("computeChannelStatsMedianKernel");
            }
            
            // Step 2: Compute std from median of current valid channel stds
            {
                dim3 blockSize(1);
                dim3 gridSize(1);
                
                computeChannelStatsStdFromMedianKernel<<<gridSize, blockSize>>>(
                    d_channel_stds, d_channel_mask, d_current_median, d_current_std, nchan);
                CUDA_KERNEL_CHECK("computeChannelStatsStdFromMedianKernel");
            }
            
            // Copy current statistics to host for convergence checking
            float h_current_median, h_current_std;
            cudaMemcpy(&h_current_median, d_current_median, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_current_std, d_current_std, sizeof(float), cudaMemcpyDeviceToHost);
            
            // Step 3: Flag outlier channels
            int iter_flagged = 0;
            {
                int blockSize = 256;
                int gridSize = (nchan + blockSize - 1) / blockSize;
                
                flagOutlierChannelsKernel<<<gridSize, blockSize>>>(
                    d_channel_stds, d_channel_mask, d_flaggedChans,
                    d_current_median, d_current_std, nchan, NSigmaOutChan);
                CUDA_KERNEL_CHECK("flagOutlierChannelsKernel");
                
                // Count newly flagged channels
                int *h_channel_mask = (int *)malloc(chan_size);
                cudaMemcpy(h_channel_mask, d_channel_mask, chan_size, cudaMemcpyDeviceToHost);
                
                for (int i = 0; i < nchan; i++) {
                    if (h_channel_mask[i]) iter_flagged++;
                }
                free(h_channel_mask);
            }
            
            // Update valid count
            valid_count = nchan - iter_flagged;
            total_flagged = iter_flagged;
            
            // Check convergence (after first iteration)
            int converged = 0;
            if (iter > 0) {
                float median_change = fabsf(h_current_median - h_prev_median);
                float std_change_rate = (h_prev_std > 0) ? fabsf(h_current_std - h_prev_std) / h_prev_std : 0.0f;
                
                if (median_change < MEDIAN_CHANGE_THRESHOLD && std_change_rate < STD_CHANGE_THRESHOLD) {
                    converged = 1;
                }
                
                printf("median=%.6f, std=%.6f, med_change=%.8f, std_change_rate=%.6f, flagged=%d, remaining=%d/%d (%.1f%%)",
                       h_current_median, h_current_std, median_change, std_change_rate, 
                       iter_flagged, valid_count, nchan, (float)valid_count/nchan*100);
                
                if (converged) {
                    printf(" -> CONVERGED\n");
                } else {
                    printf("\n");
                }
            } else {
                printf("median=%.6f, std=%.6f, flagged=%d, remaining=%d/%d (%.1f%%)\n",
                       h_current_median, h_current_std, iter_flagged, valid_count, nchan, (float)valid_count/nchan*100);
            }
            
            // Update previous values
            h_prev_median = h_current_median;
            h_prev_std = h_current_std;
            
            // Check stopping conditions
            if (converged || iter_flagged == 0) {
                printf("  Converged after %d iterations\n", iter + 1);
                break;
            }
        }
        
        printf("CUDA outChannel detection: flagged %d channels total\n", total_flagged);
        
        // Cleanup
        CUDA_FREE(d_channel_mask);
        CUDA_FREE(d_current_median);
        CUDA_FREE(d_current_std);
    }
    
    // === 6. Expand channel mask to 2D ===
    printf("=== CUDA Expand Channel Mask ===\n");
    {
        dim3 blockSize(256);  // Each block processes one channel
        dim3 gridSize(nchan);
        
        expandChannelMaskKernel<<<gridSize, blockSize>>>(d_flaggedChans, d_horizontalMask, 
                                                        nsamp, nchan);
        CUDA_KERNEL_CHECK("expandChannelMaskKernel");
        
        // Count flagged channels and expanded pixels
        int *h_flaggedChans = (int *)malloc(chan_size);
        cudaMemcpy(h_flaggedChans, d_flaggedChans, chan_size, cudaMemcpyDeviceToHost);
        
        int flaggedChannels = 0;
        for (int i = 0; i < nchan; i++) {
            if (h_flaggedChans[i]) flaggedChannels++;
        }
        
        printf("CUDA expand channel mask: flagged %d channels (%d pixels total)\n", 
               flaggedChannels, flaggedChannels * nsamp);
        free(h_flaggedChans);
    }
    
    // === 7. Logical OR: horizontalMask -> globalMask ===
    {
        int blockSize = 256;
        int gridSize = (nsamp * nchan + blockSize - 1) / blockSize;
        
        logicalORKernel<<<gridSize, blockSize>>>(d_globalMask, d_horizontalMask, 
                                                d_globalMask, nsamp * nchan);
    }
    
    // === 8. Cross-channel substitution for flagged channels ===
    printf("=== CUDA Cross-Channel Substitution ===\n");
    {
        // Use one block per time sample, with threads for each channel
        int blockSize = nchan;  // One thread per channel
        int gridSize = (nsamp + blockSize - 1) / blockSize;  // One block per time sample
        
        // Shared memory: nchan floats for source values + 1 int for count
        size_t sharedMemSize = nchan * sizeof(float) + sizeof(int);
        
        crossChannelSubstitutionKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_data, d_flaggedChans, d_pointMask, nsamp, nchan);
        CUDA_KERNEL_CHECK("crossChannelSubstitutionKernel");
        
        // Count substituted channels
        int *h_flaggedChans = (int *)malloc(chan_size);
        cudaMemcpy(h_flaggedChans, d_flaggedChans, chan_size, cudaMemcpyDeviceToHost);
        
        int substitutedChannels = 0;
        for (int i = 0; i < nchan; i++) {
            if (h_flaggedChans[i]) substitutedChannels++;
        }
        
        printf("CUDA cross-channel substitution: processed %d flagged channels\n", substitutedChannels);
        free(h_flaggedChans);
    }
    
    // Copy results back to host
    CUDA_MEMCPY(data, d_data, data_size, cudaMemcpyDeviceToHost, "output data");
    CUDA_MEMCPY(masks->pointMask, d_pointMask, mask_size, cudaMemcpyDeviceToHost, "pointMask");
    CUDA_MEMCPY(masks->horizontalMask, d_horizontalMask, mask_size, cudaMemcpyDeviceToHost, "horizontalMask");
    CUDA_MEMCPY(masks->globalMask, d_globalMask, mask_size, cudaMemcpyDeviceToHost, "globalMask");
    CUDA_MEMCPY(flaggedChans, d_flaggedChans, chan_size, cudaMemcpyDeviceToHost, "flaggedChans");
    
    // TODO: Compute final statistics
    *finalMedian = 0.0f; // Placeholder
    *finalStd = 1.0f;    // Placeholder
    
    printf("CUDA RFI detection completed\n");
    
cleanup:
    // Free device memory
    CUDA_FREE(d_data);
    CUDA_FREE(d_pointMask);
    CUDA_FREE(d_horizontalMask);
    CUDA_FREE(d_globalMask);
    CUDA_FREE(d_flaggedChans);
    CUDA_FREE(d_channel_means);
    CUDA_FREE(d_channel_stds);
    CUDA_FREE(d_channel_medians);
    CUDA_FREE(d_channel_std_from_median);
    CUDA_FREE(d_good_samples_count);
    CUDA_FREE(d_good_samples_indices);
    
    return result;
}

/**
 * CUDA identSubstNSigma wrapper with availability checking
 */
int cuda_identSubstNSigma(
    float *data, int nsamp, int nchan,
    float NSigmaInChan, float NSigmaOutChan,
    int iterationIndex, int plot,
    void *masks,
    float *finalMedian, float *finalStd, int *flaggedChans)
{
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA RFI detection requested but CUDA is not available\n");
        return -1;
    }

    printf("=== Testing CUDA-accelerated RFI detection ===\n");
    double start_time = omp_get_wtime();

    int result = cudaIdentSubstNSigma(data, nsamp, nchan, NSigmaInChan, NSigmaOutChan,
                                      iterationIndex, plot, (IdentNSigmaMasks*)masks,
                                      finalMedian, finalStd, flaggedChans);

    double elapsed = omp_get_wtime() - start_time;
    if (result == 0) {
        printf("CUDA RFI detection completed successfully in %.4f seconds\n", elapsed);
    } else {
        printf("CUDA RFI detection failed with code %d\n", result);
    }

    return result;
}

} // extern "C"
