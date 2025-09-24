#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Include header for CUDA functions
#include "cuda_acceleration.h"

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
// CUDA Kernel Implementations
// ============================================================================

/**
 * CUDA kernel for subtracting channel medians
 * Each thread processes one data element
 */
__global__ void subtractChannelMediansKernel(float *data, const float *channel_medians, 
                                            int nsamp, int nchan)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nsamp * nchan;
    
    if (idx < total_size) {
        int channel = idx / nsamp;  // Which channel this element belongs to
        data[idx] -= channel_medians[channel];
    }
}

/**
 * CUDA kernel for calculating mean and standard deviation per channel
 * Uses shared memory for reduction within each block
 */
__global__ void channelStatsKernel(const float *data, float *means, float *stds, 
                                 int nsamp, int nchan)
{
    int channel = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sum_sq = sdata + blockSize;
    
    if (channel >= nchan) return;
    
    const float *channel_data = data + channel * nsamp;
    
    // Initialize shared memory
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    
    // Each thread processes multiple elements if needed
    for (int i = tid; i < nsamp; i += blockSize) {
        float value = channel_data[i];
        s_sum[tid] += value;
        s_sum_sq[tid] += value * value;
    }
    
    __syncthreads();
    
    // Reduction within block
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        float mean = s_sum[0] / nsamp;
        float variance = (s_sum_sq[0] - nsamp * mean * mean) / nsamp;
        means[channel] = mean;
        stds[channel] = sqrtf(fmaxf(variance, 0.0f)); // Ensure non-negative
    }
}

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

/**
 * CUDA kernel for 2D downsampling
 * Each thread processes one output element
 */
__global__ void downsample2DKernel(const float *input, float *output,
                                 int nsamp, int nchan,
                                 int binFactorTime, int binFactorFreq,
                                 int nsampBinned, int nchanBinned)
{
    int out_i = blockIdx.x * blockDim.x + threadIdx.x; // time index in output
    int out_j = blockIdx.y * blockDim.y + threadIdx.y; // freq index in output
    
    if (out_i >= nsampBinned || out_j >= nchanBinned) return;
    
    float sum = 0.0f;
    
    // Sum over the bin
    for (int ti = 0; ti < binFactorTime; ti++) {
        for (int fj = 0; fj < binFactorFreq; fj++) {
            int in_i = out_i * binFactorTime + ti;
            int in_j = out_j * binFactorFreq + fj;
            
            if (in_i < nsamp && in_j < nchan) {
                sum += input[in_i * nchan + in_j];
            }
        }
    }
    
    output[out_i * nchanBinned + out_j] = sum / (binFactorTime * binFactorFreq);
}

// ---------------------------------------------------------------------------
// In-channel thresholding: flag samples where |x - mean[channel]| > Nsigma * std[channel]
// Assumes data is channel-major: each channel occupies a contiguous block of nsamp
__global__ void inChanThresholdKernel(const float *data, const float *means, const float *stds,
                                      int nsamp, int nchan, float Nsigma, int *mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsamp * nchan;
    if (idx >= total) return;

    int ch = idx / nsamp;
    int off = idx % nsamp;
    float mu = means[ch];
    float sd = stds[ch];
    float thr = Nsigma * sd;
    float v = data[ch * nsamp + off];
    if (!(sd > 0.0f)) { mask[ch * nsamp + off] = 0; return; }
    mask[ch * nsamp + off] = (fabsf(v - mu) > thr) ? 1 : 0;
}

// Expand per-channel mask (0/1 per channel) to 2D mask (channel-major)
__global__ void expandChannelMask2DKernel(const int *chan_mask, int nchan, int nsamp, int *mask2d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsamp * nchan;
    if (idx >= total) return;
    int ch = idx / nsamp;
    mask2d[idx] = chan_mask[ch] ? 1 : 0;
}

// ============================================================================
// Host Interface Functions
// ============================================================================

/**
 * CUDA-accelerated channel median subtraction
 */
void cuda_subtractChannelMedians_impl(float *data, const float *channel_medians, 
                                int nsamp, int nchan)
{
    int total_size = nsamp * nchan;
    
    // Device memory allocation
    float *d_data, *d_medians;
    CUDA_CHECK(cudaMalloc(&d_data, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_medians, nchan * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, data, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_medians, channel_medians, nchan * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;
    
    subtractChannelMediansKernel<<<gridSize, blockSize>>>(d_data, d_medians, nsamp, nchan);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(data, d_data, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_medians));
}

/**
 * CUDA-accelerated channel statistics calculation
 */
void cuda_calculateChannelStats_impl(const float *data, float *means, float *stds, 
                               int nsamp, int nchan)
{
    int total_size = nsamp * nchan;
    
    // Device memory allocation
    float *d_data, *d_means, *d_stds;
    CUDA_CHECK(cudaMalloc(&d_data, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, nchan * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stds, nchan * sizeof(float)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_data, data, total_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel - one block per channel
    int blockSize = 256; // threads per block
    int sharedMemSize = 2 * blockSize * sizeof(float); // for sum and sum_sq
    
    channelStatsKernel<<<nchan, blockSize, sharedMemSize>>>(d_data, d_means, d_stds, nsamp, nchan);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(means, d_means, nchan * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stds, d_stds, nchan * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_stds));
}

/**
 * CUDA-accelerated matrix transpose
 */
void cuda_transpose_impl(const float *input, float *output, int rows, int cols)
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
 * CUDA-accelerated 2D downsampling
 */
void cuda_downsample2D_impl(const float *input, float *output, 
                      int nsamp, int nchan,
                      int binFactorTime, int binFactorFreq)
{
    int nsampBinned = nsamp / binFactorTime;
    int nchanBinned = nchan / binFactorFreq;
    
    // Device memory allocation
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, nsamp * nchan * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, nsampBinned * nchanBinned * sizeof(float)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, nsamp * nchan * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel with 2D grid
    dim3 blockSize(16, 16);
    dim3 gridSize((nsampBinned + blockSize.x - 1) / blockSize.x,
                  (nchanBinned + blockSize.y - 1) / blockSize.y);
    
    downsample2DKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                               nsamp, nchan,
                                               binFactorTime, binFactorFreq,
                                               nsampBinned, nchanBinned);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, nsampBinned * nchanBinned * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// ---------------------------------------------------------------------------
// Public (not yet integrated) helpers for in-channel and out-channel detection

// In-channel outlier detection using per-channel mean/std and Nsigma threshold
// data: channel-major layout (ch * nsamp + t)
// mask_out: host pointer, size nsamp*nchan, 1 for outlier sample
void cuda_inChanThreshold_impl(const float *data, int nsamp, int nchan, float Nsigma,
                               int *mask_out)
{
    int total = nsamp * nchan;
    if (nsamp <= 0 || nchan <= 0 || !data || !mask_out) {
        fprintf(stderr, "cuda_inChanThreshold_impl: invalid input parameters\n");
        return;
    }

    float *d_data = nullptr, *d_means = nullptr, *d_stds = nullptr;
    int *d_mask = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, nchan * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stds, nchan * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, total * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, data, total * sizeof(float), cudaMemcpyHostToDevice));

    // Reuse existing kernel: one block per channel
    {
        int blockSize = 256;
        int sharedMemSize = 2 * blockSize * sizeof(float);
        channelStatsKernel<<<nchan, blockSize, sharedMemSize>>>(d_data, d_means, d_stds, nsamp, nchan);
        CUDA_CHECK(cudaGetLastError());
    }

    // Threshold per element
    {
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        inChanThresholdKernel<<<gridSize, blockSize>>>(d_data, d_means, d_stds, nsamp, nchan, Nsigma, d_mask);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(mask_out, d_mask, total * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_stds));
    CUDA_CHECK(cudaFree(d_mask));
}

// Out-channel detection via IQR on per-channel standard deviations (host-side IQR)
// channel_mask_out: host pointer, size nchan, 1 for flagged channel
void cuda_outChanIQR_impl(const float *data, int nsamp, int nchan, float q,
                          int *channel_mask_out)
{
    if (nsamp <= 0 || nchan <= 0 || !data || !channel_mask_out) {
        fprintf(stderr, "cuda_outChanIQR_impl: invalid input parameters\n");
        return;
    }

    int total = nsamp * nchan;
    float *d_data = nullptr, *d_means = nullptr, *d_stds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, nchan * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stds, nchan * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, total * sizeof(float), cudaMemcpyHostToDevice));

    // Per-channel stats on GPU
    {
        int blockSize = 256;
        int sharedMemSize = 2 * blockSize * sizeof(float);
        channelStatsKernel<<<nchan, blockSize, sharedMemSize>>>(d_data, d_means, d_stds, nsamp, nchan);
        CUDA_CHECK(cudaGetLastError());
    }

    // Copy stds back to host for IQR (robust and simple on host)
    float *h_stds = (float*)malloc(nchan * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_stds, d_stds, nchan * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute Q1/Q3 on host
    float *sorted = (float*)malloc(nchan * sizeof(float));
    memcpy(sorted, h_stds, nchan * sizeof(float));
    // simple qsort comparator
    auto cmp = [](const void* a, const void* b){
        float fa = *(const float*)a, fb = *(const float*)b;
        return (fa > fb) - (fa < fb);
    };
    qsort(sorted, nchan, sizeof(float), cmp);
    auto pct_idx = [&](float p){
        int idx = (int)(p * (nchan - 1));
        if (idx < 0) idx = 0; if (idx >= nchan) idx = nchan - 1; return idx;
    };
    float q1 = sorted[pct_idx(0.25f)];
    float q3 = sorted[pct_idx(0.75f)];
    float iqr = q3 - q1;
    float vmin = q1 - q * iqr;
    float vmax = q3 + q * iqr;

    for (int ch = 0; ch < nchan; ++ch) {
        float s = h_stds[ch];
        channel_mask_out[ch] = (s < vmin || s > vmax) ? 1 : 0;
    }

    free(sorted);
    free(h_stds);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_stds));
}

// Expand per-channel mask to 2D mask on GPU (channel-major)
void cuda_expandChannelMask2D_impl(const int *channel_mask, int nchan, int nsamp, int *mask2d_out)
{
    if (!channel_mask || !mask2d_out || nchan <= 0 || nsamp <= 0) {
        fprintf(stderr, "cuda_expandChannelMask2D_impl: invalid input parameters\n");
        return;
    }
    int total = nchan * nsamp;
    int *d_chan = nullptr, *d_mask2d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chan, nchan * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask2d, total * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_chan, channel_mask, nchan * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    expandChannelMask2DKernel<<<gridSize, blockSize>>>(d_chan, nchan, nsamp, d_mask2d);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(mask2d_out, d_mask2d, total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_chan));
    CUDA_CHECK(cudaFree(d_mask2d));
}

// ============================================================================
// Binary Morphological Filtering (binarySIR) CUDA Implementation
// ============================================================================

/**
 * CUDA kernel for binary morphological filtering with window-based neighbor counting
 * Each thread processes one pixel in the 2D mask
 */
__global__ void binarySIRKernel(int *mask, int nsamp, int nchan,
                               int win_samp, int win_chan,
                               float thr_up, float thr_down)
{
    // Calculate global thread position
    int i = blockIdx.x * blockDim.x + threadIdx.x; // time sample index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // channel index
    
    // Check bounds
    if (i >= nsamp || j >= nchan) return;
    
    // Calculate window radii
    const int rad_samp = win_samp / 2;
    const int rad_chan = win_chan / 2;
    
    int count = 0, total = 0;
    
    // Count neighbors in the window
    for (int dj = -rad_chan; dj <= rad_chan; dj++) {
        int jj = j + dj;
        if (jj < 0 || jj >= nchan) continue;
        
        for (int di = -rad_samp; di <= rad_samp; di++) {
            int ii = i + di;
            if (ii < 0 || ii >= nsamp) continue;
            
            // Check if neighbor is flagged (non-zero)
            if (mask[jj * nsamp + ii] != 0) count++;
            total++;
        }
    }
    
    // Apply threshold-based decision
    if (total > 0) {
        float ratio = (float)count / total;
        if (ratio >= thr_up) {
            mask[j * nsamp + i] = 1;
        } else if (ratio < thr_down) {
            mask[j * nsamp + i] = 0;
        }
        // If ratio is between thresholds, keep original value
    }
}

/**
 * Host function for CUDA-accelerated binarySIR morphological filtering
 */
void cuda_binarySIR_impl(int *mask, int nsamp, int nchan,
                        int win_samp, int win_chan, 
                        float thr_up, float thr_down)
{
    if (!cuda_isAvailable()) {
        printf("Error: CUDA not available for binarySIR\n");
        return;
    }
    
    // Validate window sizes (must be odd)
    if (((win_samp | win_chan) & 1) == 0) {
        printf("Error: Window sizes must be odd for binarySIR\n");
        return;
    }
    
    // Count pixels before filtering for statistics
    int pixelsBefore = 0;
    for (int idx = 0; idx < nsamp * nchan; idx++) {
        if (mask[idx] != 0) pixelsBefore++;
    }
    
    size_t maskSize = nsamp * nchan * sizeof(int);
    int *d_mask;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_mask, maskSize));
    
    // Copy mask to device
    CUDA_CHECK(cudaMemcpy(d_mask, mask, maskSize, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((nsamp + blockSize.x - 1) / blockSize.x,
                  (nchan + blockSize.y - 1) / blockSize.y);
    
    printf("Launching binarySIR CUDA kernel: grid(%d,%d), block(%d,%d)\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    printf("Processing %dx%d mask with %dx%d window (thresholds: %.3f/%.3f)\n",
           nsamp, nchan, win_samp, win_chan, thr_up, thr_down);
    
    // Launch kernel
    binarySIRKernel<<<gridSize, blockSize>>>(d_mask, nsamp, nchan,
                                            win_samp, win_chan,
                                            thr_up, thr_down);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(mask, d_mask, maskSize, cudaMemcpyDeviceToHost));
    
    // Count pixels after filtering for statistics
    int pixelsAfter = 0;
    for (int idx = 0; idx < nsamp * nchan; idx++) {
        if (mask[idx] != 0) pixelsAfter++;
    }
    
    printf("CUDA binarySIR filtering statistics:\n");
    printf("  - Window size: %dx%d (samples x channels)\n", win_samp, win_chan);
    printf("  - Thresholds: up=%.3f, down=%.3f\n", thr_up, thr_down);
    printf("  - Pixels before: %d/%d (%.4f%%)\n", 
           pixelsBefore, nsamp*nchan, (float)pixelsBefore/(nsamp*nchan)*100);
    printf("  - Pixels after: %d/%d (%.4f%%)\n", 
           pixelsAfter, nsamp*nchan, (float)pixelsAfter/(nsamp*nchan)*100);
    printf("  - Filtered out: %d pixels (%.4f%%)\n", 
           pixelsBefore - pixelsAfter, (float)(pixelsBefore - pixelsAfter)/(nsamp*nchan)*100);
    printf("  - Reduction ratio: %.2fx\n", 
           pixelsBefore > 0 ? (float)pixelsBefore/pixelsAfter : 0.0f);
    
    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_mask));
}

// CUDA implementation functions are now called via weak symbols from wrapper
// The wrapper handles availability checking and error reporting

} // extern "C"
