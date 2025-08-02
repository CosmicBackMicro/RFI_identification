/*
 * CUDA Acceleration Module for deRFI
 * 
 * This file contains CUDA implementations of compute-intensive functions
 * from the RFI detection pipeline to accelerate processing on GPU.
 * 
 * Author: GitHub Copilot
 * Date: 2025-08-02
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Include header for CUDA functions
#include "cuda_acceleration.h"

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

// ============================================================================
// Host Interface Functions
// ============================================================================

/**
 * CUDA-accelerated channel median subtraction
 */
void cuda_subtractChannelMedians(float *data, const float *channel_medians, 
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
void cuda_calculateChannelStats(const float *data, float *means, float *stds, 
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
void cuda_transpose(const float *input, float *output, int rows, int cols)
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
void cuda_downsample2D(const float *input, float *output, 
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

/**
 * Initialize CUDA and check device capabilities
 */
int cuda_init()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return -1;
    }
    
    // Use the first device
    CUDA_CHECK(cudaSetDevice(0));
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    printf("CUDA Device Initialized:\n");
    printf("  Device: %s\n", deviceProp.name);
    printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Global Memory: %.1f GB\n", deviceProp.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    
    return 0;
}

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup()
{
    CUDA_CHECK(cudaDeviceReset());
}

/**
 * Check if CUDA is available and functional
 */
int cuda_isAvailable()
{
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        return 0; // CUDA not available
    }
    
    return (deviceCount > 0) ? 1 : 0;
}
