#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
// Removed unused headers to simplify

// Include header for CUDA functions
#include "cuda_acceleration.h"
// Needed for CPU fallback in cuda_identSubstNSigma wrapper (C symbols)
extern "C" {
#include "identification.h"
}
// Ensure C linkage only for exported C-callable APIs (defined below)

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
// Public CUDA Interface Functions (C linkage)
// ============================================================================
extern "C" {
    int cuda_isAvailable(void) {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0) ? 1 : 0;
    }

    int cuda_init(void) {
        if (!cuda_isAvailable()) {
            return -1;
        }
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            return -1;
        }
        printf("CUDA initialization successful.\n");
        return 0;
    }
} // extern "C" (end of availability/init/cleanup)


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

/**
 * CUDA-accelerated matrix transpose (single entry point)
 */
extern "C" void cuda_transpose(const float *input, float *output, int rows, int cols) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA transpose requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }

    // Device memory allocation
    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, (size_t)rows * (size_t)cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)rows * (size_t)cols * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, (size_t)rows * (size_t)cols * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with 2D grid
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    transposeKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, (size_t)rows * (size_t)cols * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}