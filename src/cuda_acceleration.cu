#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>

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

} // extern "C"
