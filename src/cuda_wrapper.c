#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "cuda_acceleration.h"

// Global state for CUDA availability
static int cuda_availability_checked = 0;
static int cuda_is_available = 0;

// Function pointers for dynamically loaded CUDA functions
static void* cuda_lib_handle = NULL;
static int (*cuda_runtime_init)(void) = NULL;
static void (*cuda_runtime_cleanup)(void) = NULL;
static void (*cuda_runtime_transpose)(const float*, float*, int, int) = NULL;
static void (*cuda_runtime_subtractChannelMedians)(float*, const float*, int, int) = NULL;
static void (*cuda_runtime_downsample2D)(const float*, float*, int, int, int, int) = NULL;
static void (*cuda_runtime_computeStatistics)(const float*, float*, float*, int, int) = NULL;

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

// Weak symbol declarations for CUDA implementation functions
// These will be resolved at link time if CUDA code is compiled in
extern void cuda_transpose_impl(const float *input, float *output, int rows, int cols) __attribute__((weak));
extern void cuda_subtractChannelMedians_impl(float *data, const float *channel_medians, 
                                            int nsamp, int nchan) __attribute__((weak));
extern void cuda_downsample2D_impl(const float *input, float *output, int nsamp, int nchan,
                                  int binFactorTime, int binFactorFreq) __attribute__((weak));
extern void cuda_computeStatistics_impl(const float *data, float *means, float *stds,
                                       int nsamp, int nchan) __attribute__((weak));
extern void cuda_binarySIR_impl(int *mask, int nsamp, int nchan,
                               int win_samp, int win_chan, 
                               float thr_up, float thr_down) __attribute__((weak));

void cuda_transpose(const float *input, float *output, int rows, int cols) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA transpose requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }
    
    if (cuda_transpose_impl) {
        cuda_transpose_impl(input, output, rows, cols);
    } else {
        fprintf(stderr, "Error: CUDA transpose function not linked\n");
        exit(EXIT_FAILURE);
    }
}

void cuda_subtractChannelMedians(float *data, const float *channel_medians, 
                                int nsamp, int nchan) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA subtractChannelMedians requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }
    
    if (cuda_subtractChannelMedians_impl) {
        cuda_subtractChannelMedians_impl(data, channel_medians, nsamp, nchan);
    } else {
        fprintf(stderr, "Error: CUDA subtractChannelMedians function not linked\n");
        exit(EXIT_FAILURE);
    }
}

void cuda_downsample2D(const float *input, float *output, int nsamp, int nchan,
                      int binFactorTime, int binFactorFreq) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA downsample2D requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }
    
    if (cuda_downsample2D_impl) {
        cuda_downsample2D_impl(input, output, nsamp, nchan, binFactorTime, binFactorFreq);
    } else {
        fprintf(stderr, "Error: CUDA downsample2D function not linked\n");
        exit(EXIT_FAILURE);
    }
}

void cuda_computeStatistics(const float *data, float *means, float *stds,
                           int nsamp, int nchan) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA computeStatistics requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }
    
    if (cuda_computeStatistics_impl) {
        cuda_computeStatistics_impl(data, means, stds, nsamp, nchan);
    } else {
        fprintf(stderr, "Error: CUDA computeStatistics function not linked\n");
        exit(EXIT_FAILURE);
    }
}

void cuda_binarySIR(int *mask, int nsamp, int nchan,
                   int win_samp, int win_chan, 
                   float thr_up, float thr_down) {
    if (!cuda_isAvailable()) {
        fprintf(stderr, "Error: CUDA binarySIR requested but CUDA is not available\n");
        exit(EXIT_FAILURE);
    }
    
    if (cuda_binarySIR_impl) {
        cuda_binarySIR_impl(mask, nsamp, nchan, win_samp, win_chan, thr_up, thr_down);
    } else {
        fprintf(stderr, "Error: CUDA binarySIR function not linked\n");
        exit(EXIT_FAILURE);
    }
}
