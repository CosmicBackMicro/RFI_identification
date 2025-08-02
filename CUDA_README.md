# CUDA Acceleration for deRFI

This document describes the CUDA acceleration features added to the deRFI project for GPU-accelerated radio frequency interference (RFI) detection.

## Overview

The CUDA acceleration module (`src/cuda_acceleration.cu`) provides GPU-accelerated implementations of compute-intensive functions in the RFI detection pipeline. This can significantly speed up processing of large radio astronomy datasets.

## Features

### Currently Implemented CUDA Functions

1. **Matrix Transpose** (`cuda_transpose`)
   - Optimized 2D matrix transpose using shared memory tiles
   - Coalesced memory access for better performance
   - Tile size: 32x32 with bank conflict avoidance

2. **Channel Statistics** (`cuda_calculateChannelStats`)
   - Parallel calculation of mean and standard deviation for each frequency channel
   - Uses shared memory reduction within thread blocks
   - One block per channel for optimal parallelization

3. **Channel Median Subtraction** (`cuda_subtractChannelMedians`)
   - Parallel subtraction of median values from frequency channels
   - Each thread processes one data element
   - Simple but effective parallelization

4. **2D Downsampling** (`cuda_downsample2D`)
   - GPU-accelerated averaging-based downsampling
   - 2D thread grid for optimal memory access patterns
   - Supports arbitrary downsampling factors

### Hot Spot Functions Identified for Future CUDA Implementation

Based on code analysis, the following functions are prime candidates for CUDA acceleration:

1. **Statistical Operations**:
   - `findMeanStd()` - Mean and standard deviation calculation
   - `median()` - Median calculation (can use GPU sorting)
   - `mad()` - Median Absolute Deviation

2. **Data Processing**:
   - `normalizeChannelData()` - Channel normalization
   - `substitute_pixels()` - Pixel substitution for RFI mitigation
   - `binarySIR()` - Binary morphological filtering

3. **Detection Algorithms**:
   - `sumthreshold_2d()` - 2D sum-threshold RFI detection
   - `identSubstNSigma()` - N-sigma outlier detection
   - `flagChannelsByStdOutliers()` - Channel flagging based on statistics

## Performance Benefits

### Expected Speedups

For typical radio astronomy datasets:
- **Matrix Transpose**: 3-10x speedup for large matrices (>1024x1024)
- **Channel Statistics**: 5-20x speedup depending on number of channels
- **Parallel Reductions**: 10-50x speedup for operations on large arrays
- **2D Convolutions/Filtering**: 5-15x speedup for morphological operations

### Memory Considerations

- GPU memory is limited compared to system RAM
- Data transfer overhead can reduce benefits for small datasets
- Optimal for datasets where GPU memory can hold working data
- Consider data streaming for very large datasets

## Usage

### Compilation

The build system automatically detects CUDA capability:

```bash
# Build with CUDA support (if nvcc is available)
make clean && make -j4

# Check if CUDA was enabled
./build/ReadFASTData --help  # Will show CUDA status at startup
```

### Runtime

CUDA functions are automatically used when:
1. CUDA-capable GPU is available
2. Program was compiled with CUDA support
3. CUDA drivers are properly installed

Example output:
```
CUDA Device Initialized:
  Device: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9
  Global Memory: 24.0 GB
  Multiprocessors: 128
  Max Threads per Block: 1024

Performing matrix transpose (2048 x 1024)...
CUDA transpose completed in 0.0023 seconds
```

### Fallback Behavior

If CUDA is not available, the program automatically falls back to CPU implementations with no loss of functionality.

## Technical Implementation

### Architecture

```
Host (CPU) Code          Device (GPU) Code
├── Memory allocation    ├── CUDA kernels
├── Data transfer        ├── Shared memory optimization
├── Kernel launch        ├── Thread block organization
├── Result retrieval     └── Memory coalescing
└── Error handling
```

### Memory Management

- Automatic GPU memory allocation and deallocation
- Error checking with informative messages
- Resource cleanup on program exit

### Optimization Techniques

1. **Shared Memory**: Used for data reuse and reduction operations
2. **Memory Coalescing**: Optimized memory access patterns
3. **Thread Divergence Minimization**: Balanced workload distribution
4. **Bank Conflict Avoidance**: Padded shared memory arrays

## Extending CUDA Support

### Adding New CUDA Functions

1. **Implement the CUDA kernel** in `src/cuda_acceleration.cu`:
   ```cuda
   __global__ void myKernel(float *data, int size) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < size) {
           // Your GPU code here
       }
   }
   ```

2. **Add host interface function**:
   ```cuda
   void cuda_myFunction(float *data, int size) {
       // GPU memory management
       // Kernel launch
       // Error checking
   }
   ```

3. **Update header file** `include/cuda_acceleration.h`

4. **Integrate into main code** with fallback:
   ```c
   #ifdef HAVE_CUDA
   if (cuda_isAvailable()) {
       cuda_myFunction(data, size);
   } else {
       cpu_myFunction(data, size);
   }
   #else
   cpu_myFunction(data, size);
   #endif
   ```

### Performance Profiling

Use NVIDIA profiling tools:
```bash
# Profile the application
nvprof ./build/ReadFASTData <args>

# Or use nsight-compute for detailed analysis
ncu --set full ./build/ReadFASTData <args>
```

## System Requirements

### Hardware
- NVIDIA GPU with Compute Capability 3.5 or higher
- Sufficient GPU memory for dataset processing
- PCIe connection for fast data transfer

### Software
- CUDA Toolkit 10.0 or later
- Compatible NVIDIA driver
- nvcc compiler in system PATH

### Tested Configurations
- CUDA 12.6 with RTX 4090 (24GB VRAM)
- CUDA 11.8 with RTX 3080 (10GB VRAM)
- Tesla V100 (16GB HBM2)

## Future Enhancements

### Planned Features
1. **Multi-GPU Support**: Distribute processing across multiple GPUs
2. **CUDA Streams**: Overlap computation and data transfer
3. **Unified Memory**: Simplify memory management
4. **cuFFT Integration**: GPU-accelerated FFT operations
5. **Thrust Library**: GPU-accelerated sorting and reductions

### Algorithm Optimizations
1. **Persistent Kernels**: Reduce kernel launch overhead
2. **Dynamic Parallelism**: GPU-initiated kernel launches
3. **Tensor Core Usage**: Utilize specialized AI hardware
4. **Memory Pool**: Reduce allocation/deallocation overhead

## Troubleshooting

### Common Issues

1. **CUDA not detected**:
   ```
   Built without CUDA support, using CPU only.
   ```
   - Check if nvcc is in PATH
   - Verify CUDA installation

2. **Runtime errors**:
   ```
   CUDA error at cuda_acceleration.cu:45 - out of memory
   ```
   - Reduce dataset size or increase GPU memory
   - Check for memory leaks

3. **Performance not improved**:
   - Dataset may be too small for GPU benefits
   - Check data transfer overhead
   - Profile with nvprof to identify bottlenecks

### Debug Mode

Build in debug mode for verbose CUDA information:
```bash
make MODE=debug -j4
```

This enables additional error checking and performance timing information.

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Radio Astronomy Software](https://science.nrao.edu/enss/evla/evla-compute-resources/software)
