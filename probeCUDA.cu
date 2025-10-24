#include <cuda_runtime.h>
#include <iostream>

// 获取架构名称（基于计算能力）
std::string getArchitectureName(int major, int minor) {
    switch (major) {
        case 1:
            return "Tesla";
        case 2:
            return "Fermi";
        case 3:
            return "Kepler";
        case 5:
            return "Maxwell";
        case 6:
            return "Pascal";
        case 7:
            return (minor == 0) ? "Volta" : "Turing";
        case 8:
            return (minor == 0) ? "Ampere" : (minor == 6) ? "Ada Lovelace" : "Ampere";
        case 9:
            return "Hopper";
        default:
            return "Unknown";
    }
}

// 计算每个SM的CUDA核心数量（基于计算能力）
int getCudaCoresPerSM(int major, int minor) {
    // 基于NVIDIA官方架构信息
    switch (major) {
        case 1:
            return 8;   // Tesla
        case 2:
            return 32;  // Fermi
        case 3:
            return 192; // Kepler
        case 5:
            return 128; // Maxwell
        case 6:
            return (minor == 0) ? 64 : 128; // GP100 (Pascal) vs Other Pascal
        case 7:
            return 64;  // GV100 (Volta) and GV10x (Turing)
        case 8:
            return (minor == 0) ? 64 : (minor == 6) ? 128 : 128; // GA100 (Ampere), GA10x (Ada Lovelace), Other Ampere
        case 9:
            return 128; // GH100 (Hopper)
        default:
            // 对于未知架构，使用保守估计
            return 128;
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << "\n"
                  << "-> " << cudaGetErrorString(error_id) << "\n";
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices were detected.\n";
    } else {
        std::cout << "检测到 " << deviceCount << " 个可用CUDA设备:\n";
        for (int dev = 0; dev < deviceCount; ++dev) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            std::cout << dev << "号设备: " << deviceProp.name << "\n";
            std::cout << "  计算能力: " << deviceProp.major << "." << deviceProp.minor << "\n";
            std::cout << "  架构: " << getArchitectureName(deviceProp.major, deviceProp.minor) << "\n";
            std::cout << "  总全局内存: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
            std::cout << "  多处理器SM数量: " << deviceProp.multiProcessorCount << "\n";
            
            // 计算准确的CUDA核心数量
            int coresPerSM = getCudaCoresPerSM(deviceProp.major, deviceProp.minor);
            int totalCudaCores = deviceProp.multiProcessorCount * coresPerSM;
            
            std::cout << "  CUDA核心数量: " << totalCudaCores 
                      << " (" << coresPerSM << " cores/SM)\n";
            std::cout << "  最大并发（分时交替执行）线程数/SM: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
            std::cout << "  最大并行（真正同时执行）线程数/Block: " << deviceProp.maxThreadsPerBlock << "\n";
            std::cout << "  最大Grid尺寸: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << "\n";
            std::cout << "  最大Block尺寸: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << "\n";
            std::cout << "  共享内存/Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB\n";
            std::cout << "  总常量内存: " << deviceProp.totalConstMem / 1024 << " KB\n";
            std::cout << "  时钟频率: " << deviceProp.clockRate / 1000 << " MHz\n";
            std::cout << "  内存时钟频率: " << deviceProp.memoryClockRate / 1000 << " MHz\n";
            std::cout << "  内存总线宽度: " << deviceProp.memoryBusWidth << " bits\n";
            std::cout << "  L2缓存大小: " << deviceProp.l2CacheSize / (1024 * 1024) << " MB\n";
            std::cout << "  纹理对齐: " << deviceProp.textureAlignment << " bytes\n";
            std::cout << "  并发内核: " << (deviceProp.concurrentKernels ? "支持" : "不支持") << "\n";
            std::cout << "  设备重叠: " << (deviceProp.deviceOverlap ? "支持" : "不支持") << "\n";
            std::cout << "  统一寻址: " << (deviceProp.unifiedAddressing ? "支持" : "不支持") << "\n";
            std::cout << "  映射主机内存: " << (deviceProp.canMapHostMemory ? "支持" : "不支持") << "\n";
            std::cout << "  ECC内存: " << (deviceProp.ECCEnabled ? "启用" : "禁用") << "\n";
            std::cout << "  计算模式: " << (deviceProp.computeMode == cudaComputeModeDefault ? "默认" : 
                                           deviceProp.computeMode == cudaComputeModeExclusive ? "独占" :
                                           deviceProp.computeMode == cudaComputeModeProhibited ? "禁止" : "独占进程") << "\n";
            std::cout << "\n";
        }
    }
    return EXIT_SUCCESS;
}