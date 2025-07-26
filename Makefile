# 编译器配置
CC := clang
CXX := clang++

# 检测编译器类型
IS_CLANG := $(shell $(CC) --version 2>/dev/null | grep -q clang && echo 1 || echo 0)
IS_GCC := $(shell $(CC) --version 2>/dev/null | grep -q gcc && echo 1 || echo 0)

# 构建模式：
# 可通过 make MODE=release 或 make MODE=debug 切换，
# 也可以直接修改下面这行来改变默认模式
MODE ?= release


# 基础编译选项
BASE_CFLAGS := -Wall -Wno-format-security -D_GNU_SOURCE
BASE_CXXFLAGS := -Wall -Wno-format-security -D_GNU_SOURCE 

# 调试模式
ifeq ($(MODE), debug)
    CFLAGS := $(BASE_CFLAGS) -pg -g -O0 -DDEBUG
    CXXFLAGS := $(BASE_CXXFLAGS) -pg -g -O0 -DDEBUG
    # Debug模式也需要OpenMP，因为代码中使用了OpenMP函数
    CFLAGS += -fopenmp
    CXXFLAGS += -fopenmp
# 性能模式
else ifeq ($(MODE), release)
    CFLAGS := $(BASE_CFLAGS) -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp -DNDEBUG
    # 科学计算优化选项
    CFLAGS += -fno-signed-zeros -fno-trapping-math -ffinite-math-only
    # 向量化优化
    CFLAGS += -ftree-vectorize
    # 内存优化
    CFLAGS += -fstrict-aliasing -falign-functions=32 -falign-loops=32
    # 针对数组密集计算的优化
    CFLAGS += -funsafe-math-optimizations
    
    # 编译器特定的优化选项
    ifeq ($(IS_GCC), 1)
        # GCC专用优化
        CFLAGS += -fvect-cost-model=cheap -floop-nest-optimize -fpredictive-commoning
    else ifeq ($(IS_CLANG), 1)
        # Clang专用优化
        CFLAGS += -fvectorize -fslp-vectorize
    endif
    
    CXXFLAGS := $(BASE_CXXFLAGS) -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp -DNDEBUG
    CXXFLAGS += -fno-signed-zeros -fno-trapping-math -ffinite-math-only
    CXXFLAGS += -ftree-vectorize
    CXXFLAGS += -fstrict-aliasing -falign-functions=32 -falign-loops=32
    CXXFLAGS += -funsafe-math-optimizations
    
    ifeq ($(IS_GCC), 1)
        CXXFLAGS += -fvect-cost-model=cheap -floop-nest-optimize -fpredictive-commoning
    else ifeq ($(IS_CLANG), 1)
        CXXFLAGS += -fvectorize -fslp-vectorize
    endif
# 科学计算极速模式 (激进优化)
else ifeq ($(MODE), turbo)
    CFLAGS := $(BASE_CFLAGS) -Ofast -march=native -mtune=native -fopenmp -DNDEBUG
    # 激进的浮点优化
    CFLAGS += -ffast-math -funsafe-math-optimizations -fno-signed-zeros -fno-trapping-math
    CFLAGS += -ffinite-math-only -fno-rounding-math -fno-signaling-nans
    # 激进的循环和向量化优化
    CFLAGS += -funroll-all-loops -ftree-vectorize -ftree-slp-vectorize
    # 内存和缓存优化
    CFLAGS += -fstrict-aliasing -falign-functions=64 -falign-loops=64
    # LTO和内联优化
    CFLAGS += -flto=auto -finline-functions -finline-limit=1000
    
    # 编译器特定的激进优化
    ifeq ($(IS_GCC), 1)
        CFLAGS += -fvect-cost-model=unlimited -floop-nest-optimize -fprefetch-loop-arrays -ftracer
    else ifeq ($(IS_CLANG), 1)
        CFLAGS += -fvectorize -fslp-vectorize
    endif
    
    CXXFLAGS := $(BASE_CXXFLAGS) -Ofast -march=native -mtune=native -fopenmp -DNDEBUG
    CXXFLAGS += -ffast-math -funsafe-math-optimizations -fno-signed-zeros -fno-trapping-math
    CXXFLAGS += -ffinite-math-only -fno-rounding-math -fno-signaling-nans
    CXXFLAGS += -funroll-all-loops -ftree-vectorize -ftree-slp-vectorize
    CXXFLAGS += -fstrict-aliasing -falign-functions=64 -falign-loops=64
    CXXFLAGS += -flto=auto -finline-functions -finline-limit=1000
    
    ifeq ($(IS_GCC), 1)
        CXXFLAGS += -fvect-cost-model=unlimited -floop-nest-optimize -fprefetch-loop-arrays -ftracer
    else ifeq ($(IS_CLANG), 1)
        CXXFLAGS += -fvectorize -fslp-vectorize
    endif
# 分析模式 (保留符号信息但优化)
else ifeq ($(MODE), profile)
    CFLAGS := $(BASE_CFLAGS) -pg -g -O2 -march=native -fopenmp
    # 适度的科学计算优化 (不影响profiling)
    CFLAGS += -ftree-vectorize -fstrict-aliasing
    
    # 编译器特定的profile优化
    ifeq ($(IS_GCC), 1)
        CFLAGS += -floop-nest-optimize
    endif
    
    CXXFLAGS := $(BASE_CXXFLAGS) -pg -g -O2 -march=native -fopenmp
    CXXFLAGS += -ftree-vectorize -fstrict-aliasing
    
    ifeq ($(IS_GCC), 1)
        CXXFLAGS += -floop-nest-optimize
    endif
else
    $(error Invalid MODE. Use: debug, release, profile, or turbo)
endif

# 目录结构
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
DEP_DIR := $(BUILD_DIR)/dep

# 源文件和目标文件
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(SRCS))

# 目标可执行文件
TARGET := $(BUILD_DIR)/ReadFASTData

# 库路径和链接选项
LIBS := -lcfitsio -lgfortran -lcpgplot -lm -lfftw3f -lpng
# 添加OpenMP支持
ifeq ($(findstring -fopenmp,$(CFLAGS)),-fopenmp)
    LIBS += -lgomp
endif

# 包含目录
INCLUDES := -I$(INC_DIR) -I/usr/include
# 自动检测fftw3f的pkg-config
FFTW3F_CFLAGS := $(shell pkg-config --cflags fftw3f 2>/dev/null || echo "")
FFTW3F_LIBS := $(shell pkg-config --libs fftw3f 2>/dev/null || echo "")
ifneq ($(FFTW3F_CFLAGS),)
    INCLUDES += $(FFTW3F_CFLAGS)
endif
ifneq ($(FFTW3F_LIBS),)
    LIBS += $(FFTW3F_LIBS)
endif

# NVTX支持 (如果CUDA可用)
NVTX_PATH := $(shell find /usr/local/cuda* /opt/cuda* -name "nvToolsExt.h" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
ifneq ($(NVTX_PATH),)
    # 检查库文件是否真的存在
    NVTX_LIB := $(shell find /usr/local/cuda* /opt/cuda* -name "libnvToolsExt.so*" -type f 2>/dev/null | head -1)
    ifneq ($(NVTX_LIB),)
        INCLUDES += -I$(dir $(NVTX_PATH))
        LIBS += -lnvToolsExt
        CFLAGS += -DHAVE_NVTX
        CXXFLAGS += -DHAVE_NVTX
        # 添加CUDA库路径
        CUDA_LIB_DIR := $(dir $(NVTX_LIB))
        LIBS += -L$(CUDA_LIB_DIR)
    endif
endif

# 默认构建目标
all: $(TARGET)

# 快速构建(适合开发阶段)
quick: 
	$(MAKE) MODE=debug $(TARGET)

# 性能构建
release: 
	$(MAKE) MODE=release $(TARGET)

# 性能分析构建
profile: 
	$(MAKE) MODE=profile $(TARGET)

# 科学计算极速构建 (激进优化)
turbo: 
	$(MAKE) MODE=turbo $(TARGET)

# 链接可执行文件
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

# 编译规则（含依赖生成）
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR) $(DEP_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -MF $(DEP_DIR)/$*.d -c $< -o $@

# 包含依赖文件
-include $(DEPS)

# 创建目录
$(OBJ_DIR) $(DEP_DIR):
	mkdir -p $@

# 清理构建文件
clean:
	rm -rf $(BUILD_DIR)

# 显示编译信息
info:
	@echo "Current build mode: $(MODE)"
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LIBS: $(LIBS)"
	@echo "Target: $(TARGET)"

# 显示所有可用的构建模式
help:
	@echo "Available build targets:"
	@echo "  quick/debug  - Fast compilation, full debug info, no optimization"
	@echo "  profile      - Moderate optimization + profiling support"
	@echo "  release      - High optimization for production use"
	@echo "  turbo        - Maximum optimization for scientific computing"
	@echo ""
	@echo "Usage examples:"
	@echo "  make quick          # Fast build for development"
	@echo "  make release        # Production build"
	@echo "  make turbo          # Maximum performance build"
	@echo "  make info MODE=turbo # Show compile flags for turbo mode"
	@echo "  make check-deps     # Check required libraries"

# 安装到系统路径 (可选)
install: $(TARGET)
	install -D $(TARGET) $(HOME)/bin/ReadFASTData

# 运行分析
analyze: profile
	./$(TARGET) --help && echo "Profile data: gmon.out"

# 检查依赖库
check-deps:
	@echo "Checking dependencies..."
	@pkg-config --exists fftw3f && echo "✓ fftw3f found" || echo "✗ fftw3f not found"
	@ldconfig -p | grep -q cfitsio && echo "✓ cfitsio found" || echo "✗ cfitsio not found"
	@ldconfig -p | grep -q cpgplot && echo "✓ cpgplot found" || echo "✗ cpgplot not found"

# 伪目标声明
.PHONY: all clean quick release profile turbo info help install analyze check-deps