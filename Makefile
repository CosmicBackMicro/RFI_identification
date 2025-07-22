# 编译器配置
CC := clang
CXX := clang++

# Set Warnings
CFLAGS := -Wall -pg -g -Wno-format-security
CXXFLAGS := -Wall -pg -g -Wno-format-security

# Set Optimization Flags
# CFLAGS += -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp
# CXXFLAGS += -O3 -march=native -ffast-math -funroll-loops -flto -fopenmp

CFLAGS += -D_GNU_SOURCE
CXXFLAGS += -D_GNU_SOURCE

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
LIBS := -lcfitsio -lgfortran -lcpgplot -lm -lfftw3f -lpng -lgomp
INCLUDES := -I$(INC_DIR) -I/usr/include $(shell pkg-config --cflags fftw3f)

# 默认构建目标
all: $(TARGET)

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

# 伪目标声明
.PHONY: all clean