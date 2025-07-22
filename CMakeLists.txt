cmake_minimum_required(VERSION 3.10)
project(ReadFASTData LANGUAGES C CXX)

# 设置 C 标准并启用警告
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funroll-loops -flto")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -flto")
    #以上两个选项会减少gprof能获得的profile信息
endif()

find_package(PkgConfig REQUIRED)
# 查找 OpenMP
find_package(OpenMP REQUIRED)
# 查找 libpng
find_package(PNG REQUIRED)

pkg_search_module(FFTW3 REQUIRED fftw3f)

# 查找依赖库（使用现代 CMake 的 find_package 或 find_library）
find_library(CFITSIO_LIBRARY NAMES cfitsio REQUIRED)
find_library(GFORTRAN_LIBRARY NAMES gfortran REQUIRED)
find_library(CPGPLOT_LIBRARY NAMES cpgplot REQUIRED)
find_library(MATH_LIBRARY NAMES m REQUIRED)


# 添加可执行文件
add_executable(ReadFASTData
    src/ReadFASTData.c
    src/findStats.c
    src/psrPalett.c
    src/plot.c
    src/cmd.c
    src/identification.c
    src/transpose.c
    include/ReadFASTData.h
    include/findStats.h
    include/psrPalett.h
    include/plot.h
    include/cmd.h
    include/identification.h
    include/transpose.h
)

# 设置头文件包含路径（现代 CMake 推荐 target_include_directories）
target_include_directories(ReadFASTData PRIVATE
    include
    /usr/include
    ${FFTW3_INCLUDE_DIRS}
)

# 链接库（现代 CMake 推荐 target_link_libraries）
target_link_libraries(ReadFASTData PRIVATE
    ${CFITSIO_LIBRARY}
    ${GFORTRAN_LIBRARY}
    ${CPGPLOT_LIBRARY}
    ${MATH_LIBRARY}
    ${FFTW3_LIBRARIES}
    OpenMP::OpenMP_C  # 链接 OpenMP 目标
    PNG::PNG  # 链接 libpng 目标
)
