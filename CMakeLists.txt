cmake_minimum_required(VERSION 3.18)

project(cuda-ml LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(cuda-ml SHARED
    src/tensor.cu
    src/autodiff.cpp
    src/kernels.cu
    src/data.cu
    src/network.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/utils.cu
)

install(TARGETS cuda-ml DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/cuda-ml)
