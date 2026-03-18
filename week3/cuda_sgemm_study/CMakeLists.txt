cmake_minimum_required(VERSION 3.18)

project(cuda_cmake_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_cmake_demo
    src/main.cu
)

# 可选：指定 GPU 架构
# 如果你是 30 系列可写 86，40 系列一般写 89
set_target_properties(cuda_cmake_demo PROPERTIES
    CUDA_ARCHITECTURES 89
)

# 可选：让 .cu 文件可分离编译，适合多文件 CUDA 项目
set_target_properties(cuda_cmake_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)