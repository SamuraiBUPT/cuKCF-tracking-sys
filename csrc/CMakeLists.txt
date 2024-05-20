cmake_minimum_required(VERSION 3.10)
project(cuKCF)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

# 添加 PyTorch 包
set(Torch_DIR "/d_workspace/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_86,code=sm_86)

# 添加 CUDA 库
cuda_add_library(cuKCF_lib SHARED fhog.cu)

# 添加 PyTorch 和 pybind11 模块
add_library(cuKCF MODULE wrapper.cpp)
target_link_libraries(cuKCF PRIVATE cuKCF_lib ${CUDA_LIBRARIES} pybind11::module ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(cuKCF PRIVATE ${CUDA_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
set_target_properties(cuKCF PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# 设置 PyTorch 的编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
