cmake_minimum_required(VERSION 3.10)
project(my_cuda_extension)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_50,code=sm_50)

cuda_add_library(my_cuda_lib SHARED kernel.cu)

add_library(my_cuda_extension MODULE wrapper.cpp)
target_link_libraries(my_cuda_extension PRIVATE my_cuda_lib ${CUDA_LIBRARIES} pybind11::module)
target_include_directories(my_cuda_extension PRIVATE ${CUDA_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})
set_target_properties(my_cuda_extension PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
