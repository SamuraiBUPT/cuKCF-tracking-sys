cmake_minimum_required(VERSION 3.10)
project(MKCFup)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 寻找OpenCV、FFTW库
find_package(OpenCV REQUIRED)

# 注意：在这里你可能要手动调试找到Pybind11的位置
# 使用pip show pybind11获取安装位置，然后拷贝即可
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

find_package(OpenMP)
find_library(FFTW3F_LIBRARY NAMES fftw3f)

# 包含FFTW和OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

# 添加可执行文件
add_executable(MKCFup main.cpp fhog.hpp cnfeat.hpp cnfeat.cpp ComplexMat.cpp gradientMex.cpp params.h libMkcfup.cpp)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(MKCFup PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

# 链接OpenCV和FFTW库到你的项目
target_link_libraries(MKCFup PUBLIC ${OpenCV_LIBS})

link_directories(/usr/lib/x86_64-linux-gnu)
target_link_libraries(MKCFup PUBLIC fftw3)
# target_link_libraries(MKCFup ${FFTW_LIBRARIES})

target_link_libraries(MKCFup PUBLIC ${FFTW3F_LIBRARY})

# 添加Python模块
pybind11_add_module(mkcfup bindings.cpp libMkcfup.cpp fhog.hpp cnfeat.hpp cnfeat.cpp ComplexMat.cpp gradientMex.cpp params.h)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(mkcfup PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

# 链接OpenCV和FFTW库到你的项目
target_link_libraries(mkcfup PUBLIC ${OpenCV_LIBS})

link_directories(/usr/lib/x86_64-linux-gnu)
target_link_libraries(mkcfup PUBLIC fftw3)
# target_link_libraries(MKCFup ${FFTW_LIBRARIES})

target_link_libraries(mkcfup PUBLIC ${FFTW3F_LIBRARY})
