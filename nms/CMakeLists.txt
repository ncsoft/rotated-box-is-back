cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./include/
    ./
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(nms_kernel SHARED
  nms_kernel.cu nms_kernel.cpp rbox_nms_kernel.cu)

target_link_libraries(nms_kernel
  ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES}
  cudart)

set_target_properties(nms_kernel PROPERTIES PREFIX "")
