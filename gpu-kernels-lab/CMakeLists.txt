cmake_minimum_required(VERSION 3.16)

project(gpu_kernels_lab LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GKL_SOURCES src/reference_ops.cpp)

if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
endif()

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
    endif()
  endif()

  list(APPEND GKL_SOURCES
       kernels/gemm_kernels.cu
       kernels/softmax_kernels.cu
       kernels/layernorm_kernels.cu
       kernels/stencil_kernels.cu)
endif()

add_library(gkl_lib ${GKL_SOURCES})
target_include_directories(gkl_lib PUBLIC include)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  target_compile_definitions(gkl_lib PUBLIC USE_CUDA)
  target_link_libraries(gkl_lib PUBLIC CUDA::cudart CUDA::nvToolsExt)
endif()

add_executable(gpu_kernels_bench bench/main.cpp)
target_link_libraries(gpu_kernels_bench PRIVATE gkl_lib)

add_executable(cpu_tests tests/test_cpu.cpp)
target_link_libraries(cpu_tests PRIVATE gkl_lib)

enable_testing()
add_test(NAME cpu_tests COMMAND cpu_tests)

add_custom_target(format
  COMMAND clang-format -i
          include/common/*.h
          include/*.h
          src/*.cpp
          bench/*.cpp
          tests/*.cpp
          kernels/*.cu
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running clang-format"
)
