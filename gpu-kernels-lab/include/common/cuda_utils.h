#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gkl {

inline void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

#ifdef USE_CUDA
inline void check_cuda(cudaError_t result, const char *file, int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << ": "
              << cudaGetErrorString(result) << std::endl;
    throw std::runtime_error("CUDA failure");
  }
}

#define GKL_CUDA_CHECK(expr) gkl::check_cuda((expr), __FILE__, __LINE__)

struct DeviceInfo {
  int device_id = 0;
  int major = 0;
  int minor = 0;
  int multiprocessor_count = 0;
  int warp_size = 0;
  size_t shared_mem_per_block = 0;
  size_t shared_mem_per_sm = 0;
  size_t global_mem = 0;
  std::string name;
};

inline DeviceInfo query_device(int device_id = 0) {
  DeviceInfo info{};
  cudaDeviceProp prop{};
  GKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  info.device_id = device_id;
  info.major = prop.major;
  info.minor = prop.minor;
  info.multiprocessor_count = prop.multiProcessorCount;
  info.warp_size = prop.warpSize;
  info.shared_mem_per_block = prop.sharedMemPerBlock;
  info.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  info.global_mem = prop.totalGlobalMem;
  info.name = prop.name;
  return info;
}

inline bool is_cuda_available() {
  int count = 0;
  auto result = cudaGetDeviceCount(&count);
  return result == cudaSuccess && count > 0;
}

inline void print_device_header() {
  DeviceInfo info = query_device();
  std::cout << "GPU " << info.device_id << ": " << info.name << "\n";
  std::cout << "  Compute capability: " << info.major << "." << info.minor
            << "\n";
  std::cout << "  SMs: " << info.multiprocessor_count << "\n";
  std::cout << "  Warp size: " << info.warp_size << "\n";
  std::cout << "  Shared memory per block: " << info.shared_mem_per_block / 1024
            << " KB\n";
  std::cout << "  Shared memory per SM: " << info.shared_mem_per_sm / 1024
            << " KB\n";
  std::cout << "  Global memory: " << info.global_mem / (1024 * 1024)
            << " MB\n";
}

inline bool supports_tensor_cores() {
  DeviceInfo info = query_device();
  return info.major >= 7;
}

inline bool supports_warp_shfl() {
  return true;
}

#else
#define GKL_CUDA_CHECK(expr) (void)(expr)

inline bool is_cuda_available() { return false; }
inline void print_device_header() {
  std::cout << "CUDA not available. Running CPU-only mode.\n";
}

#endif

} // namespace gkl
