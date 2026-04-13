#pragma once

#include <chrono>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gkl {

class CpuTimer {
 public:
  void start() { start_ = Clock::now(); }
  double stop_ms() {
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> diff = end - start_;
    return diff.count();
  }

 private:
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_;
};

#ifdef USE_CUDA
class CudaEventTimer {
 public:
  CudaEventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t stream = 0) { cudaEventRecord(start_, stream); }
  float stop_ms(cudaStream_t stream = 0) {
    cudaEventRecord(stop_, stream);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};
#endif

} // namespace gkl
