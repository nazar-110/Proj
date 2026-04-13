#include "kernels.h"

#include <cuda_fp16.h>
#include <nvToolsExt.h>

#include "common/cuda_utils.h"
#include "common/timer.h"

namespace gkl {

namespace {

template <typename T>
__device__ float to_float(T val) {
  return static_cast<float>(val);
}

template <>
__device__ float to_float<__half>(__half val) {
  return __half2float(val);
}

template <typename T>
__device__ T from_float(float val) {
  return static_cast<T>(val);
}

template <>
__device__ __half from_float<__half>(float val) {
  return __float2half(val);
}

template <typename T>
__global__ void layernorm_baseline(const T *input, T *output, const T *gamma,
                                   const T *beta, int rows, int cols,
                                   float epsilon) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float *buf = smem;
  float mean = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    mean += to_float(input[row * cols + col]);
  }
  buf[threadIdx.x] = mean;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float row_mean = buf[0] / cols;
  float var = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float diff = to_float(input[row * cols + col]) - row_mean;
    var += diff * diff;
  }
  buf[threadIdx.x] = var;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float inv_std = rsqrtf(buf[0] / cols + epsilon);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float val = (to_float(input[row * cols + col]) - row_mean) * inv_std;
    float g = gamma ? to_float(gamma[col]) : 1.0f;
    float b = beta ? to_float(beta[col]) : 0.0f;
    output[row * cols + col] = from_float<T>(val * g + b);
  }
}

template <typename T>
__global__ void layernorm_fused(const T *input, T *output, const T *gamma,
                                const T *beta, int rows, int cols,
                                float epsilon) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float *buf = smem;
  float mean = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    mean += to_float(input[row * cols + col]);
  }
  buf[threadIdx.x] = mean;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float row_mean = buf[0] / cols;
  float var = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float diff = to_float(input[row * cols + col]) - row_mean;
    var += diff * diff;
  }
  buf[threadIdx.x] = var;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float inv_std = rsqrtf(buf[0] / cols + epsilon);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float val = (to_float(input[row * cols + col]) - row_mean) * inv_std;
    float g = gamma ? to_float(gamma[col]) : 1.0f;
    float b = beta ? to_float(beta[col]) : 0.0f;
    output[row * cols + col] = from_float<T>(val * g + b);
  }
}

template <typename T>
float run_layernorm(const T *input, T *output, const T *gamma, const T *beta,
                    const LayerNormParams &params, LayerNormKernel kernel,
                    int iters, int warmup) {
  dim3 block(256);
  dim3 grid(params.rows);
  size_t shared_bytes = block.x * sizeof(float);
  CudaEventTimer timer;
  nvtxRangePushA("layernorm_warmup");
  for (int i = 0; i < warmup; ++i) {
    if (kernel == LayerNormKernel::kBaseline) {
      layernorm_baseline<<<grid, block, shared_bytes>>>(
          input, output, gamma, beta, params.rows, params.cols, params.epsilon);
    } else {
      layernorm_fused<<<grid, block, shared_bytes>>>(input, output, gamma, beta,
                                                     params.rows, params.cols,
                                                     params.epsilon);
    }
  }
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePushA("layernorm_benchmark");
  timer.start();
  for (int i = 0; i < iters; ++i) {
    if (kernel == LayerNormKernel::kBaseline) {
      layernorm_baseline<<<grid, block, shared_bytes>>>(
          input, output, gamma, beta, params.rows, params.cols, params.epsilon);
    } else {
      layernorm_fused<<<grid, block, shared_bytes>>>(input, output, gamma, beta,
                                                     params.rows, params.cols,
                                                     params.epsilon);
    }
  }
  float ms = timer.stop_ms();
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  return ms / iters;
}

} // namespace

KernelResult launch_layernorm(const void *input, void *output, const void *gamma,
                              const void *beta, const LayerNormParams &params,
                              DType dtype, LayerNormKernel kernel, int iters,
                              int warmup, float *ms_out) {
  if (!is_cuda_available()) {
    return {false, "CUDA device not available"};
  }
  if (dtype == DType::kF16) {
    *ms_out = run_layernorm(reinterpret_cast<const __half *>(input),
                            reinterpret_cast<__half *>(output),
                            reinterpret_cast<const __half *>(gamma),
                            reinterpret_cast<const __half *>(beta), params,
                            kernel, iters, warmup);
  } else {
    *ms_out = run_layernorm(reinterpret_cast<const float *>(input),
                            reinterpret_cast<float *>(output),
                            reinterpret_cast<const float *>(gamma),
                            reinterpret_cast<const float *>(beta), params,
                            kernel, iters, warmup);
  }
  return {true, ""};
}

} // namespace gkl
