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
__global__ void softmax_baseline(const T *input, T *output, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float *smax = smem;
  float *ssum = smem + blockDim.x;
  float max_val = -1e20f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    max_val = fmaxf(max_val, to_float(input[row * cols + col]));
  }
  smax[threadIdx.x] = max_val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_value = smax[0];
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += expf(to_float(input[row * cols + col]) - max_value);
  }
  ssum[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ssum[threadIdx.x] += ssum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = ssum[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float val = expf(to_float(input[row * cols + col]) - max_value) / denom;
    output[row * cols + col] = from_float<T>(val);
  }
}
}

__device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ float warp_reduce_max(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename T>
__global__ void softmax_warp(const T *input, T *output, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  float max_val = -1e20f;
  for (int col = tid; col < cols; col += blockDim.x) {
    max_val = fmaxf(max_val, to_float(input[row * cols + col]));
  }
  float max_reduce = warp_reduce_max(max_val);
  __shared__ float warp_max[32];
  if ((tid & (warpSize - 1)) == 0) {
    warp_max[tid / warpSize] = max_reduce;
  }
  __syncthreads();
  float block_max = (tid < blockDim.x / warpSize) ? warp_max[tid] : -1e20f;
  block_max = warp_reduce_max(block_max);
  block_max = __shfl_sync(0xffffffff, block_max, 0);

  float sum_val = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    sum_val += expf(to_float(input[row * cols + col]) - block_max);
  }
  float warp_sum = warp_reduce_sum(sum_val);
  __shared__ float warp_sums[32];
  if ((tid & (warpSize - 1)) == 0) {
    warp_sums[tid / warpSize] = warp_sum;
  }
  __syncthreads();
  float block_sum = (tid < blockDim.x / warpSize) ? warp_sums[tid] : 0.0f;
  block_sum = warp_reduce_sum(block_sum);
  block_sum = __shfl_sync(0xffffffff, block_sum, 0);

  for (int col = tid; col < cols; col += blockDim.x) {
    float val = expf(to_float(input[row * cols + col]) - block_max) / block_sum;
    output[row * cols + col] = from_float<T>(val);
  }
}

template <typename T>
float run_softmax(const T *input, T *output, const SoftmaxParams &params,
                  SoftmaxKernel kernel, int iters, int warmup) {
  dim3 block(256);
  dim3 grid(params.rows);
  CudaEventTimer timer;
  nvtxRangePushA("softmax_warmup");
  for (int i = 0; i < warmup; ++i) {
    if (kernel == SoftmaxKernel::kBaseline) {
      softmax_baseline<<<grid, block, block.x * sizeof(float) * 2>>>(
          input, output, params.rows, params.cols);
    } else {
      softmax_warp<<<grid, block>>>(input, output, params.rows, params.cols);
    }
  }
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePushA("softmax_benchmark");
  timer.start();
  for (int i = 0; i < iters; ++i) {
    if (kernel == SoftmaxKernel::kBaseline) {
      softmax_baseline<<<grid, block, block.x * sizeof(float) * 2>>>(
          input, output, params.rows, params.cols);
    } else {
      softmax_warp<<<grid, block>>>(input, output, params.rows, params.cols);
    }
  }
  float ms = timer.stop_ms();
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  return ms / iters;
}

} // namespace

KernelResult launch_softmax(const void *input, void *output,
                            const SoftmaxParams &params, DType dtype,
                            SoftmaxKernel kernel, int iters, int warmup,
                            float *ms_out) {
  if (!is_cuda_available()) {
    return {false, "CUDA device not available"};
  }
  if (kernel == SoftmaxKernel::kWarp && !supports_warp_shfl()) {
    return {false, "warp shuffle not supported"};
  }
  if (dtype == DType::kF16) {
    *ms_out = run_softmax(reinterpret_cast<const __half *>(input),
                          reinterpret_cast<__half *>(output), params, kernel,
                          iters, warmup);
  } else {
    *ms_out = run_softmax(reinterpret_cast<const float *>(input),
                          reinterpret_cast<float *>(output), params, kernel,
                          iters, warmup);
  }
  return {true, ""};
}

} // namespace gkl
