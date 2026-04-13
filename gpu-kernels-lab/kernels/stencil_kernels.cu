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
__global__ void stencil_baseline(const T *input, T *output, int height,
                                 int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  int idx = y * width + x;
  float center = to_float(input[idx]);
  float up = (y > 0) ? to_float(input[idx - width]) : center;
  float down = (y + 1 < height) ? to_float(input[idx + width]) : center;
  float left = (x > 0) ? to_float(input[idx - 1]) : center;
  float right = (x + 1 < width) ? to_float(input[idx + 1]) : center;
  output[idx] = from_float<T>((center + up + down + left + right) / 5.0f);
}

template <typename T>
__global__ void stencil_shared(const T *input, T *output, int height,
                               int width) {
  extern __shared__ T tile[];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int tile_width = blockDim.x + 2;
  int tile_x = tx + 1;
  int tile_y = ty + 1;
  if (x < width && y < height) {
    tile[tile_y * tile_width + tile_x] = input[y * width + x];
  }
  if (tx == 0 && y < height) {
    tile[tile_y * tile_width] =
        (x > 0) ? input[y * width + x - 1] : input[y * width + x];
  }
  if (tx == blockDim.x - 1 && y < height) {
    tile[tile_y * tile_width + tile_x + 1] =
        (x + 1 < width) ? input[y * width + x + 1] : input[y * width + x];
  }
  if (ty == 0 && x < width) {
    tile[tile_x] =
        (y > 0) ? input[(y - 1) * width + x] : input[y * width + x];
  }
  if (ty == blockDim.y - 1 && x < width) {
    tile[(tile_y + 1) * tile_width + tile_x] =
        (y + 1 < height) ? input[(y + 1) * width + x]
                         : input[y * width + x];
  }
  __syncthreads();
  if (x >= width || y >= height) return;
  float center = to_float(tile[tile_y * tile_width + tile_x]);
  float up = to_float(tile[(tile_y - 1) * tile_width + tile_x]);
  float down = to_float(tile[(tile_y + 1) * tile_width + tile_x]);
  float left = to_float(tile[tile_y * tile_width + tile_x - 1]);
  float right = to_float(tile[tile_y * tile_width + tile_x + 1]);
  output[y * width + x] =
      from_float<T>((center + up + down + left + right) / 5.0f);
}

template <typename T>
float run_stencil(const T *input, T *output, const StencilParams &params,
                  StencilKernel kernel, int iters, int warmup) {
  dim3 block(16, 16);
  dim3 grid((params.width + block.x - 1) / block.x,
            (params.height + block.y - 1) / block.y);
  size_t shared_bytes = (block.x + 2) * (block.y + 2) * sizeof(T);
  CudaEventTimer timer;
  nvtxRangePushA("stencil_warmup");
  for (int i = 0; i < warmup; ++i) {
    if (kernel == StencilKernel::kBaseline) {
      stencil_baseline<<<grid, block>>>(input, output, params.height,
                                        params.width);
    } else {
      stencil_shared<<<grid, block, shared_bytes>>>(input, output,
                                                    params.height,
                                                    params.width);
    }
  }
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePushA("stencil_benchmark");
  timer.start();
  for (int i = 0; i < iters; ++i) {
    if (kernel == StencilKernel::kBaseline) {
      stencil_baseline<<<grid, block>>>(input, output, params.height,
                                        params.width);
    } else {
      stencil_shared<<<grid, block, shared_bytes>>>(input, output,
                                                    params.height,
                                                    params.width);
    }
  }
  float ms = timer.stop_ms();
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  return ms / iters;
}

} // namespace

KernelResult launch_stencil(const void *input, void *output,
                            const StencilParams &params, DType dtype,
                            StencilKernel kernel, int iters, int warmup,
                            float *ms_out) {
  if (!is_cuda_available()) {
    return {false, "CUDA device not available"};
  }
  if (dtype == DType::kF16) {
    *ms_out = run_stencil(reinterpret_cast<const __half *>(input),
                          reinterpret_cast<__half *>(output), params, kernel,
                          iters, warmup);
  } else {
    *ms_out = run_stencil(reinterpret_cast<const float *>(input),
                          reinterpret_cast<float *>(output), params, kernel,
                          iters, warmup);
  }
  return {true, ""};
}

} // namespace gkl
