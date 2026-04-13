#include "kernels.h"

#include <cuda_fp16.h>
#include <nvToolsExt.h>

#include <vector>

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
__global__ void gemm_naive(const T *a, const T *b, const T *bias, T *c, int m,
                           int n, int k, bool use_bias, bool use_relu) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += to_float(a[row * k + i]) * to_float(b[i * n + col]);
    }
    if (use_bias) {
      sum += to_float(bias[col]);
    }
    if (use_relu) {
      sum = sum > 0.0f ? sum : 0.0f;
    }
    c[row * n + col] = from_float<T>(sum);
  }
}

template <typename T, int TILE>
__global__ void gemm_tiled(const T *a, const T *b, const T *bias, T *c, int m,
                           int n, int k, bool use_bias, bool use_relu) {
  __shared__ T smem_a[TILE][TILE];
  __shared__ T smem_b[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;
  for (int tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
    int tiled_col = tile * TILE + threadIdx.x;
    int tiled_row = tile * TILE + threadIdx.y;
    smem_a[threadIdx.y][threadIdx.x] =
        (row < m && tiled_col < k) ? a[row * k + tiled_col] : from_float<T>(0.0f);
    smem_b[threadIdx.y][threadIdx.x] =
        (tiled_row < k && col < n) ? b[tiled_row * n + col] : from_float<T>(0.0f);
    __syncthreads();
    for (int i = 0; i < TILE; ++i) {
      sum += to_float(smem_a[threadIdx.y][i]) * to_float(smem_b[i][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    if (use_bias) {
      sum += to_float(bias[col]);
    }
    if (use_relu) {
      sum = sum > 0.0f ? sum : 0.0f;
    }
    c[row * n + col] = from_float<T>(sum);
  }
}

__global__ void gemm_vectorized(const float *a, const float *b, const float *bias,
                                float *c, int m, int n, int k, bool use_bias,
                                bool use_relu) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (row < m && col < n) {
    float4 sum4{0, 0, 0, 0};
    for (int i = 0; i < k; ++i) {
      float aval = a[row * k + i];
      const float4 *b_ptr =
          reinterpret_cast<const float4 *>(b + i * n + col);
      float4 bval = *b_ptr;
      sum4.x += aval * bval.x;
      sum4.y += aval * bval.y;
      sum4.z += aval * bval.z;
      sum4.w += aval * bval.w;
    }
    if (use_bias) {
      const float4 *bias_ptr = reinterpret_cast<const float4 *>(bias + col);
      float4 bval = *bias_ptr;
      sum4.x += bval.x;
      sum4.y += bval.y;
      sum4.z += bval.z;
      sum4.w += bval.w;
    }
    float *out_ptr = c + row * n + col;
    if (use_relu) {
      sum4.x = sum4.x > 0.0f ? sum4.x : 0.0f;
      sum4.y = sum4.y > 0.0f ? sum4.y : 0.0f;
      sum4.z = sum4.z > 0.0f ? sum4.z : 0.0f;
      sum4.w = sum4.w > 0.0f ? sum4.w : 0.0f;
    }
    out_ptr[0] = sum4.x;
    if (col + 1 < n) out_ptr[1] = sum4.y;
    if (col + 2 < n) out_ptr[2] = sum4.z;
    if (col + 3 < n) out_ptr[3] = sum4.w;
  }
}

template <typename T>
float run_gemm_kernel(const T *a, const T *b, const T *bias, T *c,
                      const GemmParams &params, GemmKernel kernel, int iters,
                      int warmup, int tile_size) {
  dim3 block(16, 16);
  dim3 grid((params.n + block.x - 1) / block.x,
            (params.m + block.y - 1) / block.y);
  dim3 tile_block(tile_size, tile_size);
  dim3 tile_grid((params.n + tile_size - 1) / tile_size,
                 (params.m + tile_size - 1) / tile_size);

  CudaEventTimer timer;
  nvtxRangePushA("gemm_warmup");
  for (int i = 0; i < warmup; ++i) {
    if (kernel == GemmKernel::kNaive) {
      gemm_naive<<<grid, block>>>(a, b, bias, c, params.m, params.n, params.k,
                                  params.use_bias, params.use_relu);
    } else if (kernel == GemmKernel::kTiled) {
      switch (tile_size) {
        case 8:
          gemm_tiled<T, 8><<<tile_grid, tile_block>>>(a, b, bias, c, params.m,
                                                      params.n, params.k,
                                                      params.use_bias,
                                                      params.use_relu);
          break;
        case 16:
          gemm_tiled<T, 16><<<tile_grid, tile_block>>>(a, b, bias, c, params.m,
                                                       params.n, params.k,
                                                       params.use_bias,
                                                       params.use_relu);
          break;
        case 32:
          gemm_tiled<T, 32><<<tile_grid, tile_block>>>(a, b, bias, c, params.m,
                                                       params.n, params.k,
                                                       params.use_bias,
                                                       params.use_relu);
          break;
      }
    } else if (kernel == GemmKernel::kVectorized) {
      dim3 vblock(16, 16);
      dim3 vgrid((params.n + vblock.x * 4 - 1) / (vblock.x * 4),
                 (params.m + vblock.y - 1) / vblock.y);
      gemm_vectorized<<<vgrid, vblock>>>(reinterpret_cast<const float *>(a),
                                         reinterpret_cast<const float *>(b),
                                         reinterpret_cast<const float *>(bias),
                                         reinterpret_cast<float *>(c), params.m,
                                         params.n, params.k, params.use_bias,
                                         params.use_relu);
    } else if (kernel == GemmKernel::kFused) {
      gemm_naive<<<grid, block>>>(a, b, bias, c, params.m, params.n, params.k,
                                  params.use_bias, params.use_relu);
    }
  }
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePushA("gemm_benchmark");
  timer.start();
  for (int i = 0; i < iters; ++i) {
    if (kernel == GemmKernel::kNaive) {
      gemm_naive<<<grid, block>>>(a, b, bias, c, params.m, params.n, params.k,
                                  params.use_bias, params.use_relu);
    } else if (kernel == GemmKernel::kTiled) {
      switch (tile_size) {
        case 8:
          gemm_tiled<T, 8><<<tile_grid, tile_block>>>(a, b, bias, c, params.m,
                                                      params.n, params.k,
                                                      params.use_bias,
                                                      params.use_relu);
          break;
        case 16:
          gemm_tiled<T, 16><<<tile_grid, tile_block>>>(a, b, bias, c, params.m,
                                                       params.n, params.k,
                                                       params.use_bias,
                                                       params.use_relu);
          break;
        case 32:
          gemm_tiled<T, 32><<<tile_grid, tile_block>>>(a, b, bias, c, params.m,
                                                       params.n, params.k,
                                                       params.use_bias,
                                                       params.use_relu);
          break;
      }
    } else if (kernel == GemmKernel::kVectorized) {
      dim3 vblock(16, 16);
      dim3 vgrid((params.n + vblock.x * 4 - 1) / (vblock.x * 4),
                 (params.m + vblock.y - 1) / vblock.y);
      gemm_vectorized<<<vgrid, vblock>>>(reinterpret_cast<const float *>(a),
                                         reinterpret_cast<const float *>(b),
                                         reinterpret_cast<const float *>(bias),
                                         reinterpret_cast<float *>(c), params.m,
                                         params.n, params.k, params.use_bias,
                                         params.use_relu);
    } else if (kernel == GemmKernel::kFused) {
      gemm_naive<<<grid, block>>>(a, b, bias, c, params.m, params.n, params.k,
                                  params.use_bias, params.use_relu);
    }
  }
  float ms = timer.stop_ms();
  GKL_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  return ms / iters;
}

} // namespace

KernelResult launch_gemm(const void *a, const void *b, const void *bias, void *c,
                         const GemmParams &params, DType dtype,
                         GemmKernel kernel, int iters, int warmup,
                         float *ms_out) {
  if (!is_cuda_available()) {
    return {false, "CUDA device not available"};
  }

  int device = 0;
  cudaDeviceProp prop{};
  GKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int tile_size = 16;
  if (kernel == GemmKernel::kAutotune) {
    if (dtype != DType::kF32) {
      return {false, "autotune only implemented for fp32"};
    }
    std::vector<int> tiles = {8, 16, 32};
    float best_ms = 1e9f;
    int best_tile = 16;
    for (int candidate : tiles) {
      float ms = run_gemm_kernel<float>(reinterpret_cast<const float *>(a),
                                        reinterpret_cast<const float *>(b),
                                        reinterpret_cast<const float *>(bias),
                                        reinterpret_cast<float *>(c), params,
                                        GemmKernel::kTiled, iters, warmup,
                                        candidate);
      if (ms < best_ms) {
        best_ms = ms;
        best_tile = candidate;
      }
    }
    tile_size = best_tile;
    *ms_out = best_ms;
    return {true, "autotune picked tile " + std::to_string(tile_size)};
  }

  if (kernel == GemmKernel::kVectorized) {
    if (dtype != DType::kF32) {
      return {false, "vectorized kernel only supports fp32"};
    }
    if (params.n % 4 != 0 || reinterpret_cast<uintptr_t>(b) % 16 != 0) {
      return {false, "vectorized kernel requires n multiple of 4 and aligned"};
    }
    *ms_out =
        run_gemm_kernel<float>(reinterpret_cast<const float *>(a),
                               reinterpret_cast<const float *>(b),
                               reinterpret_cast<const float *>(bias),
                               reinterpret_cast<float *>(c), params, kernel,
                               iters, warmup, 16);
    return {true, ""};
  }

  if (kernel == GemmKernel::kTiled) {
    tile_size = (prop.sharedMemPerBlock >= 48 * 1024) ? 16 : 8;
  }

  if (dtype == DType::kF16) {
    *ms_out = run_gemm_kernel<__half>(reinterpret_cast<const __half *>(a),
                                      reinterpret_cast<const __half *>(b),
                                      reinterpret_cast<const __half *>(bias),
                                      reinterpret_cast<__half *>(c), params,
                                      kernel, iters, warmup, tile_size);
  } else {
    *ms_out = run_gemm_kernel<float>(reinterpret_cast<const float *>(a),
                                     reinterpret_cast<const float *>(b),
                                     reinterpret_cast<const float *>(bias),
                                     reinterpret_cast<float *>(c), params, kernel,
                                     iters, warmup, tile_size);
  }
  return {true, ""};
}

} // namespace gkl
