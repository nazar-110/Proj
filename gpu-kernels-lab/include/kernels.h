#pragma once

#include "common/types.h"
#include "ops.h"

namespace gkl {

enum class GemmKernel { kNaive, kTiled, kVectorized, kFused, kAutotune };

enum class SoftmaxKernel { kBaseline, kWarp };

enum class LayerNormKernel { kBaseline, kFused };

enum class StencilKernel { kBaseline, kShared };

struct KernelResult {
  bool supported = true;
  std::string message;
};

#ifdef USE_CUDA
KernelResult launch_gemm(const void *a, const void *b, const void *bias, void *c,
                         const GemmParams &params, DType dtype,
                         GemmKernel kernel, int iters, int warmup,
                         float *ms_out);

KernelResult launch_softmax(const void *input, void *output,
                            const SoftmaxParams &params, DType dtype,
                            SoftmaxKernel kernel, int iters, int warmup,
                            float *ms_out);

KernelResult launch_layernorm(const void *input, void *output,
                              const void *gamma, const void *beta,
                              const LayerNormParams &params, DType dtype,
                              LayerNormKernel kernel, int iters, int warmup,
                              float *ms_out);

KernelResult launch_stencil(const void *input, void *output,
                            const StencilParams &params, DType dtype,
                            StencilKernel kernel, int iters, int warmup,
                            float *ms_out);
#else
inline KernelResult launch_gemm(const void *, const void *, const void *, void *,
                                const GemmParams &, DType, GemmKernel, int, int,
                                float *) {
  return {false, "CUDA not available"};
}

inline KernelResult launch_softmax(const void *, void *, const SoftmaxParams &,
                                   DType, SoftmaxKernel, int, int, float *) {
  return {false, "CUDA not available"};
}

inline KernelResult launch_layernorm(const void *, void *, const void *,
                                     const void *, const LayerNormParams &,
                                     DType, LayerNormKernel, int, int,
                                     float *) {
  return {false, "CUDA not available"};
}

inline KernelResult launch_stencil(const void *, void *,
                                   const StencilParams &, DType, StencilKernel,
                                   int, int, float *) {
  return {false, "CUDA not available"};
}
#endif

} // namespace gkl
