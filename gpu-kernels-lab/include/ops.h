#pragma once

#include <cstddef>
#include <vector>

#include "common/types.h"

namespace gkl {

struct GemmParams {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  bool use_bias = false;
  bool use_relu = false;
};

struct SoftmaxParams {
  int rows = 1024;
  int cols = 1024;
};

struct LayerNormParams {
  int rows = 1024;
  int cols = 1024;
  float epsilon = 1e-5f;
};

struct StencilParams {
  int height = 1024;
  int width = 1024;
};

void cpu_gemm(const float *a, const float *b, const float *bias, float *c,
              const GemmParams &params);
void cpu_softmax(const float *input, float *output,
                 const SoftmaxParams &params);
void cpu_layernorm(const float *input, float *output, const float *gamma,
                   const float *beta, const LayerNormParams &params);
void cpu_stencil(const float *input, float *output,
                 const StencilParams &params);

struct ErrorStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
};

ErrorStats compare_tensors(const std::vector<float> &ref,
                           const std::vector<float> &out);

} // namespace gkl
