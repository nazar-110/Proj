#include "ops.h"

#include <algorithm>
#include <cmath>

namespace gkl {

void cpu_gemm(const float *a, const float *b, const float *bias, float *c,
              const GemmParams &params) {
  for (int i = 0; i < params.m; ++i) {
    for (int j = 0; j < params.n; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < params.k; ++k) {
        sum += a[i * params.k + k] * b[k * params.n + j];
      }
      if (params.use_bias && bias) {
        sum += bias[j];
      }
      if (params.use_relu) {
        sum = std::max(0.0f, sum);
      }
      c[i * params.n + j] = sum;
    }
  }
}

void cpu_softmax(const float *input, float *output,
                 const SoftmaxParams &params) {
  for (int row = 0; row < params.rows; ++row) {
    const float *row_ptr = input + row * params.cols;
    float *out_ptr = output + row * params.cols;
    float max_val = row_ptr[0];
    for (int col = 1; col < params.cols; ++col) {
      max_val = std::max(max_val, row_ptr[col]);
    }
    float sum = 0.0f;
    for (int col = 0; col < params.cols; ++col) {
      float val = std::exp(row_ptr[col] - max_val);
      out_ptr[col] = val;
      sum += val;
    }
    for (int col = 0; col < params.cols; ++col) {
      out_ptr[col] /= sum;
    }
  }
}

void cpu_layernorm(const float *input, float *output, const float *gamma,
                   const float *beta, const LayerNormParams &params) {
  for (int row = 0; row < params.rows; ++row) {
    const float *row_ptr = input + row * params.cols;
    float *out_ptr = output + row * params.cols;
    float mean = 0.0f;
    for (int col = 0; col < params.cols; ++col) {
      mean += row_ptr[col];
    }
    mean /= params.cols;
    float var = 0.0f;
    for (int col = 0; col < params.cols; ++col) {
      float diff = row_ptr[col] - mean;
      var += diff * diff;
    }
    var /= params.cols;
    float inv_std = 1.0f / std::sqrt(var + params.epsilon);
    for (int col = 0; col < params.cols; ++col) {
      float norm = (row_ptr[col] - mean) * inv_std;
      float g = gamma ? gamma[col] : 1.0f;
      float b = beta ? beta[col] : 0.0f;
      out_ptr[col] = norm * g + b;
    }
  }
}

void cpu_stencil(const float *input, float *output,
                 const StencilParams &params) {
  auto idx = [=](int r, int c) { return r * params.width + c; };
  for (int r = 0; r < params.height; ++r) {
    for (int c = 0; c < params.width; ++c) {
      float center = input[idx(r, c)];
      float up = (r > 0) ? input[idx(r - 1, c)] : center;
      float down = (r + 1 < params.height) ? input[idx(r + 1, c)] : center;
      float left = (c > 0) ? input[idx(r, c - 1)] : center;
      float right = (c + 1 < params.width) ? input[idx(r, c + 1)] : center;
      output[idx(r, c)] = (center + up + down + left + right) / 5.0f;
    }
  }
}

ErrorStats compare_tensors(const std::vector<float> &ref,
                           const std::vector<float> &out) {
  ErrorStats stats{};
  size_t size = std::min(ref.size(), out.size());
  for (size_t i = 0; i < size; ++i) {
    float diff = std::abs(ref[i] - out[i]);
    stats.max_abs = std::max(stats.max_abs, diff);
    float denom = std::max(std::abs(ref[i]), 1e-6f);
    stats.max_rel = std::max(stats.max_rel, diff / denom);
  }
  return stats;
}

} // namespace gkl
