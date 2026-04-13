#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "ops.h"

using namespace gkl;

int main() {
  {
    GemmParams params{2, 3, 4, true, true};
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b = {1, 0, 2, 1, 0, 1, 3, 1, 1, 2, 1, 0};
    std::vector<float> bias = {1, 1, 1};
    std::vector<float> c(6, 0.0f);
    cpu_gemm(a.data(), b.data(), bias.data(), c.data(), params);
    assert(c.size() == 6);
    for (float val : c) {
      assert(!std::isnan(val));
    }
  }
  {
    SoftmaxParams params{2, 4};
    std::vector<float> input = {1, 2, 3, 4, 1, 1, 1, 1};
    std::vector<float> output(8, 0.0f);
    cpu_softmax(input.data(), output.data(), params);
    for (int row = 0; row < params.rows; ++row) {
      float sum = 0.0f;
      for (int col = 0; col < params.cols; ++col) {
        sum += output[row * params.cols + col];
      }
      assert(std::abs(sum - 1.0f) < 1e-4f);
    }
  }
  {
    LayerNormParams params{2, 4, 1e-5f};
    std::vector<float> input = {1, 2, 3, 4, 2, 3, 4, 5};
    std::vector<float> output(8, 0.0f);
    std::vector<float> gamma(4, 1.0f);
    std::vector<float> beta(4, 0.0f);
    cpu_layernorm(input.data(), output.data(), gamma.data(), beta.data(),
                  params);
    for (float val : output) {
      assert(!std::isnan(val));
    }
  }
  {
    StencilParams params{3, 3};
    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> output(9, 0.0f);
    cpu_stencil(input.data(), output.data(), params);
    assert(std::abs(output[4] - 5.0f) < 1e-4f);
  }

  std::cout << "CPU tests passed." << std::endl;
  return 0;
}
