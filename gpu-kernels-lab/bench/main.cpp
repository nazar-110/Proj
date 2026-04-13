#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cuda_utils.h"
#include "common/types.h"
#include "kernels.h"
#include "ops.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#endif

namespace gkl {

struct Args {
  std::string op = "gemm";
  std::string kernel = "naive";
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int rows = 1024;
  int cols = 1024;
  int height = 1024;
  int width = 1024;
  int iters = 100;
  int warmup = 10;
  int repeats = 5;
  DType dtype = DType::kF32;
};

class ArgParser {
 public:
  explicit ArgParser(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key.rfind("--", 0) == 0 && i + 1 < argc) {
        args_[key.substr(2)] = argv[++i];
      }
    }
  }

  bool has(const std::string &key) const {
    return args_.find(key) != args_.end();
  }

  std::string get(const std::string &key, const std::string &def) const {
    auto it = args_.find(key);
    return it == args_.end() ? def : it->second;
  }

  int get_int(const std::string &key, int def) const {
    auto it = args_.find(key);
    return it == args_.end() ? def : std::stoi(it->second);
  }

 private:
  std::unordered_map<std::string, std::string> args_;
};

Args parse_args(int argc, char **argv) {
  ArgParser parser(argc, argv);
  Args args;
  args.op = parser.get("op", args.op);
  args.kernel = parser.get("kernel", args.kernel);
  args.m = parser.get_int("m", args.m);
  args.n = parser.get_int("n", args.n);
  args.k = parser.get_int("k", args.k);
  args.rows = parser.get_int("rows", args.rows);
  args.cols = parser.get_int("cols", args.cols);
  args.height = parser.get_int("height", args.height);
  args.width = parser.get_int("width", args.width);
  args.iters = parser.get_int("iters", args.iters);
  args.warmup = parser.get_int("warmup", args.warmup);
  args.repeats = parser.get_int("repeats", args.repeats);
  std::string dtype = parser.get("dtype", "fp32");
  if (dtype == "fp16") {
    args.dtype = DType::kF16;
  }
  return args;
}

struct Stats {
  double mean_ms = 0.0;
  double std_ms = 0.0;
};

struct Tolerance {
  float abs = 1e-3f;
  float rel = 1e-3f;
};

Stats summarize(const std::vector<float> &samples) {
  Stats stats{};
  if (samples.empty()) return stats;
  double sum = 0.0;
  for (float val : samples) sum += val;
  stats.mean_ms = sum / samples.size();
  double var = 0.0;
  for (float val : samples) {
    double diff = val - stats.mean_ms;
    var += diff * diff;
  }
  stats.std_ms = std::sqrt(var / samples.size());
  return stats;
}

float random_uniform() {
  static std::mt19937 gen(42);
  static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(gen);
}

Tolerance get_tolerance(const std::string &op, DType dtype) {
  if (dtype == DType::kF16) {
    if (op == "softmax") return {5e-2f, 5e-2f};
    return {2e-2f, 2e-2f};
  }
  if (op == "softmax") return {1e-3f, 1e-3f};
  return {1e-4f, 1e-4f};
}

#ifdef USE_CUDA
void *device_alloc(size_t bytes) {
  void *ptr = nullptr;
  GKL_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  return ptr;
}

void device_free(void *ptr) { GKL_CUDA_CHECK(cudaFree(ptr)); }
#endif

} // namespace gkl

int main(int argc, char **argv) {
  using namespace gkl;
  Args args = parse_args(argc, argv);

  if (args.iters <= 0 || args.warmup < 0 || args.repeats <= 0) {
    std::cerr << "iters, warmup, repeats must be positive." << std::endl;
    return 1;
  }

  std::cout << "gpu-kernels-lab benchmark" << std::endl;
  print_device_header();

#ifndef USE_CUDA
  std::cout << "CUDA is not enabled in this build. "
               "Reconfigure with a CUDA toolchain to run GPU benchmarks."
            << std::endl;
  return 0;
#endif

  std::vector<float> times;
  ErrorStats error{};
  bool supported = true;
  std::string message;

  if (args.op == "gemm") {
    bool fused = args.kernel == "fused";
    GemmParams params{args.m, args.n, args.k, fused, fused};
    size_t a_elems = static_cast<size_t>(params.m) * params.k;
    size_t b_elems = static_cast<size_t>(params.k) * params.n;
    size_t c_elems = static_cast<size_t>(params.m) * params.n;
    std::vector<float> h_a(a_elems);
    std::vector<float> h_b(b_elems);
    std::vector<float> h_c(c_elems, 0.0f);
    std::vector<float> h_ref(c_elems, 0.0f);
    std::vector<float> h_bias(params.n, 0.0f);
    for (auto &val : h_a) val = random_uniform();
    for (auto &val : h_b) val = random_uniform();
    for (auto &val : h_bias) val = random_uniform();

    cpu_gemm(h_a.data(), h_b.data(), h_bias.data(), h_ref.data(), params);

#ifdef USE_CUDA
    size_t bytes_a = a_elems * dtype_size(args.dtype);
    size_t bytes_b = b_elems * dtype_size(args.dtype);
    size_t bytes_c = c_elems * dtype_size(args.dtype);
    size_t bytes_bias = params.n * dtype_size(args.dtype);
    void *d_a = device_alloc(bytes_a);
    void *d_b = device_alloc(bytes_b);
    void *d_c = device_alloc(bytes_c);
    void *d_bias = device_alloc(bytes_bias);
    nvtxRangePushA("H2D");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_a_half(a_elems);
      std::vector<__half> h_b_half(b_elems);
      std::vector<__half> h_bias_half(params.n);
      for (size_t i = 0; i < a_elems; ++i) h_a_half[i] = __float2half(h_a[i]);
      for (size_t i = 0; i < b_elems; ++i) h_b_half[i] = __float2half(h_b[i]);
      for (int i = 0; i < params.n; ++i) h_bias_half[i] = __float2half(h_bias[i]);
      GKL_CUDA_CHECK(cudaMemcpy(d_a, h_a_half.data(), bytes_a,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_b, h_b_half.data(), bytes_b,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_bias, h_bias_half.data(), bytes_bias,
                                cudaMemcpyHostToDevice));
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_a,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_b,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bytes_bias,
                                cudaMemcpyHostToDevice));
    }
    nvtxRangePop();

    GemmKernel kernel = GemmKernel::kNaive;
    if (args.kernel == "tiled") kernel = GemmKernel::kTiled;
    if (args.kernel == "vectorized") kernel = GemmKernel::kVectorized;
    if (args.kernel == "fused") kernel = GemmKernel::kFused;
    if (args.kernel == "autotune") kernel = GemmKernel::kAutotune;

    for (int r = 0; r < args.repeats; ++r) {
      float ms = 0.0f;
      KernelResult result = launch_gemm(d_a, d_b, d_bias, d_c, params, args.dtype,
                                        kernel, args.iters, args.warmup, &ms);
      supported = result.supported;
      message = result.message;
      if (!supported) break;
      times.push_back(ms);
    }

    nvtxRangePushA("D2H");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_c_half(c_elems);
      GKL_CUDA_CHECK(cudaMemcpy(h_c_half.data(), d_c, bytes_c,
                                cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < c_elems; ++i) {
        h_c[i] = __half2float(h_c_half[i]);
      }
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes_c,
                                cudaMemcpyDeviceToHost));
    }
    nvtxRangePop();

    device_free(d_a);
    device_free(d_b);
    device_free(d_c);
    device_free(d_bias);
#endif
    error = compare_tensors(h_ref, h_c);
    Tolerance tol = get_tolerance(args.op, args.dtype);
    bool ok = error.max_abs <= tol.abs && error.max_rel <= tol.rel;
    double flops = 2.0 * params.m * params.n * params.k;
    Stats stats = summarize(times);
    std::cout << "Op: GEMM" << std::endl;
    if (!supported) {
      std::cout << "Kernel not supported: " << message << std::endl;
      return 1;
    }
    if (!message.empty()) {
      std::cout << "Note: " << message << std::endl;
    }
    std::cout << "Latency (ms): mean=" << stats.mean_ms
              << " std=" << stats.std_ms << std::endl;
    std::cout << "Throughput (TFLOPs/s): "
              << flops / (stats.mean_ms * 1e-3) / 1e12 << std::endl;
    std::cout << "Max abs error: " << error.max_abs
              << " max rel error: " << error.max_rel << std::endl;
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << std::endl;
  } else if (args.op == "softmax") {
    SoftmaxParams params{args.rows, args.cols};
    size_t elems = static_cast<size_t>(params.rows) * params.cols;
    std::vector<float> h_in(elems);
    std::vector<float> h_out(elems, 0.0f);
    std::vector<float> h_ref(elems, 0.0f);
    for (auto &val : h_in) val = random_uniform();
    cpu_softmax(h_in.data(), h_ref.data(), params);

#ifdef USE_CUDA
    size_t bytes = elems * dtype_size(args.dtype);
    void *d_in = device_alloc(bytes);
    void *d_out = device_alloc(bytes);
    nvtxRangePushA("H2D");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_in_half(elems);
      for (size_t i = 0; i < elems; ++i) h_in_half[i] = __float2half(h_in[i]);
      GKL_CUDA_CHECK(cudaMemcpy(d_in, h_in_half.data(), bytes,
                                cudaMemcpyHostToDevice));
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes,
                                cudaMemcpyHostToDevice));
    }
    nvtxRangePop();

    SoftmaxKernel kernel =
        args.kernel == "warp" ? SoftmaxKernel::kWarp : SoftmaxKernel::kBaseline;
    for (int r = 0; r < args.repeats; ++r) {
      float ms = 0.0f;
      KernelResult result = launch_softmax(d_in, d_out, params, args.dtype,
                                           kernel, args.iters, args.warmup, &ms);
      supported = result.supported;
      message = result.message;
      if (!supported) break;
      times.push_back(ms);
    }

    nvtxRangePushA("D2H");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_out_half(elems);
      GKL_CUDA_CHECK(cudaMemcpy(h_out_half.data(), d_out, bytes,
                                cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < elems; ++i) {
        h_out[i] = __half2float(h_out_half[i]);
      }
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes,
                                cudaMemcpyDeviceToHost));
    }
    nvtxRangePop();

    device_free(d_in);
    device_free(d_out);
#endif
    error = compare_tensors(h_ref, h_out);
    Tolerance tol = get_tolerance(args.op, args.dtype);
    bool ok = error.max_abs <= tol.abs && error.max_rel <= tol.rel;
    Stats stats = summarize(times);
    double bytes_moved = 2.0 * elems * dtype_size(args.dtype);
    std::cout << "Op: Softmax" << std::endl;
    if (!supported) {
      std::cout << "Kernel not supported: " << message << std::endl;
      return 1;
    }
    std::cout << "Latency (ms): mean=" << stats.mean_ms
              << " std=" << stats.std_ms << std::endl;
    std::cout << "Bandwidth (GB/s): "
              << bytes_moved / (stats.mean_ms * 1e-3) / 1e9 << std::endl;
    std::cout << "Max abs error: " << error.max_abs
              << " max rel error: " << error.max_rel << std::endl;
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << std::endl;
  } else if (args.op == "layernorm") {
    LayerNormParams params{args.rows, args.cols, 1e-5f};
    size_t elems = static_cast<size_t>(params.rows) * params.cols;
    std::vector<float> h_in(elems);
    std::vector<float> h_out(elems, 0.0f);
    std::vector<float> h_ref(elems, 0.0f);
    std::vector<float> h_gamma(params.cols, 1.0f);
    std::vector<float> h_beta(params.cols, 0.0f);
    for (auto &val : h_in) val = random_uniform();
    cpu_layernorm(h_in.data(), h_ref.data(), h_gamma.data(), h_beta.data(),
                  params);

#ifdef USE_CUDA
    size_t bytes = elems * dtype_size(args.dtype);
    size_t bytes_affine = params.cols * dtype_size(args.dtype);
    void *d_in = device_alloc(bytes);
    void *d_out = device_alloc(bytes);
    void *d_gamma = device_alloc(bytes_affine);
    void *d_beta = device_alloc(bytes_affine);
    nvtxRangePushA("H2D");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_in_half(elems);
      std::vector<__half> h_gamma_half(params.cols);
      std::vector<__half> h_beta_half(params.cols);
      for (size_t i = 0; i < elems; ++i) h_in_half[i] = __float2half(h_in[i]);
      for (int i = 0; i < params.cols; ++i) {
        h_gamma_half[i] = __float2half(h_gamma[i]);
        h_beta_half[i] = __float2half(h_beta[i]);
      }
      GKL_CUDA_CHECK(cudaMemcpy(d_in, h_in_half.data(), bytes,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma_half.data(), bytes_affine,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_beta, h_beta_half.data(), bytes_affine,
                                cudaMemcpyHostToDevice));
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), bytes_affine,
                                cudaMemcpyHostToDevice));
      GKL_CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), bytes_affine,
                                cudaMemcpyHostToDevice));
    }
    nvtxRangePop();

    LayerNormKernel kernel =
        args.kernel == "fused" ? LayerNormKernel::kFused
                                 : LayerNormKernel::kBaseline;
    for (int r = 0; r < args.repeats; ++r) {
      float ms = 0.0f;
      KernelResult result = launch_layernorm(d_in, d_out, d_gamma, d_beta,
                                             params, args.dtype, kernel,
                                             args.iters, args.warmup, &ms);
      supported = result.supported;
      message = result.message;
      if (!supported) break;
      times.push_back(ms);
    }

    nvtxRangePushA("D2H");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_out_half(elems);
      GKL_CUDA_CHECK(cudaMemcpy(h_out_half.data(), d_out, bytes,
                                cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < elems; ++i) {
        h_out[i] = __half2float(h_out_half[i]);
      }
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes,
                                cudaMemcpyDeviceToHost));
    }
    nvtxRangePop();

    device_free(d_in);
    device_free(d_out);
    device_free(d_gamma);
    device_free(d_beta);
#endif
    error = compare_tensors(h_ref, h_out);
    Tolerance tol = get_tolerance(args.op, args.dtype);
    bool ok = error.max_abs <= tol.abs && error.max_rel <= tol.rel;
    Stats stats = summarize(times);
    double bytes_moved = 2.0 * elems * dtype_size(args.dtype);
    std::cout << "Op: LayerNorm" << std::endl;
    if (!supported) {
      std::cout << "Kernel not supported: " << message << std::endl;
      return 1;
    }
    std::cout << "Latency (ms): mean=" << stats.mean_ms
              << " std=" << stats.std_ms << std::endl;
    std::cout << "Bandwidth (GB/s): "
              << bytes_moved / (stats.mean_ms * 1e-3) / 1e9 << std::endl;
    std::cout << "Max abs error: " << error.max_abs
              << " max rel error: " << error.max_rel << std::endl;
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << std::endl;
  } else if (args.op == "stencil") {
    StencilParams params{args.height, args.width};
    size_t elems = static_cast<size_t>(params.height) * params.width;
    std::vector<float> h_in(elems);
    std::vector<float> h_out(elems, 0.0f);
    std::vector<float> h_ref(elems, 0.0f);
    for (auto &val : h_in) val = random_uniform();
    cpu_stencil(h_in.data(), h_ref.data(), params);

#ifdef USE_CUDA
    size_t bytes = elems * dtype_size(args.dtype);
    void *d_in = device_alloc(bytes);
    void *d_out = device_alloc(bytes);
    nvtxRangePushA("H2D");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_in_half(elems);
      for (size_t i = 0; i < elems; ++i) h_in_half[i] = __float2half(h_in[i]);
      GKL_CUDA_CHECK(cudaMemcpy(d_in, h_in_half.data(), bytes,
                                cudaMemcpyHostToDevice));
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes,
                                cudaMemcpyHostToDevice));
    }
    nvtxRangePop();

    StencilKernel kernel =
        args.kernel == "shared" ? StencilKernel::kShared
                                 : StencilKernel::kBaseline;
    for (int r = 0; r < args.repeats; ++r) {
      float ms = 0.0f;
      KernelResult result = launch_stencil(d_in, d_out, params, args.dtype,
                                           kernel, args.iters, args.warmup, &ms);
      supported = result.supported;
      message = result.message;
      if (!supported) break;
      times.push_back(ms);
    }

    nvtxRangePushA("D2H");
    if (args.dtype == DType::kF16) {
      std::vector<__half> h_out_half(elems);
      GKL_CUDA_CHECK(cudaMemcpy(h_out_half.data(), d_out, bytes,
                                cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < elems; ++i) {
        h_out[i] = __half2float(h_out_half[i]);
      }
    } else {
      GKL_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes,
                                cudaMemcpyDeviceToHost));
    }
    nvtxRangePop();

    device_free(d_in);
    device_free(d_out);
#endif
    error = compare_tensors(h_ref, h_out);
    Tolerance tol = get_tolerance(args.op, args.dtype);
    bool ok = error.max_abs <= tol.abs && error.max_rel <= tol.rel;
    Stats stats = summarize(times);
    double bytes_moved = 2.0 * elems * dtype_size(args.dtype);
    std::cout << "Op: Stencil" << std::endl;
    if (!supported) {
      std::cout << "Kernel not supported: " << message << std::endl;
      return 1;
    }
    std::cout << "Latency (ms): mean=" << stats.mean_ms
              << " std=" << stats.std_ms << std::endl;
    std::cout << "Bandwidth (GB/s): "
              << bytes_moved / (stats.mean_ms * 1e-3) / 1e9 << std::endl;
    std::cout << "Max abs error: " << error.max_abs
              << " max rel error: " << error.max_rel << std::endl;
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << std::endl;
  } else {
    std::cerr << "Unknown op: " << args.op << std::endl;
    return 1;
  }

  return 0;
}
