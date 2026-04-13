# gpu-kernels-lab

A CUDA C++ mini-library + benchmarking CLI that compares multiple GPU kernel
implementations for GEMM, softmax, layer normalization, and stencil operations.
This repo is intentionally scoped as an NVIDIA internship-ready portfolio
project with emphasis on performance engineering, profiling, and correctness.

## What this demonstrates (NVIDIA internship focus)

- CUDA kernel design: baseline vs. optimized (shared memory, vectorization,
  warp-level primitives, and fusion).
- Profiling proficiency: NVTX ranges + Nsight Systems/Compute workflows.
- Portability: runtime device queries, architecture-agnostic CMake builds, and
  robust fallbacks when features are unavailable.
- Verification discipline: CPU reference implementations and configurable
  tolerances per op/dtype.

## Setup & Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

### CUDA architectures

By default, the project uses:

- `CMAKE_CUDA_ARCHITECTURES=native` when supported by your CMake version.
- A fallback list (`70;75;80;86;89`) for older CMake releases.

Override explicitly if needed:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
```

## First run

```bash
./gpu_kernels_bench --op gemm --m 4096 --n 4096 --k 4096 \
  --dtype fp32 --kernel tiled --iters 100 --warmup 10
```

Expected output format (example):

```
gpu-kernels-lab benchmark
GPU 0: NVIDIA RTX 4090
  Compute capability: 8.9
  SMs: 128
  Warp size: 32
  Shared memory per block: 64 KB
  Shared memory per SM: 228 KB
  Global memory: 24564 MB
Op: GEMM
Latency (ms): mean=2.15 std=0.04
Throughput (TFLOPs/s): 63.1
Max abs error: 0.0006 max rel error: 0.0009
Correctness: PASS
```

## Profiling guide

See [`docs/profiling.md`](docs/profiling.md) for concrete Nsight commands.

## Results

Fill in your best measurements below.

| Op | Kernel | DType | Shape | Latency (ms) | Throughput (TFLOPs/s or GB/s) |
| --- | --- | --- | --- | --- | --- |
| GEMM | tiled | fp32 | 4096x4096x4096 |  |  |
| Softmax | warp | fp16 | 4096x4096 |  |  |
| LayerNorm | fused | fp16 | 4096x4096 |  |  |
| Stencil | shared | fp32 | 4096x4096 |  |  |

## What I learned

- How shared-memory tiling increases arithmetic intensity for GEMM.
- The impact of memory coalescing and vectorized loads on throughput.
- Warp-level primitives and how they reduce synchronization overhead.
- Occupancy tradeoffs when balancing shared memory and registers.
- Numerical stability considerations in softmax and normalization kernels.

## Project structure

```
/gpu-kernels-lab
  /src
  /include
  /kernels
  /bench
  /tests
  /docs
  /scripts
  CMakeLists.txt
  README.md
  LICENSE
  CONTRIBUTING.md
  .clang-format
  .github/workflows/ci.yml
```
