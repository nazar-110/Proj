# Design Notes

## Overview

This mini-library implements multiple CUDA kernels per primitive, focusing on
clear correctness and progressive optimization.

## GEMM

- **Naive**: Direct global memory loads for A/B. Baseline correctness.
- **Tiled**: Shared memory tiling to improve reuse. Tile sizes 8/16/32 with a
  small autotune mode that selects the best tile size at runtime.
- **Vectorized**: Uses `float4` loads when input alignment and `N % 4 == 0`.
  Falls back when alignment requirements are not met.
- **Fused**: GEMM + bias + ReLU to demonstrate kernel fusion and reduced
  memory traffic.

Tradeoffs: tiled kernels increase shared memory usage but improve arithmetic
intensity; vectorized kernels improve memory throughput when aligned.

## Softmax

- **Baseline**: Numerically stable (subtract max) with block-level reductions.
- **Warp-optimized**: Uses warp-level reductions to reduce synchronization and
  shared-memory pressure.

## LayerNorm

- **Baseline**: Separate reductions for mean/variance.
- **Fused**: Combines mean/variance reductions and normalization within a
  single kernel for better cache locality and fewer launches.

## Stencil (2D)

- **Baseline**: Global memory only.
- **Shared**: Loads tile and halo into shared memory to improve locality.

## Portability

- No SM architecture is hardcoded. Default build uses `CMAKE_CUDA_ARCHITECTURES`
  set to `native` when supported, with a fallback list for older CMake.
- Runtime device queries guard optional optimizations and print GPU details in
  the benchmark header.
