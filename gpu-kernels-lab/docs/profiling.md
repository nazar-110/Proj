# Profiling Guide

This project is instrumented with NVTX ranges around H2D, kernel launches,
benchmark loops, D2H transfers, and verification.

## Nsight Systems

```bash
nsys profile --stats=true --output gkl_profile \
  ./gpu_kernels_bench --op gemm --m 4096 --n 4096 --k 4096 \
  --dtype fp32 --kernel tiled --iters 100 --warmup 10
```

## Nsight Compute

```bash
ncu --set full --target-processes all \
  ./gpu_kernels_bench --op softmax --rows 4096 --cols 4096 \
  --dtype fp16 --kernel warp --iters 100 --warmup 10
```

## Tips

- Use `--kernel autotune` for GEMM to evaluate multiple tile sizes.
- Combine `nsys` and `ncu` runs to correlate launch-level and kernel-level
  performance.
