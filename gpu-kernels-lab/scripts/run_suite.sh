#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/gpu_kernels_bench}

if [[ ! -x "${BIN}" ]]; then
  echo "Benchmark binary not found at ${BIN}. Build first." >&2
  exit 1
fi

echo "| Op | Kernel | DType | Shape | Latency (ms) | Throughput |"
echo "| --- | --- | --- | --- | --- | --- |"

run_and_parse() {
  local op=$1
  local kernel=$2
  local dtype=$3
  local shape=$4
  local cmd=$5
  local output
  output=$(eval "${BIN} ${cmd}")
  local latency
  local throughput
  latency=$(echo "${output}" | rg "Latency" | awk '{print $4}' | sed 's/mean=//')
  throughput=$(echo "${output}" | rg -e "Throughput" -e "Bandwidth" | awk '{print $3}')
  echo "| ${op} | ${kernel} | ${dtype} | ${shape} | ${latency} | ${throughput} |"
}

run_and_parse GEMM tiled fp32 "4096x4096x4096" "--op gemm --m 4096 --n 4096 --k 4096 --dtype fp32 --kernel tiled --iters 100 --warmup 10"
run_and_parse Softmax warp fp16 "4096x4096" "--op softmax --rows 4096 --cols 4096 --dtype fp16 --kernel warp --iters 100 --warmup 10"
run_and_parse LayerNorm fused fp16 "4096x4096" "--op layernorm --rows 4096 --cols 4096 --dtype fp16 --kernel fused --iters 100 --warmup 10"
run_and_parse Stencil shared fp32 "4096x4096" "--op stencil --height 4096 --width 4096 --dtype fp32 --kernel shared --iters 100 --warmup 10"
