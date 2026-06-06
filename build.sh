#!/usr/bin/env bash
# Build the LFM2.5 inference engine. Requires nvcc (CUDA >= 11) and an sm_86 GPU (RTX 3060).
# Author: g023 (https://github.com/g023/)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Prefer nvcc on PATH; override with NVCC=/path/to/nvcc ./build.sh
NVCC="${NVCC:-$(command -v nvcc || true)}"
[ -n "$NVCC" ] || { echo "[build] nvcc not found on PATH; set NVCC=/path/to/nvcc" >&2; exit 1; }
mkdir -p "$ROOT/build"
echo "[build] nvcc: $NVCC"
"$NVCC" -O3 -std=c++17 -arch=sm_86 -diag-suppress 1650 \
    -Xcompiler -Wno-unused-result \
    "$ROOT/src/engine.cu" -o "$ROOT/build/engine"
echo "[build] -> $ROOT/build/engine"

# Sparse Tensor-Core GEMM validation (mma.sp). Off the generation path: 2:4 on pretrained
# weights needs retraining to stay coherent (see kb/02_decisions.md), so this is a
# correctness harness for the kernel, not part of inference. 
"$NVCC" -O3 -std=c++17 -arch=sm_86 \
    "$ROOT/src/mma_sp_test.cu" -o "$ROOT/build/mma_sp_test" 2>/dev/null
echo "[build] -> $ROOT/build/mma_sp_test  (run to validate sparse Tensor-Core GEMM)"
