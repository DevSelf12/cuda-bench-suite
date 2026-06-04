#!/bin/bash
# profile_ncu.sh — Profile a benchmark with Nsight Compute

BENCHMARK=${1:-matmul}
SIZE=${2:-4096}
OUTPUT=${3:-profile.ncu-rep}

echo "Profiling $BENCHMARK (size=$SIZE) with Nsight Compute..."
echo "Output: $OUTPUT"

ncu --set full \
    --kernel-name "sgemm_kernel|hgemm_wmma_kernel|wmma_bf16_gemm" \
    -o "$OUTPUT" \
    python -m cubench "$BENCHMARK" --size "$SIZE" --iterations 1 --warmup 0

echo "Profile saved to $OUTPUT"
echo "Open with: ncu-ui $OUTPUT"
