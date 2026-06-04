# Getting Started

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 12.0 | 12.4+ |
| Python | 3.10 | 3.12 |
| GCC (Linux) | 11 | 13 |
| GPU | Any Nvidia (SM 70+) | H100 / H200 / RTX 5090 |
| VRAM | 8 GB | 80 GB+ |

## Installation

```bash
# Clone
git clone https://github.com/DevSelf12/cuda-bench-suite.git
cd cuda-bench-suite

# Install in editable mode
pip install -e ".[dev]"

# Verify
cubench info
```

## First Benchmark

```bash
# Quick matmul test
cubench matmul --precision fp32 --size 4096 --iterations 50

# Full suite from config
cubench run --config configs/h100_sxm.yaml
```

## Interpreting Results

### TFLOPS
- **FP32**: Standard floating point throughput
- **FP16/BF16**: Half-precision (2x throughput on Tensor Cores)
- **TF32**: Tensor Float 32 (Nvidia's hybrid format, SM 80+)

### Bandwidth
- **H2D**: Host (CPU) to Device (GPU) — limited by PCIe gen
- **D2H**: Device to Host — same as H2D
- **D2D**: Device to Device — limited by VRAM bandwidth

### Roofline
A kernel is:
- **Memory-bound** if arithmetic intensity < ridge point
- **Compute-bound** if arithmetic intensity > ridge point

Optimize memory-bound kernels with:
- Shared memory tiling
- Vectorized loads (float4)
- Memory coalescing

Optimize compute-bound kernels with:
- Tensor Cores (WMMA/MMA)
- Mixed precision
- Instruction-level parallelism
