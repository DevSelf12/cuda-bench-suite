<p align="center">
  <img src="https://img.shields.io/badge/CUDA-12.x-green?style=flat-square&logo=nvidia" alt="CUDA 12.x">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License MIT">
  <img src="https://img.shields.io/badge/SM-89%20%7C%2090%20%7C%2090a-orange?style=flat-square" alt="SM Architectures">
  <img src="https://img.shields.io/github/stars/DevSelf12/cuda-bench-suite?style=flat-square" alt="Stars">
</p>

<h1 align="center">cuda-bench-suite</h1>

<p align="center">
  <b>CUDA Benchmark & Profiling Toolkit for Nvidia GPUs</b><br>
  RTX 5090 (Blackwell) · H100 (Hopper) · H200 (Hopper)<br>
  <sub>Memory bandwidth · Compute throughput · Tensor Core utilization · Multi-GPU scaling</sub>
</p>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      cuda-bench-suite                            │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Benchmark      │   Profiler       │   Reporter                │
│   Runner         │   Engine         │   Engine                  │
│                  │                  │                           │
│  ┌────────────┐  │  ┌────────────┐  │  ┌──────────────────────┐ │
│  │ matmul     │  │  │ Nsight     │  │  │ JSON / CSV / HTML    │ │
│  │ conv2d     │  │  │ CUPTI      │  │  │ comparison tables    │ │
│  │ reduction  │  │  │ nvml       │  │  │ roofline plots       │ │
│  │ bandwidth  │  │  │ custom     │  │  │ thermal timeline     │ │
│  │ tensorcore │  │  └────────────┘  │  └──────────────────────┘ │
│  │ nccl       │  │                  │                           │
│  └────────────┘  │                  │                           │
├──────────────────┴──────────────────┴───────────────────────────┤
│                    CUDA Kernels (.cu)                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ SM 89    │ │ SM 90    │ │ SM 90a   │ │ Generic  │           │
│  │ RTX 5090 │ │ H100     │ │ H200     │ │ Fallback │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Support

| GPU | Architecture | SM | VRAM | FP16 TFLOPS | BF16 TFLOPS | FP32 TFLOPS | Tensor Core |
|-----|-------------|-----|------|-------------|-------------|-------------|-------------|
| RTX 5090 | Blackwell | SM 89 | 32 GB GDDR7 | ~105 | ~105 | ~52.5 | Gen 5 |
| H100 SXM | Hopper | SM 90 | 80 GB HBM3 | 989 | 989 | 67 | Gen 4 |
| H200 SXM | Hopper | SM 90a | 141 GB HBM3e | 989 | 989 | 67 | Gen 4 |

## Quick Start

### Prerequisites

- CUDA Toolkit 12.4+
- Python 3.10+
- Nvidia GPU (SM 89 / 90 / 90a)
- GCC 11+ (Linux) or MSVC 2022 (Windows)

### Install

```bash
git clone https://github.com/DevSelf12/cuda-bench-suite.git
cd cuda-bench-suite
pip install -e ".[dev]"
```

### Run All Benchmarks

```bash
cubench run --config configs/h100_sxm.yaml
cubench run --config configs/rtx5090.yaml --output results/
```

### Run Single Benchmark

```bash
cubench matmul --precision fp16 --size 8192 --iterations 100
cubench bandwidth --direction host-to-device --size 1gb
cubench tensorcore --op bf16_gemm --m 4096 --n 4096 --k 4096
```

### Compare GPUs

```bash
cubench compare results/h100.json results/h200.json --format html
```

## Benchmark Suite

| Benchmark | Description | Metric | Kernel |
|-----------|-------------|--------|--------|
| `matmul` | Dense matrix multiplication | TFLOPS | `sgemm.cu`, `hgemm.cu` |
| `conv2d` | 2D convolution (NCHW) | TFLOPS | `conv2d_nchw.cu` |
| `reduction` | Parallel sum / max / argmax | GB/s | `reduce_warp.cu` |
| `bandwidth` | H2D / D2H / D2D transfer | GB/s | N/A (driver) |
| `tensorcore` | WMMA / MMA tensor ops | TFLOPS | `wmma_gemm.cu` |
| `nccl` | Multi-GPU all-reduce / all-gather | GB/s | N/A (NCCL) |
| `latency` | Kernel launch latency | μs | `launch_latency.cu` |
| `thermal` | Clock / temp / power under load | W / °C / MHz | N/A (NVML) |

## Configuration

```yaml
# configs/h100_sxm.yaml
device: 0
benchmarks:
  - name: matmul
    sizes: [1024, 2048, 4096, 8192, 16384]
    precisions: [fp16, bf16, fp32, tf32]
    iterations: 100
    warmup: 10

  - name: tensorcore
    ops: [bf16_gemm, fp16_gemm, tf32_gemm]
    sizes: [[4096, 4096, 4096], [8192, 8192, 8192]]

  - name: bandwidth
    directions: [h2d, d2h, d2d]
    sizes: [256mb, 512mb, 1gb, 4gb]

  - name: nccl
    operations: [allreduce, allgather]
    gpus: [0, 1, 2, 3, 4, 5, 6, 7]
    message_sizes: [1mb, 64mb, 1gb]

output:
  format: json
  path: results/h100_sxm/
  include_raw: true
```

## Project Structure

```
cuda-bench-suite/
├── src/cubench/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── config.py           # YAML config loader
│   ├── runner.py           # Benchmark orchestrator
│   ├── device.py           # GPU detection & capabilities
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── loader.py       # JIT kernel compiler
│   │   ├── sgemm.cu        # FP32 GEMM kernel
│   │   ├── hgemm.cu        # FP16 GEMM kernel
│   │   ├── wmma_gemm.cu    # Tensor Core WMMA kernel
│   │   ├── conv2d_nchw.cu  # Conv2D kernel
│   │   ├── reduce_warp.cu  # Warp-level reduction
│   │   └── launch_latency.cu
│   ├── profilers/
│   │   ├── __init__.py
│   │   ├── cupti_prof.py   # CUPTI activity tracing
│   │   ├── nvml_prof.py    # NVML power/thermal/clock
│   │   └── roofline.py     # Roofline model analysis
│   └── reporters/
│       ├── __init__.py
│       ├── json_report.py
│       ├── csv_report.py
│       └── html_report.py
├── benchmarks/             # Standalone benchmark scripts
│   ├── bench_matmul.py
│   ├── bench_bandwidth.py
│   ├── bench_tensorcore.py
│   └── bench_nccl.py
├── configs/                # GPU-specific configs
│   ├── h100_sxm.yaml
│   ├── h200_sxm.yaml
│   └── rtx5090.yaml
├── tests/
│   ├── test_matmul.py
│   ├── test_bandwidth.py
│   └── test_tensorcore.py
├── docs/
│   ├── getting_started.md
│   ├── gpu_comparison.md
│   └── adding_benchmarks.md
├── scripts/
│   ├── setup_env.sh
│   └── profile_ncu.sh
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── pyproject.toml
├── CITATION.cff
├── CONTRIBUTING.md
└── LICENSE
```

## Roofline Analysis

```
GFLOPS/s
  │
  │         H200 (HBM3e) ─────────────────────────────
  │         H100 (HBM3)  ─────────────────────
  │         RTX 5090      ────────────────
  │        ╱
  │       ╱  Compute-bound region
  │      ╱
  │     ╱
  │    ╱  Memory-bound region
  │   ╱
  │  ╱
  │ ╱
  └──────────────────────────────────────── Arithmetic Intensity
       0.1    1     10    100
```

## Multi-GPU Scaling

```bash
# Run NCCL benchmark on 8x H100
mpirun -np 8 cubench nccl --operation allreduce --sizes 1mb,64mb,1gb

# Output:
# allreduce  1MB   :  24.8 GB/s  (8 GPUs)
# allreduce  64MB  : 876.3 GB/s  (8 GPUs)
# allreduce  1GB   : 892.1 GB/s  (8 GPUs)
```

## Profiling Integration

```bash
# Nsight Compute profiling
cubench profile matmul --size 8192 --ncu --output profile.ncu-rep

# Nsight Systems timeline
cubench profile matmul --size 8192 --nsys --output profile.nsys-rep

# Custom roofline
cubench roofline --config configs/h100_sxm.yaml --output roofline.html
```

## Citation

```bibtex
@software{cuda_bench_suite,
  title     = {cuda-bench-suite: CUDA Benchmark \& Profiling Toolkit},
  author    = {DevSelf12},
  year      = {2025},
  url       = {https://github.com/DevSelf12/cuda-bench-suite},
  version   = {0.1.0}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [CUTLASS](https://github.com/NVIDIA/cutlass) for GEMM kernel templates
- [NCCL](https://github.com/NVIDIA/nccl) for multi-GPU primitives
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) for profiling
