# GPU Comparison Guide

## Architecture Overview

### Blackwell (2025)
- **RTX 5090**: Consumer flagship, SM 89, 32GB GDDR7
  - Gen 5 Tensor Cores
  - ~105 TFLOPS FP16
  - ~52.5 TFLOPS FP32
  - 1792 GB/s memory bandwidth
  - 575W TDP

### Hopper (2023-2024)
- **H100**: Datacenter, SM 90, 80GB HBM3
  - Gen 4 Tensor Cores
  - 989 TFLOPS FP16/BF16
  - 67 TFLOPS FP32
  - 3352 GB/s memory bandwidth
  - 700W TDP (SXM5)
  - FP8 support (1979 TFLOPS)

- **H200**: Datacenter refresh, SM 90a, 141GB HBM3e
  - Same compute as H100
  - 4800 GB/s memory bandwidth (+43%)
  - 141 GB HBM3e (largest in class)

## When to Use Which

| Use Case | Best GPU | Why |
|----------|----------|-----|
| Training LLM (>70B) | H200 | 141GB VRAM, highest bandwidth |
| Training LLM (<70B) | H100 | Best TFLOPS/$ ratio |
| Fine-tuning | H100 / RTX 5090 | Depends on model size |
| Inference | RTX 5090 | Cost-effective for single GPU |
| Research/Prototyping | RTX 5090 | Consumer price, pro features |
| Multi-GPU scaling | H100/H200 | NVLink 4.0, 900 GB/s |

## Memory Requirements

| Model Size | FP32 | FP16/BF16 | INT8 | INT4 |
|-----------|------|-----------|------|------|
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |
| 405B | 1.6 TB | 800 GB | 400 GB | 200 GB |

*Rule of thumb: Model params × bytes per param = VRAM needed*

## Bandwidth Comparison

```
H200 HBM3e  ████████████████████████████████████████ 4800 GB/s
H100 HBM3   ██████████████████████████████           3352 GB/s
RTX 5090    █████████████                            1792 GB/s
RTX 4090    ██████████                               1008 GB/s
A100 HBM2e  ████████████                             2039 GB/s
```
