# Adding Custom Benchmarks

## Overview

cuda-bench-suite uses a plugin-like architecture. Each benchmark is a method
in `BenchmarkRunner` that returns a standardized result dict.

## Step-by-Step

### 1. Write the CUDA Kernel

Create a new `.cu` file in `src/cubench/kernels/`:

```cuda
// my_kernel.cu
__global__ void my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

extern "C" void launch_my_kernel(float* data, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    my_kernel<<<grid, block, 0, stream>>>(data, n);
}
```

### 2. Add Runner Method

In `src/cubench/runner.py`, add to `BenchmarkRunner`:

```python
def _bench_my_benchmark(self, cfg: dict) -> dict:
    """My custom benchmark."""
    size = cfg.get("size", 1024)
    iterations = cfg.get("iterations", 100)

    # Implementation here
    results = []

    return {"name": "my_benchmark", "device": self.device, "results": results}
```

### 3. Register in Dispatch

Add to `_dispatch()`:

```python
"my_benchmark": self._bench_my_benchmark,
```

### 4. Add Config Support

Add to your YAML config:

```yaml
benchmarks:
  - name: my_benchmark
    size: 4096
    iterations: 100
```

### 5. Write Tests

```python
def test_my_benchmark(self, runner):
    result = runner.run_single("my_benchmark", size=256, iterations=10)
    assert result["name"] == "my_benchmark"
    assert len(result["results"]) > 0
```

## Result Schema

Every benchmark must return:

```json
{
  "name": "benchmark_name",
  "device": 0,
  "results": [...],
  "status": "ok"
}
```
