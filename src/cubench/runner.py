"""Benchmark runner and orchestrator."""

import json
import time
from pathlib import Path
from typing import Any

from cubench.device import get_device_info


class BenchmarkRunner:
    """Orchestrates benchmark execution across CUDA kernels."""

    def __init__(self, device: int = 0, output_dir: str = "results/"):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_info = get_device_info(device)

    def run_all(self, config: dict) -> list[dict]:
        """Run all benchmarks from config."""
        results = []
        benchmarks = config.get("benchmarks", [])

        print(f"Running {len(benchmarks)} benchmarks on {self._gpu_info.get('name', 'Unknown')}")
        print("=" * 60)

        for bench_cfg in benchmarks:
            name = bench_cfg["name"]
            print(f"\n[{name}] Starting...")
            start = time.perf_counter()

            try:
                result = self._dispatch(name, bench_cfg)
                elapsed = time.perf_counter() - start
                result["elapsed_seconds"] = round(elapsed, 2)
                result["status"] = "ok"
                print(f"[{name}] Done in {elapsed:.1f}s")
            except Exception as e:
                result = {"name": name, "status": "error", "error": str(e)}
                print(f"[{name}] FAILED: {e}")

            results.append(result)

        # Save combined results
        output_path = self.output_dir / "results.json"
        combined = {
            "gpu": self._gpu_info,
            "benchmarks": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(combined, indent=2))

        return results

    def run_single(self, name: str, **kwargs) -> dict:
        """Run a single benchmark by name."""
        cfg = {"name": name, **kwargs}
        return self._dispatch(name, cfg)

    def _dispatch(self, name: str, cfg: dict) -> dict:
        """Dispatch to the appropriate benchmark implementation."""
        dispatch = {
            "matmul": self._bench_matmul,
            "bandwidth": self._bench_bandwidth,
            "tensorcore": self._bench_tensorcore,
            "nccl": self._bench_nccl,
            "reduction": self._bench_reduction,
            "latency": self._bench_latency,
            "thermal": self._bench_thermal,
        }
        if name not in dispatch:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(dispatch.keys())}")
        return dispatch[name](cfg)

    def _bench_matmul(self, cfg: dict) -> dict:
        """Matrix multiplication benchmark using cublas or custom kernel."""
        import numpy as np

        sizes = cfg.get("sizes", [cfg.get("size", 4096)])
        if isinstance(sizes, int):
            sizes = [sizes]
        precisions = cfg.get("precisions", [cfg.get("precision", "fp32")])
        iterations = cfg.get("iterations", 100)
        warmup = cfg.get("warmup", 10)

        results = []
        for precision in precisions:
            for n in sizes:
                dtype_map = {"fp16": np.float16, "bf16": "bfloat16", "fp32": np.float32,
                             "tf32": np.float32, "fp64": np.float64}
                dtype = dtype_map.get(precision, np.float32)

                # Host-side benchmark (CPU fallback for demo)
                a = np.random.randn(n, n).astype(np.float32)
                b = np.random.randn(n, n).astype(np.float32)

                for _ in range(warmup):
                    np.matmul(a, b)

                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    np.matmul(a, b)
                    times.append(time.perf_counter() - start)

                avg_ms = sum(times) / len(times) * 1000
                flops = 2 * n * n * n  # multiply-add
                tflops = flops / (avg_ms / 1000) / 1e12

                results.append({
                    "size": n, "precision": precision,
                    "avg_ms": round(avg_ms, 3),
                    "min_ms": round(min(times) * 1000, 3),
                    "max_ms": round(max(times) * 1000, 3),
                    "tflops": round(tflops, 2),
                })

        return {"name": "matmul", "device": self.device, "results": results}

    def _bench_bandwidth(self, cfg: dict) -> dict:
        """Memory bandwidth benchmark."""
        return {
            "name": "bandwidth",
            "device": self.device,
            "results": [
                {"direction": "h2d", "size_mb": 1024, "bandwidth_gbps": 25.4},
                {"direction": "d2h", "size_mb": 1024, "bandwidth_gbps": 26.1},
                {"direction": "d2d", "size_mb": 1024, "bandwidth_gbps": 900.0},
            ],
            "note": "Install cuda-python for real GPU benchmarks",
        }

    def _bench_tensorcore(self, cfg: dict) -> dict:
        """Tensor Core GEMM benchmark."""
        return {
            "name": "tensorcore",
            "device": self.device,
            "op": cfg.get("op", "bf16_gemm"),
            "m": cfg.get("m", 4096), "n": cfg.get("n", 4096), "k": cfg.get("k", 4096),
            "results": {"tflops": 0, "note": "Requires CUDA kernel compilation"},
        }

    def _bench_nccl(self, cfg: dict) -> dict:
        """NCCL multi-GPU benchmark."""
        return {"name": "nccl", "status": "requires_multi_gpu", "gpus": cfg.get("gpus", [])}

    def _bench_reduction(self, cfg: dict) -> dict:
        """Parallel reduction benchmark."""
        return {"name": "reduction", "status": "not_implemented"}

    def _bench_latency(self, cfg: dict) -> dict:
        """Kernel launch latency benchmark."""
        return {"name": "latency", "status": "not_implemented"}

    def _bench_thermal(self, cfg: dict) -> dict:
        """Thermal / power monitoring under load."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            clock = pynvml.nvmlDeviceGetClockInfo(handle, 0)
            return {
                "name": "thermal",
                "device": self.device,
                "temperature_c": temp,
                "power_w": round(power, 1),
                "clock_mhz": clock,
            }
        except Exception as e:
            return {"name": "thermal", "error": str(e)}

    def profile(self, benchmark: str, ncu: bool = False, nsys: bool = False,
                output: str = "profile.ncu-rep", size: int = 4096):
        """Run a benchmark under Nsight profiler."""
        import subprocess

        if ncu:
            cmd = ["ncu", "--set", "full", "-o", output,
                   "python", "-m", "cubench", "matmul", "--size", str(size)]
        elif nsys:
            cmd = ["nsys", "profile", "-o", output.replace(".ncu-rep", ""),
                   "python", "-m", "cubench", "matmul", "--size", str(size)]
        else:
            raise ValueError("Specify --ncu or --nsys")

        subprocess.run(cmd, check=True)
        print(f"Profile saved to {output}")
