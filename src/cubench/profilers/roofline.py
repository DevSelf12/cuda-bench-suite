"""Roofline model analysis for CUDA kernels.

Computes arithmetic intensity and compares against theoretical
peak FLOPS and memory bandwidth for the target GPU.
"""

from dataclasses import dataclass


@dataclass
class RooflinePoint:
    """A single measurement point on the roofline plot."""
    kernel_name: str
    flops: int
    bytes_transferred: int
    time_seconds: float

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPS per byte."""
        if self.bytes_transferred == 0:
            return float("inf")
        return self.flops / self.bytes_transferred

    @property
    def achieved_gflops(self) -> float:
        if self.time_seconds == 0:
            return 0
        return (self.flops / self.time_seconds) / 1e9


# Theoretical peak specs
GPU_PEAK = {
    "H100": {
        "fp32_tflops": 67, "fp16_tflops": 989, "bf16_tflops": 989,
        "bandwidth_gbps": 3352, "vram_gb": 80,
    },
    "H200": {
        "fp32_tflops": 67, "fp16_tflops": 989, "bf16_tflops": 989,
        "bandwidth_gbps": 4800, "vram_gb": 141,
    },
    "RTX 5090": {
        "fp32_tflops": 52.5, "fp16_tflops": 105, "bf16_tflops": 105,
        "bandwidth_gbps": 1792, "vram_gb": 32,
    },
}


def classify_point(point: RooflinePoint, gpu_name: str) -> str:
    """Classify a kernel as compute-bound or memory-bound."""
    peak = GPU_PEAK.get(gpu_name, GPU_PEAK["H100"])
    ridge_point = peak["fp32_tflops"] * 1000 / peak["bandwidth_gbps"]

    if point.arithmetic_intensity > ridge_point:
        return "compute-bound"
    else:
        return "memory-bound"


def roofline_efficiency(point: RooflinePoint, gpu_name: str, precision: str = "fp32") -> float:
    """Calculate efficiency as percentage of peak."""
    peak = GPU_PEAK.get(gpu_name, GPU_PEAK["H100"])
    peak_key = f"{precision}_tflops"
    peak_gflops = peak.get(peak_key, peak["fp32_tflops"]) * 1000
    return min(100.0, point.achieved_gflops / peak_gflops * 100)
