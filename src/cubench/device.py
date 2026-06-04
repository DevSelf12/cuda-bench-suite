"""GPU device detection and capability reporting."""

import subprocess
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GPUDevice:
    index: int
    name: str
    sm_version: str
    sm_count: int
    vram_total_mb: int
    vram_type: str
    cuda_cores: int
    tensor_cores: int
    base_clock_mhz: int
    boost_clock_mhz: int
    memory_clock_mhz: int
    memory_bus_width: int
    tdp_watts: int
    compute_capability: tuple[int, int]
    l2_cache_kb: int
    ecc_enabled: bool
    mig_enabled: bool
    driver_version: str
    cuda_version: str


# Known GPU specs database
GPU_SPECS = {
    "RTX 5090": {
        "sm_version": "sm_89", "cuda_cores": 21760, "tensor_cores": 680,
        "vram_type": "GDDR7", "memory_bus_width": 512, "tdp_watts": 575,
        "compute_capability": (8, 9), "l2_cache_kb": 73728,
    },
    "H100": {
        "sm_version": "sm_90", "cuda_cores": 16896, "tensor_cores": 528,
        "vram_type": "HBM3", "memory_bus_width": 5120, "tdp_watts": 700,
        "compute_capability": (9, 0), "l2_cache_kb": 51200,
    },
    "H200": {
        "sm_version": "sm_90a", "cuda_cores": 16896, "tensor_cores": 528,
        "vram_type": "HBM3e", "memory_bus_width": 6144, "tdp_watts": 700,
        "compute_capability": (9, 0), "l2_cache_kb": 51200,
    },
    "RTX 4090": {
        "sm_version": "sm_89", "cuda_cores": 16384, "tensor_cores": 512,
        "vram_type": "GDDR6X", "memory_bus_width": 384, "tdp_watts": 450,
        "compute_capability": (8, 9), "l2_cache_kb": 73728,
    },
    "A100": {
        "sm_version": "sm_80", "cuda_cores": 6912, "tensor_cores": 432,
        "vram_type": "HBM2e", "memory_bus_width": 5120, "tdp_watts": 400,
        "compute_capability": (8, 0), "l2_cache_kb": 40960,
    },
}


def _detect_gpu_pynvml() -> Optional[dict]:
    """Detect GPU using pynvml (NVML bindings)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": name,
            "vram_total_mb": mem.total // (1024 * 1024),
            "index": 0,
        }
    except Exception:
        return None


def _detect_gpu_nvidia_smi() -> Optional[dict]:
    """Fallback: detect GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "name": parts[0],
            "vram_total_mb": int(parts[1]),
            "compute_cap": parts[2],
            "index": 0,
        }
    except Exception:
        return None


def get_device_info(device_index: int = 0) -> dict:
    """Get comprehensive GPU device information."""
    info = _detect_gpu_pynvml() or _detect_gpu_nvidia_smi()
    if info is None:
        return {"error": "No Nvidia GPU detected", "available": False}

    name = info["name"]
    # Match against known specs
    specs = None
    for key, val in GPU_SPECS.items():
        if key in name:
            specs = val
            break

    result = {
        "available": True,
        "name": name,
        "vram_total_mb": info["vram_total_mb"],
        "device_index": device_index,
    }

    if specs:
        result.update(specs)
        result["recognized"] = True
    else:
        result["recognized"] = False
        result["note"] = "GPU not in database. Add specs to device.py GPU_SPECS."

    return result


def get_sm_arch(name: str) -> str:
    """Return SM architecture string for a GPU name."""
    for key, val in GPU_SPECS.items():
        if key in name:
            return val["sm_version"]
    return "sm_70"  # generic fallback
