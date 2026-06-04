"""JIT kernel compiler and loader using cuda-python or pycuda."""

import os
from pathlib import Path

KERNEL_DIR = Path(__file__).parent


def get_kernel_path(name: str) -> Path:
    """Get path to a .cu kernel file."""
    path = KERNEL_DIR / f"{name}.cu"
    if not path.exists():
        raise FileNotFoundError(f"Kernel not found: {path}")
    return path


def list_kernels() -> list[str]:
    """List available CUDA kernel files."""
    return sorted(p.stem for p in KERNEL_DIR.glob("*.cu"))
