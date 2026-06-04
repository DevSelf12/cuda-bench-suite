"""YAML configuration loader for benchmark suite."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load and validate benchmark configuration from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(p) as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config


def _validate(config: dict) -> None:
    """Minimal validation of config structure."""
    if "benchmarks" not in config:
        raise ValueError("Config missing 'benchmarks' section")

    for bench in config["benchmarks"]:
        if "name" not in bench:
            raise ValueError("Each benchmark must have a 'name' field")

    if "output" not in config:
        config["output"] = {"format": "json", "path": "results/"}


def get_default_config(gpu_name: str) -> dict:
    """Return sensible default config for a known GPU."""
    defaults = {
        "H100": _h100_config(),
        "H200": _h200_config(),
        "RTX 5090": _rtx5090_config(),
    }
    for key, cfg in defaults.items():
        if key in gpu_name:
            return cfg
    return _generic_config()


def _h100_config() -> dict:
    return {
        "device": 0,
        "benchmarks": [
            {"name": "matmul", "sizes": [2048, 4096, 8192, 16384],
             "precisions": ["fp16", "bf16", "fp32", "tf32"], "iterations": 100, "warmup": 10},
            {"name": "tensorcore", "ops": ["bf16_gemm", "fp16_gemm", "tf32_gemm"],
             "sizes": [[4096, 4096, 4096], [8192, 8192, 8192]]},
            {"name": "bandwidth", "directions": ["h2d", "d2h", "d2d"],
             "sizes": ["256mb", "1gb", "4gb"]},
        ],
        "output": {"format": "json", "path": "results/h100_sxm/"},
    }


def _h200_config() -> dict:
    cfg = _h100_config()
    cfg["output"]["path"] = "results/h200_sxm/"
    cfg["benchmarks"].append(
        {"name": "bandwidth", "directions": ["h2d", "d2h", "d2d"],
         "sizes": ["256mb", "1gb", "4gb", "8gb"], "note": "HBM3e extended range"}
    )
    return cfg


def _rtx5090_config() -> dict:
    cfg = _h100_config()
    cfg["output"]["path"] = "results/rtx5090/"
    # RTX 5090 has 32GB VRAM, limit large sizes
    for b in cfg["benchmarks"]:
        if b["name"] == "matmul":
            b["sizes"] = [1024, 2048, 4096, 8192]
    return cfg


def _generic_config() -> dict:
    return {
        "device": 0,
        "benchmarks": [
            {"name": "matmul", "sizes": [1024, 2048, 4096],
             "precisions": ["fp32"], "iterations": 50, "warmup": 5},
            {"name": "bandwidth", "directions": ["h2d", "d2h"],
             "sizes": ["256mb", "1gb"]},
        ],
        "output": {"format": "json", "path": "results/generic/"},
    }
