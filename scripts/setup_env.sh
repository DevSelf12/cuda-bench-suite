#!/bin/bash
# setup_env.sh — Set up CUDA development environment for cuda-bench-suite

set -e

echo "=== cuda-bench-suite Environment Setup ==="

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA Toolkit not found. Install from https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CUDA Toolkit: $(nvcc --version | grep release)"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# Check GPU
echo ""
echo "=== Detected GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# Check Python
echo ""
echo "=== Python ==="
python3 --version

# Install package
echo ""
echo "=== Installing cuda-bench-suite ==="
pip install -e ".[dev]"

echo ""
echo "=== Setup complete ==="
echo "Run: cubench info"
