"""Tests for matrix multiplication benchmark."""

import numpy as np
import pytest
from cubench.runner import BenchmarkRunner


@pytest.fixture
def runner():
    return BenchmarkRunner(device=0, output_dir="/tmp/cubench_test/")


class TestMatmul:
    def test_basic_sizes(self, runner):
        """Test matmul with small sizes."""
        result = runner.run_single("matmul", size=256, precision="fp32", iterations=5, warmup=2)
        assert result["name"] == "matmul"
        assert len(result["results"]) > 0
        assert result["results"][0]["size"] == 256

    def test_tflops_positive(self, runner):
        """TFLOPS should be positive."""
        result = runner.run_single("matmul", size=512, precision="fp32", iterations=10, warmup=2)
        for r in result["results"]:
            assert r["tflops"] > 0

    def test_multiple_precisions(self, runner):
        """Should support fp16 and fp32."""
        for prec in ["fp16", "fp32"]:
            result = runner.run_single("matmul", size=256, precision=prec, iterations=5, warmup=2)
            assert result["results"][0]["precision"] == prec

    def test_larger_matrix(self, runner):
        """Test with 1024x1024."""
        result = runner.run_single("matmul", size=1024, precision="fp32", iterations=20, warmup=5)
        assert result["results"][0]["size"] == 1024
        assert result["results"][0]["avg_ms"] > 0
