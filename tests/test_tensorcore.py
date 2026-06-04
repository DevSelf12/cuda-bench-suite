"""Tests for Tensor Core benchmark."""

import pytest
from cubench.runner import BenchmarkRunner


@pytest.fixture
def runner():
    return BenchmarkRunner(device=0, output_dir="/tmp/cubench_test/")


class TestTensorCore:
    def test_tensorcore_op(self, runner):
        result = runner.run_single("tensorcore", op="bf16_gemm", m=512, n=512, k=512)
        assert result["name"] == "tensorcore"
        assert "op" in result
