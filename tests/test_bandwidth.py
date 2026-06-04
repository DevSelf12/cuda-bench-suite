"""Tests for bandwidth benchmark."""

import pytest
from cubench.runner import BenchmarkRunner


@pytest.fixture
def runner():
    return BenchmarkRunner(device=0, output_dir="/tmp/cubench_test/")


class TestBandwidth:
    def test_bandwidth_returns_results(self, runner):
        result = runner.run_single("bandwidth", direction="all", size="256mb")
        assert result["name"] == "bandwidth"
        assert "results" in result

    def test_directions_present(self, runner):
        result = runner.run_single("bandwidth", direction="all", size="256mb")
        directions = {r["direction"] for r in result["results"]}
        assert "h2d" in directions or "d2h" in directions or "d2d" in directions
