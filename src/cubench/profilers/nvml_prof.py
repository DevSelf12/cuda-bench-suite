"""NVML-based GPU monitoring (power, temperature, clocks, utilization)."""

import time
from typing import Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class NVMLProfiler:
    """Monitor GPU metrics via NVML during benchmark execution."""

    def __init__(self, device_index: int = 0):
        self.device = device_index
        self.handle = None
        self._samples: list[dict] = []
        self._running = False

    def start(self):
        if not HAS_NVML:
            raise RuntimeError("pynvml not installed. pip install pynvml")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        self._running = True
        self._samples = []

    def sample(self) -> dict:
        """Take a single measurement."""
        if not self._running or not self.handle:
            return {}

        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, 0)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            clocks = pynvml.nvmlDeviceGetClockInfo(self.handle, 0)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, 1)
            fan = pynvml.nvmlDeviceGetFanSpeed(self.handle)

            sample = {
                "timestamp": time.time(),
                "temp_c": temp,
                "power_w": round(power, 1),
                "gpu_util": util.gpu,
                "mem_util": util.memory,
                "clock_mhz": clocks,
                "mem_clock_mhz": mem_clock,
                "fan_pct": fan,
            }
            self._samples.append(sample)
            return sample
        except Exception as e:
            return {"error": str(e)}

    def stop(self) -> dict:
        """Stop monitoring and return summary."""
        self._running = False
        if not self._samples:
            return {"error": "No samples collected"}

        temps = [s["temp_c"] for s in self._samples if "temp_c" in s]
        powers = [s["power_w"] for s in self._samples if "power_w" in s]
        clocks = [s["clock_mhz"] for s in self._samples if "clock_mhz" in s]

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return {
            "samples": len(self._samples),
            "temp_range": (min(temps), max(temps)) if temps else None,
            "avg_power_w": round(sum(powers) / len(powers), 1) if powers else None,
            "max_power_w": max(powers) if powers else None,
            "avg_clock_mhz": round(sum(clocks) / len(clocks)) if clocks else None,
            "min_clock_mhz": min(clocks) if clocks else None,
            "raw_samples": self._samples,
        }
