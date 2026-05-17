"""NVIDIA GPU telemetry collector using pynvml.

This module captures GPU utilization, VRAM usage, temperature and power while
an inference request is running. It is safe to run on systems without NVML; in
that case metrics are returned as None instead of crashing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None


@dataclass
class GPUSample:
    timestamp: float
    gpu_util_pct: Optional[float]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]
    temperature_c: Optional[float]
    power_w: Optional[float]


class GPUMonitor:
    def __init__(self, device_index: int = 0, sample_interval_sec: float = 0.2):
        self.device_index = device_index
        self.sample_interval_sec = sample_interval_sec
        self.samples: List[GPUSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self.available = False

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.available = True
            except Exception:
                self.available = False

    def snapshot(self) -> GPUSample:
        if not self.available or self._handle is None:
            return GPUSample(time.time(), None, None, None, None, None)

        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)

        power_w = None
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000
        except Exception:
            power_w = None

        return GPUSample(
            timestamp=time.time(),
            gpu_util_pct=float(util.gpu),
            memory_used_mb=mem.used / 1024 / 1024,
            memory_total_mb=mem.total / 1024 / 1024,
            temperature_c=float(temp),
            power_w=power_w,
        )

    def _loop(self):
        while not self._stop_event.is_set():
            self.samples.append(self.snapshot())
            time.sleep(self.sample_interval_sec)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

        if not self.samples:
            self.samples.append(self.snapshot())

        def values(attr: str) -> List[float]:
            vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
            return vals

        gpu_vals = values("gpu_util_pct")
        mem_vals = values("memory_used_mb")
        temp_vals = values("temperature_c")
        power_vals = values("power_w")

        return {
            "gpu_available": self.available,
            "gpu_avg_util_pct": sum(gpu_vals) / len(gpu_vals) if gpu_vals else None,
            "gpu_max_util_pct": max(gpu_vals) if gpu_vals else None,
            "gpu_avg_memory_mb": sum(mem_vals) / len(mem_vals) if mem_vals else None,
            "gpu_max_memory_mb": max(mem_vals) if mem_vals else None,
            "gpu_total_memory_mb": self.samples[-1].memory_total_mb,
            "gpu_max_temperature_c": max(temp_vals) if temp_vals else None,
            "gpu_avg_power_w": sum(power_vals) / len(power_vals) if power_vals else None,
            "gpu_samples": len(self.samples),
        }
