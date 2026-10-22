"""Peak-memory methods are device-specific; host RSS is never reported for an accelerator."""

import pytest
import torch

from minilm_lab import provenance


def test_cpu_uses_host_peak_rss() -> None:
    extra, method = provenance.measure_peak_memory("cpu", lambda: torch.ones(4 * 2**20).sum())
    assert method == "process_peak_rss_delta" and extra >= 0


def test_cuda_uses_device_allocator_peak_not_host_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = {"allocated": 1000, "peak": 1000}

    def allocate() -> None:
        counters["peak"] = 1000 + 5_000_000

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: counters.update(peak=counters["allocated"]))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: counters["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: counters["peak"])
    monkeypatch.setattr(provenance, "peak_rss_bytes", lambda: pytest.fail("host RSS used for CUDA"))
    assert provenance.measure_peak_memory("cuda", allocate) == (5_000_000, "cuda_max_memory_allocated_delta")


def test_mps_uses_live_allocator_peak(monkeypatch: pytest.MonkeyPatch, mps_device: str) -> None:
    monkeypatch.setattr(provenance, "peak_rss_bytes", lambda: pytest.fail("host RSS used for MPS"))
    extra, method = provenance.measure_peak_memory("mps", lambda: torch.ones(2**20, device="mps") * 2)
    assert method == "mps_live_allocated_peak_per_op" and extra >= 2 * 4 * 2**20


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        provenance.measure_peak_memory("xpu", lambda: None)
