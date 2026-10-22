"""Environment capture and append-only experiment records."""

import hashlib
import json
import os
import platform
import resource
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[2]
# 1: original schema. 2: training/evaluation records carry deterministic model
# identity digests and explicit hyperparameters; benchmarks keep raw samples.
RECORD_VERSION = 2


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict:
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        # Uncommitted tracked changes mean the commit does not fully describe the code.
        "dirty": bool(status) if status is not None else None,
    }


def environment(device: str) -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "memory_bytes": int(_run(["sysctl", "-n", "hw.memsize"]) or 0) or None,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "git": git_state(),
        "command": shlex.join([Path(sys.executable).name, *sys.argv]),
    }
    if device == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name()
    return info


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def peak_rss_bytes() -> int:
    """Process lifetime peak resident set size (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class PeakAllocated(TorchDispatchMode):
    """Sample the MPS allocator's live bytes after every aten op (forward and backward)."""

    def __init__(self) -> None:
        super().__init__()
        self.peak = torch.mps.current_allocated_memory()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        return output


def measure_peak_memory(device: str, operation) -> tuple[int, str]:
    """Extra memory used by operation() and how it was measured.

    cuda: device allocator peak (torch.cuda.max_memory_allocated) minus allocation before.
    mps: live allocator bytes sampled after every op minus allocation before.
    cpu: process peak-RSS delta (host memory; includes allocator caching).
    Host RSS is never reported for an accelerator.
    """
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        operation()
        synchronize(device)
        return torch.cuda.max_memory_allocated() - before, "cuda_max_memory_allocated_delta"
    if device == "mps":
        before = torch.mps.current_allocated_memory()
        with PeakAllocated() as tracker:
            operation()
        synchronize(device)
        return tracker.peak - before, "mps_live_allocated_peak_per_op"
    if device != "cpu":
        raise ValueError(f"no peak-memory method for device {device!r}")
    before = peak_rss_bytes()
    operation()
    return peak_rss_bytes() - before, "process_peak_rss_delta"


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        position = fraction * (count - 1)
        low = int(position)
        high = min(low + 1, count - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    mean = statistics.fmean(ordered)
    return {
        "count": count,
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "iqr": quantile(0.75) - quantile(0.25),
        "mean": mean,
        "cv": statistics.stdev(ordered) / mean if count > 1 and mean else 0.0,
    }


def write_record(directory: str | Path, name: str, payload: dict) -> Path:
    """Write a new JSON record. Existing records are never overwritten."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = (payload.get("environment", {}).get("git", {}).get("commit") or "nogit")[:7]
    path = directory / f"{stamp}_{name}_{commit}.json"
    record = {"record_version": RECORD_VERSION, "created_utc": stamp, **payload}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return path
