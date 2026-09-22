"""Throughput, memory, and scaling bookkeeping.

Every number the dashboard shows is produced here and written to runs/*.json.
Nothing in the report is typed by hand.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    test_loss: float
    test_acc: float
    seconds: float
    images_per_sec: float
    lr: float


@dataclass
class RunRecord:
    """One training run. Serialised to runs/<name>.json."""

    name: str
    world_size: int
    device_type: str
    backend: str | None
    precision: str
    compiled: bool
    strategy: str
    batch_size_per_rank: int
    grad_accum_steps: int
    effective_batch: int
    epochs: int
    model_params: int
    subset_fraction: float = 1.0
    amp_dtype: str = ""
    hardware: str = ""
    data_source: str = ""
    memory_source: str = ""
    threads_per_rank: int = 0
    cpu_cores: int = 0
    torch_version: str = ""
    git_commit: str = ""
    command: str = ""
    epochs_log: list[EpochRecord] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    total_seconds: float = 0.0
    median_images_per_sec: float = 0.0
    best_test_acc: float = 0.0
    finished: bool = False

    def finalise(self) -> None:
        if self.epochs_log:
            rates = sorted(e.images_per_sec for e in self.epochs_log)
            mid = len(rates) // 2
            self.median_images_per_sec = (
                rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2
            )
            self.best_test_acc = max(e.test_acc for e in self.epochs_log)
            self.total_seconds = sum(e.seconds for e in self.epochs_log)
        self.finished = True

    def write(self, out_dir: str | Path = "runs") -> Path:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def describe_hardware(device: torch.device) -> str:
    if device.type == "cuda":
        n = torch.cuda.device_count()
        return f"{n}x {torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return f"Apple {platform.machine()} (MPS)"
    return f"CPU {platform.processor() or platform.machine()}"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def peak_memory_mb(device: torch.device) -> float:
    """True allocator peak, or 0.0 when the backend cannot report one.

    Only CUDA keeps a high water mark. MPS exposes `current_allocated_memory`
    and `driver_allocated_memory`, both of which are instantaneous readings
    taken after training already released its activations. Reporting either
    one in a column headed "peak" would be inventing a measurement, so MPS and
    CPU report nothing and the dashboard renders n/a.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def memory_source(device: torch.device) -> str:
    if device.type == "cuda":
        return "torch.cuda.max_memory_allocated"
    return f"unavailable on {device.type}: no allocator high water mark"


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    """Without this every timing number is a lie, because the queue is async."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Stopwatch:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.t0 = 0.0

    def __enter__(self) -> "Stopwatch":
        synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self.t0
