"""Process group setup and device selection.

The honest part of this file is `resolve_backend`. There are three plausible
execution contexts for this project and only one of them can produce a real
speedup number:

  cuda, world_size > 1   nccl    real multi device, real scaling
  cpu,  world_size > 1   gloo    correct gradients, slower than 1 process
  mps,  world_size == 1  none    fastest single device on Apple silicon

MPS has no collective backend. gloo cannot all reduce a tensor that lives on
an MPS device and nccl is CUDA only. Rather than silently moving tensors to
CPU and reporting the result as distributed MPS training, we refuse.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


class UnsupportedTopology(RuntimeError):
    """Raised when the requested device and world size cannot be honoured."""


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def available_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_backend(device_type: str, world_size: int) -> str | None:
    """Pick a collective backend, or refuse with an explanation."""
    if world_size == 1:
        return None
    if device_type == "cuda":
        return "nccl"
    if device_type == "cpu":
        return "gloo"
    if device_type == "mps":
        raise UnsupportedTopology(
            "MPS cannot participate in a process group. gloo does not support "
            "MPS tensors and nccl is CUDA only. Run world_size=1 on mps for the "
            "single device baseline, or DVIT_DEVICE=cpu for the gloo correctness "
            "path, or run on CUDA for real distributed training."
        )
    raise UnsupportedTopology(f"unknown device type {device_type!r}")


def setup(device_override: str | None = None, timeout_s: int = 1800) -> DistContext:
    """Initialise the process group from torchrun environment variables.

    Works unlaunched too: absent torchrun, this returns a world_size 1 context.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device_type = device_override or os.environ.get("DVIT_DEVICE") or available_accelerator()
    backend = resolve_backend(device_type, world_size)

    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_type)

    if backend is not None and not dist.is_initialized():
        # Two rendezvous paths. torchrun's default is a TCPStore on localhost,
        # which is what you want on a real cluster. Some sandboxes and
        # hardened laptops block localhost TCP, and the symptom is not an
        # error but a silent hang until the timeout expires. When
        # DVIT_INIT_FILE is set we use a FileStore instead, which needs no
        # sockets at all. dvit.launch sets it.
        init_file = os.environ.get("DVIT_INIT_FILE")
        extra = (
            dict(init_method=f"file://{init_file}", rank=rank, world_size=world_size)
            if init_file else {}
        )
        dist.init_process_group(
            backend=backend,
            timeout=_dt.timedelta(seconds=timeout_s),
            **extra,
        )

    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: float, ctx: DistContext) -> float:
    """Average a python scalar across ranks. Returns the input when unlaunched."""
    if not ctx.is_distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cpu" if ctx.backend == "gloo" else ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / ctx.world_size)


def all_reduce_sum(value: float, ctx: DistContext) -> float:
    if not ctx.is_distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cpu" if ctx.backend == "gloo" else ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())
