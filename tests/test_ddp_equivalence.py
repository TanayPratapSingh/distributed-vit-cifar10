"""The load bearing test: does DDP compute the gradient it claims to?

DDP all reduces gradients with a mean. So two ranks each holding batch B must
produce the same gradient as one process holding the concatenated batch 2B.
If that identity does not hold, every throughput number in this repo is
measuring a system that trains a different model than the baseline, and the
comparison is meaningless.

This runs on gloo over CPU, which is exactly why the gloo path exists: it is
slower than single process, but it is the only way to check this on a laptop
with one GPU.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dvit.model import ViT, ViTConfig

# Dropout and stochastic depth would desynchronise the replicas' RNG and mask
# the thing under test, so the probe model is deterministic by construction.
PROBE = ViTConfig(dim=48, depth=2, heads=2, dropout=0.0, drop_path=0.0)
BATCH_PER_RANK = 8
SEED = 1234


def fixed_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return x, y


def flat_grads(model: torch.nn.Module) -> torch.Tensor:
    module = model.module if hasattr(model, "module") else model
    return torch.cat([
        p.grad.reshape(-1) for _, p in sorted(module.named_parameters())
        if p.grad is not None
    ])


def _worker(rank: int, world_size: int, out_path: str, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}",
        rank=rank, world_size=world_size,
    )
    torch.manual_seed(SEED)          # identical init on every rank
    model = ViT(PROBE)
    ddp = DDP(model)

    x, y = fixed_batch(BATCH_PER_RANK * world_size)
    shard_x = x[rank * BATCH_PER_RANK:(rank + 1) * BATCH_PER_RANK]
    shard_y = y[rank * BATCH_PER_RANK:(rank + 1) * BATCH_PER_RANK]

    loss = torch.nn.functional.cross_entropy(ddp(shard_x), shard_y)
    loss.backward()

    if rank == 0:
        torch.save(flat_grads(ddp), out_path)
    dist.barrier()
    dist.destroy_process_group()


def single_process_grads() -> torch.Tensor:
    torch.manual_seed(SEED)
    model = ViT(PROBE)
    x, y = fixed_batch(BATCH_PER_RANK * 2)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    return flat_grads(model)


@pytest.mark.parametrize("world_size", [2])
def test_ddp_gradients_match_single_process(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "grads.pt")
        init = os.path.join(td, "pg_init")
        mp.spawn(_worker, args=(world_size, out, init), nprocs=world_size, join=True)
        ddp_grads = torch.load(out)

    ref = single_process_grads()
    assert ddp_grads.shape == ref.shape

    # Tolerance is float32 reduction noise over ~1.8M elements, not slack for
    # a real disagreement: a missing all reduce shows up as O(1) relative error.
    torch.testing.assert_close(ddp_grads, ref, rtol=1e-4, atol=1e-6)


def test_a_broken_reduction_would_be_caught() -> None:
    """Guard on the guard: half the batch really does give a different gradient.

    Without this, a test that trivially passed would look like proof.
    """
    torch.manual_seed(SEED)
    model = ViT(PROBE)
    x, y = fixed_batch(BATCH_PER_RANK * 2)
    torch.nn.functional.cross_entropy(model(x[:BATCH_PER_RANK]), y[:BATCH_PER_RANK]).backward()
    half = flat_grads(model)

    ref = single_process_grads()
    assert not torch.allclose(half, ref, rtol=1e-4, atol=1e-6)
