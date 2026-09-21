"""Sampler behaviour, checked without downloading CIFAR-10."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class Counting(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def indices_for(rank: int, world_size: int, epoch: int, n: int = 1000) -> list[int]:
    s = DistributedSampler(Counting(n), num_replicas=world_size, rank=rank,
                           shuffle=True, drop_last=True)
    s.set_epoch(epoch)
    return list(s)


def test_ranks_see_disjoint_shards() -> None:
    a = set(indices_for(0, 2, epoch=0))
    b = set(indices_for(1, 2, epoch=0))
    assert a and b and a.isdisjoint(b)


def test_set_epoch_actually_reshuffles() -> None:
    # The bug this guards: forgetting set_epoch, so every epoch replays one order.
    assert indices_for(0, 2, epoch=0) != indices_for(0, 2, epoch=1)


def test_drop_last_keeps_step_counts_equal_across_ranks() -> None:
    # 1000 is not divisible by 3; without drop_last the ranks would disagree on
    # step count and the job would hang waiting for an all reduce that never comes.
    counts = {len(indices_for(r, 3, epoch=0)) for r in range(3)}
    assert len(counts) == 1
