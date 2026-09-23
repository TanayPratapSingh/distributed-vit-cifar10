"""The synthetic input path is an instrument, so it needs its own checks."""

from __future__ import annotations

import torch

from dvit.data import SyntheticCIFAR


def test_shapes_match_real_cifar() -> None:
    ds = SyntheticCIFAR(1000)
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert isinstance(y, int) and 0 <= y < 10
    assert len(ds) == 1000


def test_pool_is_reused_rather_than_materialised() -> None:
    # 50,000 float32 images would be about 600 MB. The pool keeps it at ~6 MB,
    # which is the only reason this can run alongside a training job.
    ds = SyntheticCIFAR(50_000)
    assert ds.images.shape[0] == SyntheticCIFAR.POOL
    assert torch.equal(ds[0][0], ds[SyntheticCIFAR.POOL][0])


def test_two_instances_with_the_same_seed_agree() -> None:
    # Ranks build their own copy, so they must produce identical tensors or
    # the replicas would be training on different data.
    a, b = SyntheticCIFAR(100, seed=7), SyntheticCIFAR(100, seed=7)
    assert torch.equal(a.images, b.images)
    assert torch.equal(a.labels, b.labels)
