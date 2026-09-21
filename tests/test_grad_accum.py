"""Gradient accumulation must not change the gradient.

Accumulating K microbatches of size B, with each loss divided by K, has to
produce the same gradient as one step over a batch of size K*B. If it does
not, then every "effective batch" claim in RESULTS.md is describing an
optimization run that never happened.

This is the single process version of the invariant. The distributed version,
where the non final microbatches must run inside `no_sync()`, is covered by
tests/test_ddp_equivalence.py plus this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dvit.model import ViT, ViTConfig

PROBE = ViTConfig(dim=48, depth=2, heads=2, dropout=0.0, drop_path=0.0)
SEED = 99


def batch(n: int):
    g = torch.Generator().manual_seed(SEED)
    return (torch.randn(n, 3, 32, 32, generator=g),
            torch.randint(0, 10, (n,), generator=g))


def flat(model):
    return torch.cat([p.grad.reshape(-1) for _, p in sorted(model.named_parameters())
                      if p.grad is not None])


def full_batch_grads(n: int):
    torch.manual_seed(SEED)
    m = ViT(PROBE)
    x, y = batch(n)
    F.cross_entropy(m(x), y).backward()
    return flat(m)


def accumulated_grads(n: int, micro: int):
    torch.manual_seed(SEED)
    m = ViT(PROBE)
    x, y = batch(n)
    steps = n // micro
    for i in range(steps):
        lo, hi = i * micro, (i + 1) * micro
        (F.cross_entropy(m(x[lo:hi]), y[lo:hi]) / steps).backward()
    return flat(m)


def test_accumulation_matches_the_full_batch() -> None:
    torch.testing.assert_close(accumulated_grads(16, 4), full_batch_grads(16),
                               rtol=1e-4, atol=1e-6)


def test_forgetting_to_divide_by_steps_is_caught() -> None:
    """The classic bug: accumulate without scaling, gradients come out K times
    too large. Confirms the test above is actually sensitive to it."""
    torch.manual_seed(SEED)
    m = ViT(PROBE)
    x, y = batch(16)
    for i in range(4):
        F.cross_entropy(m(x[i * 4:(i + 1) * 4]), y[i * 4:(i + 1) * 4]).backward()
    unscaled = flat(m)
    torch.testing.assert_close(unscaled, full_batch_grads(16) * 4, rtol=1e-3, atol=1e-5)
