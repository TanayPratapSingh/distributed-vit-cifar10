"""Training and evaluation loop, plus the systems knobs being compared.

The knobs, and why each one is here:

  precision       fp32 / amp. On a T4 (sm_75) bf16 is not supported, so amp
                  means fp16 plus a GradScaler. On sm_80 and later it means
                  bf16 and no scaler. Selecting this wrongly is a classic
                  silent slowdown, so `pick_amp_dtype` decides from the card.
  strategy        ddp / fsdp / single. FSDP shards parameters, gradients and
                  optimizer state. At 1.8M parameters it is not expected to
                  win. Measuring that is the point.
  grad_accum      Raises effective batch without raising memory. Under DDP the
                  non final microbatches run inside `no_sync()`, otherwise you
                  pay a full all reduce per microbatch and accumulation makes
                  the job slower instead of cheaper.
  compile         torch.compile. First step pays a compile cost, so the
                  throughput number excludes a warmup epoch.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .dist import DistContext, all_reduce_mean, all_reduce_sum
from .metrics import EpochRecord, Stopwatch, synchronize


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    grad_accum_steps: int = 1
    precision: str = "fp32"      # fp32 | amp
    strategy: str = "ddp"        # single | ddp | fsdp
    compile: bool = False
    max_grad_norm: float = 1.0
    seed: int = 0


def amp_dtype_for_capability(major: int) -> tuple[torch.dtype, bool]:
    """Pick the autocast dtype from a CUDA compute capability major version.

    Split out from device inspection so it can be tested without a GPU.

    Hardware bf16 arrives with Ampere, sm_80. Do NOT use
    `torch.cuda.is_bf16_supported()` for this: on a T4 (sm_75) it returns
    True, because recent PyTorch counts emulated bf16 as supported. Taking
    that at face value selects a software emulated path and quietly measures
    a slowdown while reporting it as mixed precision.
    """
    if major >= 8:
        return torch.bfloat16, False
    return torch.float16, True          # Turing, Volta: fp16 plus a scaler


def pick_amp_dtype(device: torch.device) -> tuple[torch.dtype, bool]:
    """Return (dtype, needs_grad_scaler) for autocast on this device."""
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return amp_dtype_for_capability(major)
    if device.type == "mps":
        return torch.float16, False     # no GradScaler support on mps
    return torch.bfloat16, False        # cpu autocast


def wrap_model(model: nn.Module, cfg: TrainConfig, ctx: DistContext) -> nn.Module:
    if not ctx.is_distributed or cfg.strategy == "single":
        return model
    if cfg.strategy == "ddp":
        device_ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
        return DDP(model, device_ids=device_ids, gradient_as_bucket_view=True)
    if cfg.strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        policy = functools.partial(size_based_auto_wrap_policy, min_num_params=10_000)
        return FSDP(model, auto_wrap_policy=policy, device_id=ctx.local_rank
                    if ctx.device.type == "cuda" else None)
    raise ValueError(f"unknown strategy {cfg.strategy!r}")


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """No weight decay on norms and biases. Standard for ViT, cheap to get right."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith((".cls_token", ".pos_embed")):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )


def lr_at(step: int, steps_per_epoch: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay, computed per step."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = cfg.warmup_epochs * steps_per_epoch
    if step < warm and warm > 0:
        return cfg.lr * step / warm
    progress = (step - warm) / max(1, total - warm)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_one_epoch(model, loader, optimizer, scaler, criterion, ctx, cfg,
                    epoch: int, steps_per_epoch: int) -> tuple[float, float, float]:
    """Returns (mean loss, seconds, images per second) averaged over ranks."""
    model.train()
    amp_dtype, _ = pick_amp_dtype(ctx.device)
    use_amp = cfg.precision == "amp"
    accum = max(1, cfg.grad_accum_steps)

    total_loss, n_batches, n_images = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    with Stopwatch(ctx.device) as sw:
        for i, (x, y) in enumerate(loader):
            x = x.to(ctx.device, non_blocking=True)
            y = y.to(ctx.device, non_blocking=True)

            step = epoch * steps_per_epoch + (i // accum)
            lr = lr_at(step, steps_per_epoch, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            is_last_micro = ((i + 1) % accum == 0) or (i + 1 == len(loader))
            # Skip the all reduce on every microbatch but the last.
            sync_ctx = (
                model.no_sync()
                if (not is_last_micro and hasattr(model, "no_sync"))
                else contextlib.nullcontext()
            )

            with sync_ctx:
                with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    loss = criterion(model(x), y) / accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_last_micro:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                # FSDP shards gradients, so the free function would clip each
                # shard against the full norm and under clip. FSDP exposes its
                # own collective aware version and it must be used instead.
                if hasattr(model, "clip_grad_norm_"):
                    model.clip_grad_norm_(cfg.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum
            n_batches += 1
            n_images += x.shape[0]

    seconds = sw.elapsed
    # Throughput is a global property: sum images across ranks, use slowest time.
    global_images = all_reduce_sum(float(n_images), ctx)
    global_seconds = max(seconds, all_reduce_mean(seconds, ctx))
    mean_loss = all_reduce_mean(total_loss / max(1, n_batches), ctx)
    return mean_loss, seconds, global_images / max(1e-9, global_seconds)


@torch.no_grad()
def evaluate(model, loader, criterion, ctx, cfg) -> tuple[float, float]:
    model.eval()
    amp_dtype, _ = pick_amp_dtype(ctx.device)
    use_amp = cfg.precision == "amp"
    loss_sum, correct, total = 0.0, 0.0, 0.0

    for x, y in loader:
        x = x.to(ctx.device, non_blocking=True)
        y = y.to(ctx.device, non_blocking=True)
        with torch.autocast(device_type=ctx.device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.item() * x.shape[0]
        correct += (logits.argmax(1) == y).sum().item()
        total += x.shape[0]

    synchronize(ctx.device)
    # Sum then divide, rather than averaging per rank averages, because the
    # distributed test sampler can hand ranks different counts.
    g_loss = all_reduce_sum(loss_sum, ctx)
    g_correct = all_reduce_sum(correct, ctx)
    g_total = all_reduce_sum(total, ctx)
    return g_loss / max(1.0, g_total), g_correct / max(1.0, g_total)
