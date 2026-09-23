"""Entrypoint. Runs unlaunched (world_size 1) or under torchrun.

    python -m dvit.train --epochs 20 --precision amp
    torchrun --nproc_per_node=2 -m dvit.train --epochs 20 --strategy ddp
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from .data import DataConfig, build_loaders, describe_source
from .dist import DistContext, UnsupportedTopology, setup, teardown
from .engine import (TrainConfig, build_optimizer, evaluate, pick_amp_dtype,
                     train_one_epoch, wrap_model)
from .metrics import (EpochRecord, RunRecord, describe_hardware, git_commit,
                      memory_source, peak_memory_mb, reset_peak_memory)
from .model import build_model


def seed_everything(seed: int, rank: int) -> None:
    # Offset by rank so augmentation and dropout differ per replica, while the
    # run stays reproducible.
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed ViT on CIFAR-10")
    p.add_argument("--name", default=None, help="run name; defaults to a descriptor")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128, help="per rank")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="ddp")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset-fraction", type=float, default=1.0,
                   help="fraction of CIFAR-10 to use; for fast smoke runs")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--synthetic-data", action="store_true",
                   help="in memory tensors, to isolate the input pipeline")
    p.add_argument("--device", default=None, help="cuda | mps | cpu; auto by default")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-scale", choices=["none", "linear"], default="linear",
                   help="linear multiplies lr by world_size (Goyal et al.)")
    return p.parse_args(argv)


def default_name(args, ctx: DistContext) -> str:
    bits = [f"ws{ctx.world_size}", ctx.device.type, args.strategy, args.precision]
    if args.grad_accum_steps > 1:
        bits.append(f"ga{args.grad_accum_steps}")
    if args.compile:
        bits.append("compiled")
    return "-".join(bits)


def reproduce_command(argv: list[str], world_size: int) -> str:
    """A portable command line that reruns this configuration.

    Recorded instead of sys.argv because sys.argv[0] is an absolute path on
    the machine that happened to run it, which is neither reproducible for
    anyone else nor anything a public artifact should carry.
    """
    args = " ".join(argv)
    if world_size > 1:
        return f"python -m dvit.launch --nproc {world_size} -- {args}"
    return f"python -m dvit.train {args}"


def main(argv: list[str] | None = None) -> int:
    # Captured before parsing so the artifact can record a command that
    # actually reproduces the run. sys.argv[0] is the module file path,
    # and under dvit.launch sys.argv belongs to the launcher, so neither
    # is usable directly.
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    try:
        ctx = setup(device_override=args.device)
    except UnsupportedTopology as e:
        print(f"refusing to run: {e}", file=sys.stderr)
        return 2

    seed_everything(args.seed, ctx.rank)

    lr = args.lr * (ctx.world_size if args.lr_scale == "linear" else 1)

    tcfg = TrainConfig(
        epochs=args.epochs, lr=lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, grad_accum_steps=args.grad_accum_steps,
        precision=args.precision, strategy=args.strategy, compile=args.compile,
        seed=args.seed,
    )
    dcfg = DataConfig(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=not args.no_augment,
        subset_fraction=args.subset_fraction,
        synthetic=args.synthetic_data,
    )

    train_loader, test_loader, sampler = build_loaders(dcfg, ctx)

    model = build_model().to(ctx.device)
    n_params = model.num_parameters()
    if args.compile:
        model = torch.compile(model)
    model = wrap_model(model, tcfg, ctx)

    optimizer = build_optimizer(model, tcfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.label_smoothing)

    amp_dtype, needs_scaler = pick_amp_dtype(ctx.device)
    scaler = None
    if tcfg.precision == "amp" and needs_scaler:
        if tcfg.strategy == "fsdp" and ctx.is_distributed:
            # FSDP shards gradients, so the ordinary GradScaler would test for
            # inf and nan against one shard and let each rank reach a different
            # conclusion about whether to skip the step. ShardedGradScaler
            # reduces that decision across ranks first.
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            scaler = ShardedGradScaler()
        else:
            scaler = torch.amp.GradScaler(ctx.device.type)

    name = args.name or default_name(args, ctx)
    record = RunRecord(
        name=name, world_size=ctx.world_size, device_type=ctx.device.type,
        backend=ctx.backend, precision=tcfg.precision, compiled=tcfg.compile,
        strategy=tcfg.strategy if ctx.is_distributed else "single",
        batch_size_per_rank=dcfg.batch_size,
        grad_accum_steps=tcfg.grad_accum_steps,
        effective_batch=dcfg.batch_size * ctx.world_size * tcfg.grad_accum_steps,
        epochs=tcfg.epochs, subset_fraction=dcfg.subset_fraction,
        model_params=n_params,
        synthetic_data=dcfg.synthetic,
        num_workers=dcfg.num_workers,
        amp_dtype=(str(amp_dtype).replace("torch.", "")
                   if tcfg.precision == "amp" else "fp32"),
        hardware=describe_hardware(ctx.device),
        data_source=describe_source(dcfg.root),
        memory_source=memory_source(ctx.device),
        threads_per_rank=torch.get_num_threads(),
        cpu_cores=os.cpu_count() or 0,
        torch_version=torch.__version__, git_commit=git_commit(),
        command=reproduce_command(effective_argv, ctx.world_size),
    )

    if ctx.is_main:
        print(f"[{name}] {record.hardware} | world_size={ctx.world_size} "
              f"backend={ctx.backend} | params={n_params:,} "
              f"| effective_batch={record.effective_batch} "
              f"| autocast={record.amp_dtype}")

    reset_peak_memory(ctx.device)
    steps_per_epoch = max(1, len(train_loader) // max(1, tcfg.grad_accum_steps))

    for epoch in range(tcfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)   # without this every epoch reuses one shuffle
        tr_loss, secs, ips = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, ctx, tcfg,
            epoch, steps_per_epoch,
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion, ctx, tcfg)
        cur_lr = optimizer.param_groups[0]["lr"]

        if ctx.is_main:
            record.epochs_log.append(EpochRecord(
                epoch=epoch, train_loss=tr_loss, test_loss=te_loss,
                test_acc=te_acc, seconds=secs, images_per_sec=ips, lr=cur_lr,
            ))
            print(f"  epoch {epoch:>3} | loss {tr_loss:.4f} | test {te_acc*100:5.2f}% "
                  f"| {secs:6.2f}s | {ips:8.1f} img/s")

    if ctx.is_main:
        record.peak_memory_mb = peak_memory_mb(ctx.device)
        record.finalise()
        path = record.write(args.out)
        print(f"[{name}] best {record.best_test_acc*100:.2f}% | "
              f"median {record.median_images_per_sec:.1f} img/s | "
              f"peak {record.peak_memory_mb:.0f} MB -> {path}")

    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
