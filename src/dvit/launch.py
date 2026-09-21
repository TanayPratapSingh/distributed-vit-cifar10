"""Launch N training processes over a FileStore rendezvous.

Use this instead of torchrun where localhost TCP is unavailable. torchrun's
default rendezvous opens a TCPStore on 127.0.0.1; when a sandbox blocks that,
the failure mode is a hang rather than an error, which is worse than a crash
because it looks like slow training.

    python -m dvit.launch --nproc 2 -- --epochs 2 --device cpu --strategy ddp

Everything after `--` is passed through to dvit.train unchanged, so the two
launchers accept identical training arguments.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch.multiprocessing as mp


def _child(local_rank: int, world_size: int, init_file: str, argv: list[str]) -> None:
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        DVIT_INIT_FILE=init_file,
    )
    from .train import main as train_main

    rc = train_main(argv)
    if rc != 0:
        raise SystemExit(rc)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nproc", type=int, required=True)
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    passthrough = args.train_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.nproc == 1:
        from .train import main as train_main
        return train_main(passthrough)

    # FileStore requires a path that does not exist yet.
    tmp = Path(tempfile.gettempdir()) / f"dvit-pg-{os.getpid()}"
    if tmp.exists():
        tmp.unlink()

    try:
        mp.spawn(_child, args=(args.nproc, str(tmp), passthrough),
                 nprocs=args.nproc, join=True)
    except Exception as e:
        print(f"launch failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if tmp.exists():
            tmp.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
