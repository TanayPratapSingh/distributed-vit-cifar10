"""Run a named suite from configs/experiments.yaml.

Each run is a separate process so that a crash in one configuration cannot
corrupt the others, and so that torchrun owns the process group lifecycle
rather than this script.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

BOOL_FLAGS = {"compile", "no-augment", "synthetic-data"}


def default_launcher() -> str:
    """torchrun where its TCP rendezvous works, FileStore where it does not.

    torchrun is the right answer on a cluster and on Kaggle. On macOS, and in
    sandboxes that block localhost TCP, its TCPStore rendezvous hangs instead
    of failing, so the default there is the FileStore launcher. Override with
    DVIT_LAUNCHER=torchrun|file.
    """
    override = os.environ.get("DVIT_LAUNCHER")
    if override in {"torchrun", "file"}:
        return override
    return "file" if platform.system() == "Darwin" else "torchrun"


def build_command(run: dict, defaults: dict, out: str) -> list[str]:
    merged = {**defaults, **run}
    nproc = int(merged.pop("nproc", 1))
    name = merged.pop("name")

    launcher = merged.pop("launcher", None) or default_launcher()

    if nproc > 1 and launcher == "torchrun":
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--standalone", f"--nproc_per_node={nproc}", "-m", "dvit.train"]
    elif nproc > 1:
        cmd = [sys.executable, "-m", "dvit.launch", "--nproc", str(nproc), "--"]
    else:
        cmd = [sys.executable, "-m", "dvit.train"]

    cmd += ["--name", str(name), "--out", out]
    for key, value in merged.items():
        flag = f"--{key}"
        if key in BOOL_FLAGS:
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run an experiment suite")
    p.add_argument("suite")
    p.add_argument("--config", default="configs/experiments.yaml")
    p.add_argument("--out", default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args(argv)

    spec = yaml.safe_load(Path(args.config).read_text())
    if args.suite not in spec:
        print(f"unknown suite {args.suite!r}; have: {', '.join(spec)}", file=sys.stderr)
        return 2

    suite = spec[args.suite]
    defaults = suite.get("defaults", {})
    runs = suite["runs"]
    width = shutil.get_terminal_size((80, 20)).columns

    failures = []
    for i, run in enumerate(runs, 1):
        cmd = build_command(dict(run), dict(defaults), args.out)
        print("=" * min(width, 78))
        print(f"[{i}/{len(runs)}] {run['name']}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        dt = time.perf_counter() - t0
        if rc != 0:
            failures.append((run["name"], rc))
            print(f"  FAILED rc={rc} after {dt:.1f}s")
            if not args.continue_on_error:
                break
        else:
            print(f"  ok in {dt:.1f}s")

    if failures:
        print("\nfailed runs:")
        for name, rc in failures:
            print(f"  {name} (rc={rc})")
        return 1
    print(f"\nsuite {args.suite!r} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
