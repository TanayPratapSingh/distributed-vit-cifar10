# Distributed ViT on CIFAR-10

A vision transformer trained across multiple devices, where the distributed
systems choices are measured rather than assumed.

The interesting claim in a project like this is not "I trained a ViT." It is
"I can tell you what two GPUs actually bought me, and I can prove the two GPU
run computes the same gradient as the one GPU run." This repository is built
around making both of those statements checkable.

## What is here

| Piece | File | What it does |
|---|---|---|
| Model | `src/dvit/model.py` | ViT written out, 1,806,538 parameters, patch 4 on 32x32 |
| Process groups | `src/dvit/dist.py` | Backend selection, and refusal when a topology is impossible |
| Data | `src/dvit/data.py` | DistributedSampler, canonical or mirror source selection |
| Training | `src/dvit/engine.py` | AMP, gradient accumulation, DDP and FSDP, no_sync |
| Entrypoint | `src/dvit/train.py` | Runs unlaunched or under torchrun |
| Sweeps | `src/dvit/sweep.py` | Runs a named suite from `configs/experiments.yaml` |
| Dashboard | `src/dvit/report.py` | Builds `dashboard/index.html` from run artifacts only |

## The three things worth asking about in an interview

**1. The gradient equivalence test.** DDP all reduces gradients with a mean, so
two ranks each holding batch B must produce the same gradient as one process
holding batch 2B. `tests/test_ddp_equivalence.py` spawns a real gloo process
group and checks that identity to `rtol=1e-4`. It ships with a second test that
deliberately computes the wrong thing, to show the first test would actually
catch a broken reduction rather than passing trivially.

**2. Refusing the impossible topology.** Apple silicon has one GPU and no
collective backend: gloo cannot all reduce an MPS tensor and nccl is CUDA only.
The easy move is to quietly copy tensors to CPU and keep calling it distributed
MPS training. `resolve_backend` raises `UnsupportedTopology` instead, with a
message that names the three paths that do work. A benchmark that cannot be
trusted to refuse is a benchmark that cannot be trusted.

**3. Gradient accumulation inside `no_sync()`.** Accumulation is supposed to buy
effective batch size without buying memory. Under DDP, if you do not wrap the
non final microbatches in `no_sync()`, you pay a full all reduce per microbatch
and accumulation makes the job slower instead of cheaper. See `engine.py`.

## Hardware reality

This was developed on an Apple M5 with one GPU, so the honest split is:

| Where | Topology | What it can prove |
|---|---|---|
| M5 laptop | MPS, 1 device | Single device accuracy and throughput |
| M5 laptop | gloo, N CPU processes | Gradient correctness, and CPU weak scaling across 10 cores |
| Kaggle | nccl, 2x T4 | Real speedup and scaling efficiency |

The `gloo-overhead` suite was built expecting to measure a slowdown: more
processes on one chip add communication without adding silicon. The
measurement disagreed. On a 10 core M5, going from 1 to 4 gloo ranks raised
throughput from 552 to 879 images per second, because the extra ranks recruit
cores that a single rank leaves idle. The suite kept its name and the
prediction was deleted. See `RESULTS.md` for the numbers and the thread
settings they depend on.

T4 is sm_75, so `--precision amp` resolves to fp16 plus a GradScaler there, not
bf16. `pick_amp_dtype` decides from the card rather than hardcoding, because
guessing wrong is a silent slowdown.

On MPS the same reflex backfires outright. fp16 autocast alone measured 726
img/s against 1,061 for fp32, a 32 percent slowdown, because eager mode pays
for a cast around every operator and Apple silicon is already efficient at
fp32. The same fp16 config under `torch.compile` reaches 2,361 img/s, 2.23x
the baseline, once those casts are fused into the graph. "Turn on AMP" is CUDA
advice, and this project has the numbers showing where it does not transfer.

## Quick start

```bash
make install
python scripts/fetch_cifar10.py --root data
make test
make smoke
```

`make smoke` runs every code path on 2 percent of the data for 2 epochs. Those
runs are badged as smoke in the dashboard and their accuracy means nothing.

Real numbers:

```bash
make laptop      # single device MPS, full dataset
make gloo        # communication overhead, measured not assumed
```

On Kaggle with the accelerator set to GPU T4 x2, paste `scripts/kaggle_bootstrap.sh`
into a cell, or:

```bash
python -m dvit.sweep kaggle --continue-on-error
python -m dvit.report
```

## Running one configuration by hand

```bash
python -m dvit.train --epochs 30 --precision amp --device mps --strategy single
```

```bash
torchrun --standalone --nproc_per_node=2 -m dvit.train --epochs 30 --precision amp --strategy ddp
```

## A note on the dataset

`www.cs.toronto.edu` is unreachable from some networks, including the sandbox
this was built in, where it fails with an SSL EOF before any payload arrives.
`scripts/fetch_cifar10.py` tries the canonical tarball first and falls back to
the HuggingFace parquet mirror.

The fallback deliberately does not rebuild the original `cifar-10-batches-py`
pickles. Those carry published md5 sums a rebuilt file cannot match, so the only
way to make that path appear to work would be to disable torchvision integrity
checking. Instead the mirror writes its own `npz` format, `dvit.data` picks
whichever source is present, and every run artifact records which one it used.

## Where the numbers live

`RESULTS.md` holds the tables. `dashboard/index.html` is generated by
`python -m dvit.report` and reads `runs/*.json` exclusively. No number in either
place is typed by hand, and speedup is only ever computed against a baseline
measured on the same hardware.
