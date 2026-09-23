# Distributed ViT on CIFAR-10: working context

## What this is

A distributed training pipeline for CIFAR-10 image classification with a
vision transformer. The thesis: the valuable part of a project like this is
not that a ViT reached some accuracy, it is that the distributed systems
claims are checkable. A speedup number with no correctness proof behind it is
a number describing a system that might be training a different model than the
baseline.

## Current state

**Verified by execution.** Everything. The model, both launchers, data
loading, the training and eval loop, AMP on both MPS and CUDA, gradient
accumulation, DDP over gloo and over nccl, FSDP, the sweep runner, the
dashboard generator, and the 21 test suite. Three hardware groups have real
measurements: Apple M5, CPU gloo, and 2x Tesla T4 on Kaggle.

**Not yet attempted.** More than 2 ranks, multi node, a model large enough for
FSDP to make sense, and cross encoder scale attention. The 2x T4 ceiling is a
property of free Kaggle, not of the code.

## House rules

1. **Never report a number the code did not measure.** Every cell in
   `RESULTS.md` traces to a file in `runs/`. The dashboard reads those
   artifacts and nothing else. If a table cell is unknown, it stays empty and
   says "not yet measured".
2. **Never weaken a check to make something pass.** The md5 failure in bug 2
   was fixable in one line by disabling torchvision integrity checking. That
   would have been the wrong line.
3. **Speedup is only ever computed within one hardware group.** Comparing a T4
   against an M5 and calling the ratio a speedup is meaningless.
4. **Predictions get written down before the run, and scored after.** See
   section 4 of `RESULTS.md`. A prediction that was wrong is kept and labelled,
   as with the gloo suite in section 2.
5. **No em dashes.** No hyphens in compound phrases like "real time".
6. **Keep the bug list.** Section 5 of `RESULTS.md` is the most valuable
   output of the build.

## Next steps, in order

The 57 percent question from section 4 is closed. Section 5 isolated it with
`--synthetic-data`: the collective scales at 96 percent, and the ceiling is a
4 vCPU host that cannot feed two T4s. Remaining work follows from that.

1. Confirm the diagnosis by removing the ceiling rather than inferring it.
   Either GPU side augmentation, or the same suite on a host with more cores.
   Neither is possible on free Kaggle, so this needs paid hardware.
2. A larger model would make FSDP a fair fight. At 1.8M parameters its 0.66x
   is arithmetic, not a finding.
3. More than 2 ranks. Every code path already supports it.

Do not tune nccl. Section 5 rules it out, and the measurement is in `runs/`.

## Commands

```bash
make install                              # venv plus editable install
python scripts/fetch_cifar10.py --root data
make test                                 # 18 tests, no network, no GPU
make smoke                                # every code path, 2 percent of data
make laptop                               # single device MPS, full dataset
make gloo                                 # CPU weak scaling
make report                               # regenerate dashboard/index.html
```
