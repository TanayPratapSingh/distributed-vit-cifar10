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

1. The scaling story has a gap worth closing: amp at 2 ranks reaches only 57
   percent efficiency against fp32's 98 percent. Profile it rather than guess.
   The candidates are the nccl all reduce and the 2 dataloader workers per
   rank. Raising `--num-workers` and re measuring is the cheapest first test.
2. If input bound, move augmentation to the GPU and re measure. If
   communication bound, try gradient bucketing size and compression.
3. A larger model would make FSDP a fair fight. At 1.8M parameters its 0.66x
   is a foregone conclusion rather than a finding.
4. More than 2 ranks needs paid hardware. Everything in the code path already
   supports it.

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
