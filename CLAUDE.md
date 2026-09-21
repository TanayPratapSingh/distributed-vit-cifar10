# Distributed ViT on CIFAR-10: working context

## What this is

A distributed training pipeline for CIFAR-10 image classification with a
vision transformer. The thesis: the valuable part of a project like this is
not that a ViT reached some accuracy, it is that the distributed systems
claims are checkable. A speedup number with no correctness proof behind it is
a number describing a system that might be training a different model than the
baseline.

## Current state

**Verified by execution on an Apple M5:** model, process group setup, both
launchers, data loading from the mirror, the full training and eval loop, AMP
on MPS, gradient accumulation, DDP over gloo, the sweep runner, the dashboard
generator, and the 18 test suite.

**Written but never executed:** everything CUDA. `nccl` backend selection,
FSDP, `torch.cuda.max_memory_allocated` reporting, fp16 plus GradScaler, and
the whole `kaggle` suite. There is no CUDA device on this machine. Expect real
bugs the first time `notebooks/kaggle_2xt4.ipynb` runs.

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

1. Finish the `laptop` suite and fill section 3 of `RESULTS.md`.
2. Run `notebooks/kaggle_2xt4.ipynb` on Kaggle at GPU T4 x2. This is the only
   step that produces a real multi device scaling curve. Score the three
   predictions in section 4.
3. Bring `runs/t4-*.json` home, rerun `python -m dvit.report`, and the
   dashboard shows both hardware groups side by side.
4. Only then consider a larger model. FSDP is expected to lose at 1.8M
   parameters, and confirming that is more interesting than avoiding it.

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
