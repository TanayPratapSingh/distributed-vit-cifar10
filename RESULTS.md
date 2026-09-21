# Results

Every row here traces to a run artifact in `runs/` and the command that
produced it. `dashboard/index.html` is generated from the same artifacts by
`python -m dvit.report`, so the two cannot disagree.

Nothing on this page is copied from a paper or estimated.

## Provenance

| Field | Value |
|---|---|
| Model | ViT, 1,806,538 parameters, patch 4, dim 192, depth 6, heads 3 |
| Dataset | CIFAR-10, 50,000 train and 10,000 test |
| Data source | HuggingFace parquet mirror, rebuilt. The canonical host was unreachable |
| Laptop | Apple M5, 10 CPU cores, 8 GPU cores, 16 GB unified memory |
| Accelerator | 2x NVIDIA T4, Kaggle. **Not yet run** |

The data source matters enough to state plainly: `www.cs.toronto.edu` refused
the connection with an SSL EOF, so the pixels came from the HuggingFace mirror
of the same dataset. The mirror is not the original tarball and carries none of
its md5 sums, which is why `dvit.data` keeps the two sources separate and every
artifact records which one it used.

## 1. Gradient correctness

Not a performance number, but the one that licenses every other number here.

| Check | Result |
|---|---|
| 2 rank gloo DDP gradient vs 1 process over the same effective batch | match to `rtol=1e-4` |
| Same comparison against a deliberately halved batch | differs, as it must |

```bash
python -m pytest tests/test_ddp_equivalence.py
```

The second row is the point. A correctness test that cannot fail is decoration.

## 2. CPU weak scaling with gloo

3 epochs, 10 percent of CIFAR-10, batch 64 per rank, `dvit.launch` FileStore
rendezvous.

| Run | Ranks | Threads per rank | Effective batch | img/s | Speedup | Efficiency |
|---|---|---|---|---|---|---|
| `gloo-ws1` | 1 | 4 | 64 | 552 | baseline | |
| `gloo-ws2` | 2 | 4 | 128 | 694 | 1.26x | 63% |
| `gloo-ws4` | 4 | 4 | 256 | 879 | 1.59x | 40% |

```bash
python -m dvit.sweep gloo-overhead
```

**This suite was built expecting the opposite result.** The reasoning was that
extra processes on one chip add communication without adding silicon, so
throughput should fall. It rose.

The recorded thread counts explain why, and they are the reason the column is
in the table. PyTorch gave every rank 4 intra op threads on a 10 core machine.
The single rank baseline therefore left 6 cores idle, and the extra ranks were
not competing for a saturated CPU, they were recruiting cores nobody was using.
By 4 ranks the machine is oversubscribed at 16 threads on 10 cores, which is
where efficiency collapses to 40 percent.

Three honest caveats:

* This is weak scaling. Effective batch grows with rank count, so the runs are
  not doing identical optimization work and the accuracy column is not
  comparable across rows.
* A fair strong scaling test would pin total threads constant across
  configurations. That is not what these numbers are, and the table says so.
* These were measured on an otherwise idle machine. An earlier rerun that
  overlapped the MPS suite produced 146, 288 and 343 img/s, roughly a third of
  these figures, and even reversed the efficiency trend. Those numbers were
  discarded rather than averaged in. Any throughput measured against
  unquantified background load is not a measurement.

## 3. Single device Apple silicon

30 epochs, full CIFAR-10, batch 128.

| Run | Precision | img/s | Speedup | Best top-1 | Wall clock |
|---|---|---|---|---|---|
| `mps-fp32` | fp32 | 1,061 | baseline | 80.61% | 24.3 min |
| `mps-amp` | fp16 autocast | 726 | 0.68x | 80.77% | 31.6 min |
| `mps-amp-compiled` | fp16 plus torch.compile | 2,361 | 2.23x | 80.72% | 10.6 min |

```bash
make laptop
```

A 1.8M parameter ViT trained from scratch reaches **80.61% top-1** in 30
epochs, no pretraining, on a laptop GPU in 24 minutes. Accuracy was still
climbing at the last epoch, so this is a schedule length result, not a ceiling.

**The surprise is the middle row.** fp16 autocast on its own is a 32 percent
slowdown, 726 against 1,061 img/s. Adding `torch.compile` to the same fp16
config then gives 2.23x over the fp32 baseline and cuts wall clock from
24 minutes to 11.

The reading: on MPS, eager mode autocast pays for a cast around every operator
and Apple silicon is already efficient at fp32, so the casts cost more than the
fp16 arithmetic saves. Compilation fuses those casts into the graph and the
saving finally lands. The practical lesson is that "turn on AMP" is not
portable advice. It is a CUDA reflex, and on this backend applying it alone
made training slower.

The accuracy column is the control. All three land within 0.16 points of each
other (80.61, 80.77, 80.72), which is what should happen: precision and
compilation are throughput changes, not learning changes. If one of them had
moved accuracy materially, that would have been a bug report, not a result.

## 4. Real multi device scaling, 2x T4

**Not yet measured.** This is the table the project exists for, and it is empty
on purpose rather than filled with a plausible guess.

Run `notebooks/kaggle_2xt4.ipynb` on Kaggle with the accelerator set to
GPU T4 x2, download `runs-t4.zip`, unzip into `runs/`, and rerun
`python -m dvit.report`.

| Run | Ranks | Strategy | Precision | img/s | Speedup | Efficiency | Peak MB | Best top-1 |
|---|---|---|---|---|---|---|---|---|
| `t4-ws1-fp32` | 1 | single | fp32 | | baseline | | | |
| `t4-ws1-amp` | 1 | single | fp16 | | | | | |
| `t4-ws2-ddp-fp32` | 2 | DDP | fp32 | | | | | |
| `t4-ws2-ddp-amp` | 2 | DDP | fp16 | | | | | |
| `t4-ws2-ddp-amp-compiled` | 2 | DDP | fp16 + compile | | | | | |
| `t4-ws2-fsdp-amp` | 2 | FSDP | fp16 | | | | | |
| `t4-ws2-ddp-amp-ga2` | 2 | DDP | fp16, accum 2 | | | | | |

Predictions recorded before running, so they can be scored rather than
retrofitted:

1. DDP at 2 ranks lands near 1.8x, not 2x. The model is small and the
   all reduce is not free.
2. FSDP loses to DDP. At 1.8M parameters there is nearly nothing to shard and
   the extra collectives are pure overhead.
3. amp is a large win on T4. fp16 with a GradScaler, not bf16: T4 is sm_75.

## 5. Bugs found while building this

The most useful artifact of the week, kept deliberately.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | CIFAR-10 download died with `SSL: UNEXPECTED_EOF_WHILE_READING` | Canonical host unreachable from this network | Mirror fallback in `scripts/fetch_cifar10.py` |
| 2 | Rebuilt pickles rejected as "Dataset not found or corrupted" | torchvision checks published md5 sums a rebuilt file cannot match | Stopped impersonating the original layout. Mirror writes its own `npz` and the loader picks a source |
| 3 | Every 2 rank run hung, no error, until timeout | torchrun's default TCPStore rendezvous on localhost is blocked here. IPv6 resolution failed in a loop | `dvit.launch`, a FileStore launcher needing no sockets. `dvit.dist` honours `DVIT_INIT_FILE` |
| 4 | MPS runs reported a peak memory figure | `torch.mps.current_allocated_memory` is instantaneous, read after activations were freed. There is no MPS high water mark | Report memory only where an allocator peak exists, and record `memory_source` |
| 5 | Parameter count in two docstrings said 2.7M | Written from an estimate before the model was built | Corrected to the measured 1,806,538 |
| 6 | FSDP runs would have under clipped gradients | `torch.nn.utils.clip_grad_norm_` sees only the local shard, so each rank clips against a norm computed from a fraction of the gradient | Use FSDP's own collective aware `model.clip_grad_norm_`. Found by inspection, not by execution, because there is no CUDA device here |
| 7 | The gloo suite reported 146, 288 and 343 img/s on a rerun, against 521, 632 and 814 the first time | The rerun overlapped the MPS suite. Both were competing for the same 10 cores | Benchmarks are run on an idle machine. Numbers taken under concurrent load are discarded, not reconciled |

Bug 3 is the one worth telling in an interview. The failure mode was not a
crash, it was a hang, which reads exactly like slow training. Anything that
makes a distributed job look slow rather than broken will cost hours before
anyone suspects the launcher.
