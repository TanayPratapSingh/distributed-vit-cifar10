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

30 epochs, full CIFAR-10, batch 128 per rank, nccl, torch 2.10.0+cu128,
all seven from commit `33bcd11`. Every amp run recorded `amp_dtype: float16`,
which the artifacts carry so it can be checked rather than assumed.

| Run | Ranks | Strategy | Precision | img/s | Rank speedup | vs fp32 | Peak MB | Top-1 |
|---|---|---|---|---|---|---|---|---|
| `t4-ws1-fp32` | 1 | single | fp32 | 1,853 | baseline | reference | 553 | 80.70% |
| `t4-ws1-amp` | 1 | single | fp16 | 3,810 | baseline | 2.06x | 347 | 80.52% |
| `t4-ws2-ddp-fp32` | 2 | DDP | fp32 | 3,637 | 1.96x (98%) | 1.96x | 561 | 80.01% |
| `t4-ws2-ddp-amp` | 2 | DDP | fp16 | 4,376 | 1.15x (57%) | 2.36x | 354 | 79.71% |
| `t4-ws2-ddp-amp-compiled` | 2 | DDP | fp16 + compile | 4,706 | no baseline | 2.54x | 345 | 80.09% |
| `t4-ws2-ddp-amp-ga2` | 2 | DDP | fp16, accum 2 | 4,393 | no baseline | 2.37x | 361 | 80.15% |
| `t4-ws2-fsdp-amp` | 2 | FSDP | fp16 | 2,884 | 0.76x (38%) | 1.56x | 338 | 79.41% |

```bash
python -m dvit.sweep kaggle --continue-on-error
```

Rank speedup holds every other setting fixed and varies only rank count. It
reads "no baseline" where no matching single rank run exists, rather than
being filled with a ratio against a different configuration.

One row needs a caveat the table cannot carry. FSDP's 0.76x compares two
ranks under FSDP against one rank with no wrapper at all, so it mixes a
strategy change into a rank change. Read literally it says something true and
blunt: two T4s running FSDP are slower than one T4 running unwrapped. The
like for like comparison is FSDP against DDP at the same rank count, which is
0.66x and is scored under prediction 2.

### The predictions, scored

**1. "DDP at 2 ranks lands near 1.8x, not 2x." Wrong, and wrong in a way that
turned out to be the most interesting result here.**

In fp32, DDP scales at **1.96x, 98 percent efficiency**. My reasoning was
backwards: I argued a small model makes the all reduce costly, when a small
model means small gradient tensors, about 7 MB, which DDP overlaps with the
backward pass almost entirely.

In amp, the same DDP configuration scales at only **1.15x, 57 percent
efficiency**. Identical hardware, identical model, identical communication
volume. The only change is that each GPU now computes twice as fast.

That is the finding worth keeping. Speeding up compute did not speed up the
job proportionally, it moved the bottleneck. At 1,853 img/s per GPU the
all reduce and the 2 dataloader workers per rank are comfortably hidden. At
3,810 img/s they are not. Scaling efficiency is not a property of a
cluster, it is a property of a cluster running a particular configuration,
and optimising one layer can degrade it.

**2. "FSDP loses to DDP at this model size." Correct, and by more than expected.**

FSDP reaches 2,884 img/s against DDP's 4,376, so **0.66x**. It buys
peak memory of 338 MB against 354 MB, a 5 percent saving for a
34 percent throughput loss. At 1.8M parameters there is almost nothing
to shard and the extra collectives are close to pure overhead.

"We chose DDP because FSDP measured 0.66x at this model size" is a better
answer than "we used DDP".

**3. "amp is a large win on T4, fp16 with a GradScaler." Correct.**

2.06x at one rank, 1,853 to 3,810 img/s, with peak memory falling from
553 MB to 347 MB. This only holds because the dtype fix landed first.
Before it, every amp run requested bf16 and torch inductor replied "Tesla T4
does not support bfloat16 compilation natively, skipping". See bug 8.

Worth contrasting with section 3: the identical `--precision amp` flag is a
2.06x win on T4 and a 0.68x regression on Apple MPS. Precision advice does
not survive a change of backend.

### Smaller results

* `torch.compile` adds 1.08x on top of DDP plus amp, reaching 4,706 img/s and
  **2.54x over single GPU fp32** overall. Far less dramatic than the 2.23x it
  gave on MPS, because CUDA eager kernels are already well optimised and there
  was less waste to fuse away.
* Gradient accumulation at 2 microbatches is 1.004x, effectively free. That is
  the `no_sync()` path doing its job: without it, accumulation would pay an
  extra all reduce per microbatch and land well under 1.0x.
* Top-1 spans 79.41% to 80.70% across all seven. These are systems changes,
  and accuracy behaves like it.

## 5. Why amp scaled at 57 percent: an isolation experiment

Section 4 left a number unexplained. On the same two T4s, DDP scaled at 98
percent in fp32 and 57 percent in amp. Two suspects: the nccl all reduce, and
the input pipeline.

Raising `--num-workers` and watching what happens would have produced a
correlation. Instead `--synthetic-data` serves pre normalised tensors from a
small in memory pool, removing decode, augmentation and worker processes
entirely. Two ranks against one rank under it is compute plus communication
with nothing else in the measurement.

6 epochs, batch 128 per rank, fp16, 2x T4, all from commit `2e788b1`.

| Input | 1 rank | 2 ranks | Speedup | Efficiency |
|---|---|---|---|---|
| **Synthetic, no dataloader** | 3,956 | 7,580 | **1.92x** | **96%** |
| Real data, 2 workers per rank | 3,462 | 4,073 | 1.18x | 59% |
| Real data, 8 workers per rank | 3,599 | 3,996 | 1.11x | 56% |

```bash
python -m dvit.sweep kaggle-bottleneck
```

### The verdict

**The collective is innocent.** With no dataloader, DDP plus amp scales at
1.92x, 96 percent efficiency. The all reduce moves about 7 MB and DDP overlaps
nearly all of it with the backward pass, exactly as it did in fp32.

**The host is the ceiling.** With real data, throughput never passes about
4,073 img/s no matter how many GPUs or workers are thrown at it. The
artifacts record `cpu_cores: 4`. Kaggle gives 4 vCPUs, and 4 cores cannot
decode, crop, flip and normalise CIFAR-10 faster than roughly that rate.

One T4 in fp16 already consumes 3,956 img/s of synthetic input, which is
within 3 percent of everything the host can produce. So the second
GPU is not slow, it is unfed.

**More workers made it worse.** 3,996 img/s at 8 workers per rank against
4,073 at 2. With 4 cores, 16 worker processes buy context switches and
nothing else. PyTorch warned about this and the measurement agrees with the
warning.

**The input pipeline was already costing something at one rank.** Real data
at 1 rank reaches 3,462 img/s against 3,956 synthetic, so
12 percent is lost to the host before any distribution is involved.
At 2 ranks that gap widens to 46 percent.

### What this changes

The 57 percent figure was never a distributed systems problem, and no amount
of tuning nccl would have moved it. It is the ratio between GPU speed and host
speed, and it only appeared because amp doubled the GPU side.

That reframes section 4's headline. Turning on mixed precision did not reveal
a flaw in DDP. It moved the bottleneck off the GPU and onto a 4 core host,
where the distributed layer has no say. The same code on a machine with 16
cores feeding two T4s would likely scale close to the 96 percent the synthetic
run shows.

### Honest limits of this experiment

* Synthetic data indexes a 512 image pool, so it is also friendlier to cache
  than real data would be even with a perfect loader. The 96 percent figure is
  an upper bound on what fixing the input path could achieve, not a promise.
* The fix is untested. GPU side augmentation, or `--num-workers` on a host
  with more cores, would confirm the diagnosis by removing the ceiling. Neither
  is possible on free Kaggle.
* Accuracy from the synthetic runs is meaningless by construction, since the
  labels are random. The artifacts carry `synthetic_data: true` and the
  dashboard prints n/a rather than the roughly 10 percent those runs report.

## 6. Bugs found while building this

The most useful artifact of the week, kept deliberately.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | CIFAR-10 download died with `SSL: UNEXPECTED_EOF_WHILE_READING` | Canonical host unreachable from this network | Mirror fallback in `scripts/fetch_cifar10.py` |
| 2 | Rebuilt pickles rejected as "Dataset not found or corrupted" | torchvision checks published md5 sums a rebuilt file cannot match | Stopped impersonating the original layout. Mirror writes its own `npz` and the loader picks a source |
| 3 | Every 2 rank run hung, no error, until timeout | torchrun's default TCPStore rendezvous on localhost is blocked here. IPv6 resolution failed in a loop | `dvit.launch`, a FileStore launcher needing no sockets. `dvit.dist` honours `DVIT_INIT_FILE` |
| 4 | MPS runs reported a peak memory figure | `torch.mps.current_allocated_memory` is instantaneous, read after activations were freed. There is no MPS high water mark | Report memory only where an allocator peak exists, and record `memory_source` |
| 5 | Parameter count in two docstrings said 2.7M | Written from an estimate before the model was built | Corrected to the measured 1,806,538 |
| 6 | FSDP runs would have under clipped gradients | `torch.nn.utils.clip_grad_norm_` sees only the local shard, so each rank clips against a norm computed from a fraction of the gradient | Use FSDP's own collective aware `model.clip_grad_norm_`. Found by inspection, not by execution, because there is no CUDA device here |
| 9 | The dashboard divided a real data run by a synthetic data baseline, printing 1.11x where the matched comparison is 1.15x | `baseline_for` matched on precision and compile but not on `num_workers`, `synthetic_data`, `epochs` or `subset_fraction`, so it paired runs that differed in more than rank count | A baseline must differ in exactly one thing. All four added to the match key. Found while recording the page for a video, which is the only reason anyone looked at that cell |
| 8 | Every T4 amp run would have used emulated bf16 | `torch.cuda.is_bf16_supported()` returns True on sm_75, counting emulation as support | Select from `torch.cuda.get_device_capability`, bf16 only at sm_80 and later. Caught by reading the first cell of the Kaggle run |
| 7 | The gloo suite reported 146, 288 and 343 img/s on a rerun, against 521, 632 and 814 the first time | The rerun overlapped the MPS suite. Both were competing for the same 10 cores | Benchmarks are run on an idle machine. Numbers taken under concurrent load are discarded, not reconciled |

Bug 3 is the one worth telling in an interview. The failure mode was not a
crash, it was a hang, which reads exactly like slow training. Anything that
makes a distributed job look slow rather than broken will cost hours before
anyone suspects the launcher.
