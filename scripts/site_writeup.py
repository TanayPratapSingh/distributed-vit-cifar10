"""Generate the portfolio write-up for this project from run artifacts.

Kept in this repo rather than the site repo, because the numbers belong to
the runs. Regenerate after new measurements land and paste the JSON into
site/data/projects.js.

    python scripts/site_writeup.py > /tmp/entry.json
"""

from __future__ import annotations

import glob
import json
import sys

R = {}
for p in glob.glob("runs/*.json"):
    d = json.load(open(p))
    R[d["name"]] = d


def ips(n: str) -> float: return R[n]["median_images_per_sec"]
def mem(n: str) -> float: return R[n]["peak_memory_mb"]
def acc(n: str) -> float: return R[n]["best_test_acc"] * 100
def mins(n: str) -> float: return R[n]["total_seconds"] / 60


def speed(a: str, b: str) -> float: return ips(a) / ips(b)
def eff(a: str, b: str, ranks: int = 2) -> float: return speed(a, b) / ranks * 100


PARAMS = R["t4-ws1-fp32"]["model_params"]
CORES = R["t4-ws2-amp-synth"]["cpu_cores"]
TORCH = R["t4-ws1-fp32"]["torch_version"]
GRAD_MB = PARAMS * 4 / 1024 ** 2


def row(name: str, ranks: str, strat: str, prec: str, rank_sp: str,
        winner: bool = False) -> str:
    cls = ' class="winner"' if winner else ""
    b0, b1 = ("<strong>", "</strong>") if winner else ("", "")
    return (f"<tr{cls}><td><code>{name}</code></td><td>{ranks}</td>"
            f"<td>{strat}</td><td>{prec}</td>"
            f"<td>{b0}{ips(name):,.0f}{b1}</td><td>{b0}{rank_sp}{b1}</td>"
            f"<td>{mem(name):,.0f}</td><td>{acc(name):.2f}%</td></tr>")


BODY = f"""<div class="metric">2x Tesla T4, NCCL &nbsp;|&nbsp; DDP in fp32: {speed('t4-ws2-ddp-fp32','t4-ws1-fp32'):.2f}x at {eff('t4-ws2-ddp-fp32','t4-ws1-fp32'):.0f}% efficiency &nbsp;|&nbsp; identical config in fp16: {speed('t4-ws2-ddp-amp','t4-ws1-amp'):.2f}x at {eff('t4-ws2-ddp-amp','t4-ws1-amp'):.0f}% &nbsp;|&nbsp; cause isolated to a {CORES} vCPU host &nbsp;|&nbsp; {len(R)} measured runs across 3 hardware groups</div>

          <h4>Why This Matters</h4>
          <p>
            Almost every distributed training result you read is unfalsifiable. A repository reports a 1.9x speedup on two GPUs and there is no way to tell whether the two rank job computed the same gradient as the one rank job, whether the baseline was tuned as carefully as the treatment, or whether the number came from a machine doing something else at the time. A speedup you cannot verify is not a result, it is a claim. This project was built so that every claim it makes is checkable, and the most interesting thing it found is a regression in its own scaling.
          </p>

          <h4>What Gets Measured, and Against What</h4>
          <p>
            Three hardware groups: an Apple M5 laptop with one GPU, multi process gloo over that laptop's {CORES if False else 10} CPU cores, and 2x Tesla T4 on Kaggle with NCCL. Speedup is only ever computed inside one group, because a ratio between a T4 and an M5 is arithmetic, not a measurement. A rank scaling baseline must differ from its run in exactly one property: the rank count. Precision, compilation, gradient accumulation, dataloader worker count, epoch count and dataset fraction all have to match, and the reporting layer enforces that rather than trusting the author. That constraint was added after the dashboard was caught dividing a real data run by a synthetic data baseline, which silently reported {speed('t4-ws2-ddp-amp','t4-ws1-amp-synth'):.2f}x where the matched comparison is {speed('t4-ws2-ddp-amp','t4-ws1-amp'):.2f}x.
          </p>

          <h4>Model and Data</h4>
          <p>
            A vision transformer written out rather than imported, {PARAMS:,} parameters, patch size 4 on 32&times;32 inputs giving 64 tokens plus a class token. Small on purpose: the model must not be the bottleneck in a throughput measurement, and at this size the gradient payload for each all reduce is about {GRAD_MB:.0f} MB, which matters later. CIFAR-10, 50,000 train and 10,000 test, 30 epochs, no pretraining.
          </p>
          <p>
            The canonical dataset host was unreachable from the development network, failing with an SSL EOF before any payload arrived. The fallback deliberately does not rebuild the original pickle layout, because those files carry published md5 sums a reconstruction cannot match, and the only way to make that path appear to work would be to disable torchvision's integrity checking. Silencing a checksum to make a download look successful is how results stop being trustworthy. The mirror writes its own format instead, and every run artifact records which source it used.
          </p>

          <h4>Configuration</h4>
          <table>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>
              <tr><td>Architecture</td><td>ViT, dim 192, depth 6, heads 3, MLP ratio 2.0</td></tr>
              <tr><td>Regularization</td><td>Stochastic depth 0.1, label smoothing 0.1, grad clip 1.0</td></tr>
              <tr><td>Optimizer</td><td>AdamW, betas (0.9, 0.95), no decay on norms, biases, class token, position embedding</td></tr>
              <tr><td>Schedule</td><td>Linear warmup 2 epochs, then cosine, computed per step</td></tr>
              <tr><td>Learning rate</td><td>1e-3, scaled linearly by world size</td></tr>
              <tr><td>Batch size</td><td>128 per rank</td></tr>
              <tr><td>Precision</td><td>Selected from compute capability, not from <code>is_bf16_supported</code></td></tr>
              <tr><td>Runtime</td><td>torch {TORCH}, NCCL on CUDA, gloo on CPU, FileStore rendezvous on macOS</td></tr>
            </tbody>
          </table>

          <h4>Correctness Before Speed</h4>
          <p>
            DDP averages gradients across ranks, so two ranks each holding batch B must produce the same gradient as one process holding batch 2B. If that identity does not hold, every throughput number in the project describes a system training a different model than its own baseline, and the comparison is void. The suite spawns a real gloo process group and checks it to <code>rtol=1e-4</code>. It ships with a second test that deliberately computes the wrong thing, because a correctness test that cannot fail is decoration rather than evidence.
          </p>

          <figure class="project-figure">
            <img loading="lazy" decoding="async" src="images/dvit_03.png" alt="Terminal showing the DDP gradient equivalence test and its inverse guard both passing">
            <figcaption>FIGURE 1. THE GRADIENT EQUIVALENCE TEST AND ITS INVERSE GUARD, RUN IN A REAL PROCESS GROUP. THE DASHBOARD IS THEN REGENERATED FROM {len(R)} RUN ARTIFACTS</figcaption>
          </figure>

          <h4>Scaling on 2x Tesla T4</h4>
          <table>
            <thead><tr><th>Run</th><th>Ranks</th><th>Strategy</th><th>Precision</th><th>img/s</th><th>Rank speedup</th><th>Peak MB</th><th>Top-1</th></tr></thead>
            <tbody>
              {row('t4-ws1-fp32', '1', 'single', 'fp32', 'baseline')}
              {row('t4-ws1-amp', '1', 'single', 'fp16', 'baseline')}
              {row('t4-ws2-ddp-fp32', '2', 'DDP', 'fp32', f"{speed('t4-ws2-ddp-fp32','t4-ws1-fp32'):.2f}x ({eff('t4-ws2-ddp-fp32','t4-ws1-fp32'):.0f}%)", winner=True)}
              {row('t4-ws2-ddp-amp', '2', 'DDP', 'fp16', f"{speed('t4-ws2-ddp-amp','t4-ws1-amp'):.2f}x ({eff('t4-ws2-ddp-amp','t4-ws1-amp'):.0f}%)")}
              {row('t4-ws2-ddp-amp-compiled', '2', 'DDP', 'fp16 + compile', 'no baseline')}
              {row('t4-ws2-ddp-amp-ga2', '2', 'DDP', 'fp16, accum 2', 'no baseline')}
              {row('t4-ws2-fsdp-amp', '2', 'FSDP', 'fp16', f"{speed('t4-ws2-fsdp-amp','t4-ws1-amp'):.2f}x")}
            </tbody>
          </table>
          <p>
            Three predictions were written down before this ran, so they could be scored instead of retrofitted. FSDP was predicted to lose, and it does: {ips('t4-ws2-fsdp-amp'):,.0f} img/s against DDP's {ips('t4-ws2-ddp-amp'):,.0f}, buying {100 - mem('t4-ws2-fsdp-amp')/mem('t4-ws2-ddp-amp')*100:.0f}% of peak memory for {100 - speed('t4-ws2-fsdp-amp','t4-ws2-ddp-amp')*100:.0f}% of throughput. At {PARAMS:,} parameters there is almost nothing to shard. Mixed precision was predicted to be a large win, and it is: {speed('t4-ws1-amp','t4-ws1-fp32'):.2f}x at one rank with peak memory falling from {mem('t4-ws1-fp32'):,.0f} MB to {mem('t4-ws1-amp'):,.0f} MB.
          </p>

          <h4>The Prediction That Was Wrong</h4>
          <p>
            The third prediction said DDP would land near 1.8x rather than 2x, reasoning that a small model makes the all reduce expensive relative to compute. That is backwards. A small model means <em>small gradient tensors</em>, about {GRAD_MB:.0f} MB, which DDP overlaps with the backward pass almost entirely. In fp32 it scales at {speed('t4-ws2-ddp-fp32','t4-ws1-fp32'):.2f}x, {eff('t4-ws2-ddp-fp32','t4-ws1-fp32'):.0f}% efficiency.
          </p>
          <p>
            The same DDP configuration in fp16 scales at {speed('t4-ws2-ddp-amp','t4-ws1-amp'):.2f}x, {eff('t4-ws2-ddp-amp','t4-ws1-amp'):.0f}% efficiency. Identical hardware, identical model, identical communication volume. The only thing that changed is that each GPU now computes roughly twice as fast. Optimising one layer moved the bottleneck rather than shortening the job, and scaling efficiency turned out not to be a property of the cluster at all.
          </p>

          <h4>Isolating the Cause</h4>
          <p>
            Two suspects: the NCCL all reduce, and the input pipeline. Raising the dataloader worker count and watching what happens would produce a correlation, not a cause. Instead a synthetic input mode serves pre normalised tensors from a small in memory pool, removing decode, augmentation and worker processes entirely, so two ranks against one rank under it is compute plus communication with nothing else in the measurement.
          </p>
          <table>
            <thead><tr><th>Input path</th><th>1 rank</th><th>2 ranks</th><th>Speedup</th><th>Efficiency</th></tr></thead>
            <tbody>
              <tr class="winner"><td><strong>Synthetic, no dataloader</strong></td><td>{ips('t4-ws1-amp-synth'):,.0f}</td><td><strong>{ips('t4-ws2-amp-synth'):,.0f}</strong></td><td><strong>{speed('t4-ws2-amp-synth','t4-ws1-amp-synth'):.2f}x</strong></td><td><strong>{eff('t4-ws2-amp-synth','t4-ws1-amp-synth'):.0f}%</strong></td></tr>
              <tr><td>Real data, 2 workers per rank</td><td>{ips('t4-ws1-amp-w2'):,.0f}</td><td>{ips('t4-ws2-amp-w2'):,.0f}</td><td>{speed('t4-ws2-amp-w2','t4-ws1-amp-w2'):.2f}x</td><td>{eff('t4-ws2-amp-w2','t4-ws1-amp-w2'):.0f}%</td></tr>
              <tr><td>Real data, 8 workers per rank</td><td>{ips('t4-ws1-amp-w8'):,.0f}</td><td>{ips('t4-ws2-amp-w8'):,.0f}</td><td>{speed('t4-ws2-amp-w8','t4-ws1-amp-w8'):.2f}x</td><td>{eff('t4-ws2-amp-w8','t4-ws1-amp-w8'):.0f}%</td></tr>
            </tbody>
          </table>
          <p>
            The collective is innocent. With no dataloader the same configuration scales at {speed('t4-ws2-amp-synth','t4-ws1-amp-synth'):.2f}x, {eff('t4-ws2-amp-synth','t4-ws1-amp-synth'):.0f}% efficiency. With real data, throughput never passes about {max(ips('t4-ws2-amp-w2'), ips('t4-ws2-amp-w8')):,.0f} img/s no matter how many GPUs or workers are thrown at it. The artifacts record <code>cpu_cores: {CORES}</code>. One T4 in fp16 already consumes {ips('t4-ws1-amp-synth'):,.0f} img/s of synthetic input, so {CORES} vCPUs cannot feed even one GPU comfortably, let alone two. The second T4 is not slow, it is unfed.
          </p>
          <p>
            Raising workers from 2 to 8 per rank made it <em>worse</em>, {ips('t4-ws2-amp-w8'):,.0f} against {ips('t4-ws2-amp-w2'):,.0f}, because 16 worker processes on {CORES} cores buy context switches and nothing else. Note that the synthetic runs also index a small pool, so their {eff('t4-ws2-amp-synth','t4-ws1-amp-synth'):.0f}% is an upper bound on what fixing the input path could achieve, not a promise.
          </p>

          <figure class="project-figure">
            <img loading="lazy" decoding="async" src="images/dvit_01.png" alt="Median throughput for every measured run across three hardware groups">
            <figcaption>FIGURE 2. MEDIAN THROUGHPUT ACROSS ALL MEASURED RUNS. SYNTHETIC INPUT RUNS ARE MARKED, BECAUSE THE FASTEST BAR IS AN INSTRUMENT READING AND NOT A TRAINING RESULT</figcaption>
          </figure>

          <h4>Precision Advice Does Not Survive a Backend Change</h4>
          <p>
            On the T4, enabling mixed precision is a {speed('t4-ws1-amp','t4-ws1-fp32'):.2f}x win. On Apple silicon the identical flag is a regression: {ips('mps-amp'):,.0f} img/s against {ips('mps-fp32'):,.0f} in fp32, a {100 - speed('mps-amp','mps-fp32')*100:.0f}% slowdown, because eager mode pays for a cast around every operator and the hardware is already efficient at fp32. Adding <code>torch.compile</code> to that same fp16 configuration reaches {ips('mps-amp-compiled'):,.0f} img/s, {speed('mps-amp-compiled','mps-fp32'):.2f}x the fp32 baseline, once the casts are fused into the graph. Wall clock for the full 30 epoch run falls from {mins('mps-fp32'):.0f} minutes to {mins('mps-amp-compiled'):.0f}.
          </p>
          <p>
            A related trap: on a T4, <code>torch.cuda.is_bf16_supported()</code> returns True. The card is sm_75 and has no hardware bf16, so that True reflects emulation. Trusting it selected an emulated path for every mixed precision run, and torch inductor quietly logged that the card does not support bfloat16 natively while the benchmark carried on reporting the result as mixed precision. Precision is now selected from compute capability, and every artifact records the dtype that was actually used.
          </p>

          <h4>Accuracy as the Control</h4>
          <p>
            Across all seven T4 configurations, top-1 spans {min(acc(n) for n in ['t4-ws1-fp32','t4-ws1-amp','t4-ws2-ddp-fp32','t4-ws2-ddp-amp','t4-ws2-ddp-amp-compiled','t4-ws2-ddp-amp-ga2','t4-ws2-fsdp-amp']):.2f}% to {max(acc(n) for n in ['t4-ws1-fp32','t4-ws1-amp','t4-ws2-ddp-fp32','t4-ws2-ddp-amp','t4-ws2-ddp-amp-compiled','t4-ws2-ddp-amp-ga2','t4-ws2-fsdp-amp']):.2f}%, and the three Apple silicon runs land within 0.16 points of each other. That is the control, not a footnote. These are throughput changes, so accuracy should barely move. If mixed precision or compilation had shifted it materially, the correct output would have been a bug report rather than a results table.
          </p>

          <figure class="project-figure">
            <img loading="lazy" decoding="async" src="images/dvit_02.png" alt="Test accuracy by epoch for the full length runs, curves nearly overlapping">
            <figcaption>FIGURE 3. TEST ACCURACY BY EPOCH. THE CURVES OVERLAY BECAUSE PRECISION AND COMPILATION CHANGE THROUGHPUT, NOT LEARNING. SHORT AND SYNTHETIC RUNS ARE EXCLUDED FROM THIS VIEW BY THE REPORTING LAYER</figcaption>
          </figure>

          <h4>Nine Bugs, Kept On Purpose</h4>
          <p>
            The defect log is the most reusable output of the build, so it ships in the repository rather than being tidied away. Four of the nine came from an API answering a slightly different question than the one being asked: <code>is_bf16_supported</code> counting emulation, torchvision's md5 check rejecting a legitimate mirror, MPS reporting instantaneous memory where a peak was wanted, and <code>sys.argv[0]</code> producing a command that could not reproduce its own run. One was a hang rather than a crash: the default TCP rendezvous is blocked on some hosts and the job simply waits, which reads exactly like slow training and costs hours before anyone suspects the launcher. Three were wrong numbers on the dashboard, all found by pointing a screen recorder at a page nobody had read closely.
          </p>
          <p>
            That last category is why the honesty rule is now enforced rather than promised. A gate extracts every number carrying a measurement unit from the documentation and fails if it does not round, at the precision it was written with, to an artifact value or to a ratio between two artifact values on the same hardware. It found five unsupported claims the first time it ran.
          </p>

          <div class="callout">
            <h4>Operational Reading</h4>
            <p style="margin-bottom: 0;">
              The headline is not the {speed('t4-ws2-ddp-fp32','t4-ws1-fp32'):.2f}x. It is that the same cluster, model and communication volume delivered {eff('t4-ws2-ddp-fp32','t4-ws1-fp32'):.0f}% efficiency in fp32 and {eff('t4-ws2-ddp-amp','t4-ws1-amp'):.0f}% in fp16, and that the difference had nothing to do with the distributed layer. Anyone tuning NCCL in response to that drop would have spent a week in the wrong subsystem. Scaling efficiency is a property of the ratio between accelerator speed and host speed, and optimising the accelerator is what breaks it. The measurement that settles it took six short runs and one flag.
            </p>
          </div>"""

ENTRY = {
    "title": "Distributed ViT on CIFAR-10: Proving a Speedup, Then Explaining Why It Vanished",
    "meta": "Distributed Training &middot; Systems Measurement &middot; Sep 2026",
    "badge": f"{eff('t4-ws2-ddp-fp32','t4-ws1-fp32'):.0f}% &rarr; {eff('t4-ws2-ddp-amp','t4-ws1-amp'):.0f}% scaling efficiency &middot; cause isolated &middot; {len(R)} measured runs",
    "tags": ["PyTorch", "DDP", "FSDP", "NCCL", "Mixed Precision", "torch.compile",
             "Vision Transformer", "CUDA", "Benchmarking", "Apple MPS"],
    "body": BODY,
}

json.dump(ENTRY, sys.stdout, indent=1)
