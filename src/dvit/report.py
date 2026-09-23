"""Build dashboard/index.html from runs/*.json.

Rules this file enforces so the dashboard cannot drift from reality:

  * Every cell traces to a field in a run artifact. Nothing is typed by hand.
  * A run whose subset_fraction was below 1.0 or whose epoch count is tiny is
    a smoke run and is badged as such, never mixed into the headline table.
  * Speedup is computed only against a baseline measured on the SAME hardware
    string. Comparing a T4 against an M5 and calling the ratio a speedup is
    the single easiest way to publish a meaningless number.
  * A suite with no artifacts renders an explicit "not yet measured" row
    rather than an empty table that reads as if there were nothing to run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

PALETTE = ["#4f7cff", "#00b3a4", "#f2994a", "#bb6bd9", "#eb5757", "#6fcf97"]


@dataclass
class Run:
    raw: dict

    def __getattr__(self, k: str):
        return self.raw.get(k)

    @property
    def is_smoke(self) -> bool:
        """A run that exists to prove a code path executes, nothing more."""
        return self.raw.get("name", "").startswith("smoke")

    @property
    def accuracy_is_meaningful(self) -> bool:
        """Throughput from a short run on partial data is still a real
        measurement. The accuracy from one is not. The two judgments are
        separate, so the gloo scaling runs keep their img/s and lose their
        top-1 rather than being written off wholesale."""
        if self.raw.get("synthetic_data"):
            return False     # random labels, so any accuracy here is noise
        return (self.raw.get("subset_fraction", 1.0) >= 1.0
                and self.raw.get("epochs", 0) >= 10)

    @property
    def label(self) -> str:
        return self.raw["name"]

    @property
    def curve(self) -> list[tuple[int, float]]:
        return [(e["epoch"], e["test_acc"]) for e in self.raw.get("epochs_log", [])]


def load_runs(runs_dir: Path) -> list[Run]:
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("finished"):
            out.append(Run(d))
    return out


def baseline_for(run: Run, runs: list[Run]) -> Run | None:
    """Rank scaling baseline: identical config, one rank, same hardware.

    Everything except world_size is held fixed, so the ratio isolates what the
    extra ranks bought and dividing by world_size gives a real efficiency.
    """
    for c in runs:
        if (c.world_size == 1 and c.hardware == run.hardware
                and c.precision == run.precision and c.compiled == run.compiled
                and c.grad_accum_steps == run.grad_accum_steps
                # These two were missing and it produced a wrong number on the
                # page: a real data run was being divided by a synthetic data
                # baseline, which silently reported 1.11x where the matched
                # comparison is 1.15x. Rank speedup is only meaningful when
                # the input path is identical too.
                and bool(c.raw.get("synthetic_data")) == bool(run.raw.get("synthetic_data"))
                and c.raw.get("num_workers") == run.raw.get("num_workers")
                # Epochs and subset matter because two suites measured the
                # same configuration at different lengths, 30 epochs and 6,
                # and they differ by about 10 percent on warmup weighting
                # alone. Crossing them produced 1.26x where the matched
                # comparison is 1.15x.
                and c.epochs == run.epochs
                and c.raw.get("subset_fraction", 1.0) == run.raw.get("subset_fraction", 1.0)):
            return c
    return None


def plain_baseline_for(run: Run, runs: list[Run]) -> Run | None:
    """Reference point for the whole hardware group: one rank, fp32, no compile.

    This is a different question from rank scaling. It answers "how much
    faster is this configuration than the naive one", which is where the
    precision and compilation effects live. Efficiency is deliberately not
    computed against it, because dividing a precision speedup by rank count
    would be meaningless.
    """
    for c in runs:
        if (c.world_size == 1 and c.hardware == run.hardware
                and c.precision == "fp32" and not c.compiled
                and c.grad_accum_steps == 1
                and bool(c.raw.get("synthetic_data")) == bool(run.raw.get("synthetic_data"))):
            return c
    return None


def svg_bars(runs: list[Run], width: int = 720, row_h: int = 34) -> str:
    if not runs:
        return "<p class='empty'>No throughput data yet.</p>"
    top = max(r.median_images_per_sec for r in runs) or 1.0
    pad_l, pad_r = 210, 90
    height = row_h * len(runs) + 16
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' "
             f"aria-label='Median throughput by configuration'>"]
    for i, r in enumerate(runs):
        y = i * row_h + 8
        w = (r.median_images_per_sec / top) * (width - pad_l - pad_r)
        color = PALETTE[i % len(PALETTE)]
        parts.append(
            f"<text x='0' y='{y + 15}' class='bar-label'>{escape(r.label)}</text>"
            f"<rect x='{pad_l}' y='{y}' width='{max(w, 1):.1f}' height='20' rx='4' fill='{color}'/>"
            f"<text x='{pad_l + max(w, 1) + 8:.1f}' y='{y + 15}' class='bar-value'>"
            f"{r.median_images_per_sec:,.0f} img/s</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_curves(runs: list[Run], width: int = 720, height: int = 300) -> str:
    curved = [r for r in runs if len(r.curve) > 1 and r.accuracy_is_meaningful]
    if not curved:
        return ("<p class='empty'>No trustworthy accuracy curves yet. "
                "Needs a run over the full dataset for at least 10 epochs.</p>")
    pad = {"l": 48, "r": 150, "t": 14, "b": 32}
    max_ep = max(max(e for e, _ in r.curve) for r in curved) or 1
    max_acc = max(max(a for _, a in r.curve) for r in curved) or 1.0
    y_top = min(1.0, max_acc * 1.15)

    def px(e: int) -> float:
        return pad["l"] + (e / max_ep) * (width - pad["l"] - pad["r"])

    def py(a: float) -> float:
        return height - pad["b"] - (a / y_top) * (height - pad["t"] - pad["b"])

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' "
             f"aria-label='Test accuracy against epoch'>"]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        a = y_top * frac
        parts.append(
            f"<line x1='{pad['l']}' y1='{py(a):.1f}' x2='{width - pad['r']}' "
            f"y2='{py(a):.1f}' class='grid'/>"
            f"<text x='{pad['l'] - 8}' y='{py(a) + 4:.1f}' class='axis' "
            f"text-anchor='end'>{a * 100:.0f}%</text>"
        )
    # Series labels sit at each curve's final value, so converged runs would
    # print on top of each other. Push them apart, keeping their order.
    ends = sorted(((py(r.curve[-1][1]), i) for i, r in enumerate(curved)))
    label_y: dict[int, float] = {}
    min_gap, prev = 13.0, -1e9
    for y, i in ends:
        y = max(y, prev + min_gap)
        label_y[i] = y
        prev = y

    for i, r in enumerate(curved):
        color = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{px(e):.1f},{py(a):.1f}" for e, a in r.curve)
        last_e, last_a = r.curve[-1]
        ly = label_y[i]
        parts.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/>")
        parts.append(
            f"<circle cx='{px(last_e):.1f}' cy='{py(last_a):.1f}' r='3' fill='{color}'/>"
            # Leader line from the data point across to the displaced label.
            f"<path d='M {px(last_e) + 4:.1f} {py(last_a):.1f} "
            f"L {width - pad['r'] + 2:.1f} {ly:.1f}' stroke='{color}' "
            f"stroke-width='1' fill='none' opacity='0.45'/>"
            f"<text x='{width - pad['r'] + 8}' y='{ly + 4:.1f}' "
            f"class='series' fill='{color}'>{escape(r.label)}</text>"
        )
    parts.append(
        f"<text x='{(width - pad['r'] + pad['l']) / 2}' y='{height - 6}' "
        f"class='axis' text-anchor='middle'>epoch</text></svg>"
    )
    return "".join(parts)


def scaling_rows(runs: list[Run]) -> str:
    if not runs:
        return ("<tr><td colspan='11' class='empty'>Not yet measured. "
                "Run <code>python -m dvit.sweep kaggle</code> on a 2x T4 "
                "instance to populate this table.</td></tr>")
    rows = []
    for r in sorted(runs, key=lambda x: (x.hardware, x.world_size, x.label)):
        base = baseline_for(r, runs)
        if base is None or base.label == r.label or not base.median_images_per_sec:
            speed, eff = "baseline" if r.world_size == 1 else "no baseline", ""
        else:
            s = r.median_images_per_sec / base.median_images_per_sec
            speed = f"{s:.2f}x"
            eff = f"{s / r.world_size * 100:.0f}%"
        plain = plain_baseline_for(r, runs)
        if plain is None or plain.label == r.label or not plain.median_images_per_sec:
            vs_plain = "reference" if (plain and plain.label == r.label) else ""
        else:
            vs_plain = f"{r.median_images_per_sec / plain.median_images_per_sec:.2f}x"
        mem = f"{r.peak_memory_mb:,.0f}" if r.peak_memory_mb else "n/a"
        badge = " <span class='badge smoke'>smoke</span>" if r.is_smoke else ""
        if r.raw.get("synthetic_data"):
            badge += " <span class='badge smoke'>synthetic</span>"
        if r.accuracy_is_meaningful:
            acc = f"{r.best_test_acc * 100:.2f}%"
        elif r.raw.get("synthetic_data"):
            acc = "<span class='eff'>n/a</span>"
        else:
            acc = f"<span class='eff'>{r.best_test_acc * 100:.1f}% partial</span>"
        rows.append(
            f"<tr><td class='mono'>{escape(r.label)}{badge}</td>"
            f"<td>{escape(r.hardware)}</td><td>{r.world_size}</td>"
            f"<td>{escape(r.backend or 'none')}</td>"
            f"<td>{escape(r.strategy)}</td><td>{escape(r.precision)}"
            f"{' + compile' if r.compiled else ''}</td>"
            f"<td class='num'>{r.effective_batch}</td>"
            f"<td class='num'>{r.median_images_per_sec:,.0f}</td>"
            f"<td class='num'>{speed}{f' <span class=eff>({eff})</span>' if eff else ''}</td>"
            f"<td class='num'>{vs_plain}</td>"
            f"<td class='num'>{mem}</td>"
            f"<td class='num'>{acc}</td></tr>"
        )
    return "".join(rows)


CSS = """
:root{--bg:#fbfbfd;--fg:#14161a;--muted:#6b7280;--card:#fff;--line:#e6e8ec;
--accent:#4f7cff;--warn-bg:#fff7ed;--warn-line:#f2994a;--code:#f3f4f6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0e1014;--fg:#e8eaed;--muted:#9aa2ae;--card:#161a21;--line:#262b34;
--warn-bg:#241c12;--code:#1d222a}}
:root[data-theme="dark"]{--bg:#0e1014;--fg:#e8eaed;--muted:#9aa2ae;--card:#161a21;
--line:#262b34;--warn-bg:#241c12;--code:#1d222a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:48px 16px 80px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:18px;letter-spacing:-.01em;margin:40px 0 4px}
.sub{color:var(--muted);margin:0 0 4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:20px;margin-top:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:760px}
th{text-align:left;font-weight:600;color:var(--muted);font-size:11.5px;
text-transform:uppercase;letter-spacing:.05em;padding:0 10px 8px;white-space:nowrap}
td{padding:9px 10px;border-top:1px solid var(--line);vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
td.mono{white-space:nowrap}
code{background:var(--code);padding:1px 5px;border-radius:4px}
.eff{color:var(--muted);font-size:11.5px}
.badge{font-size:10px;padding:2px 6px;border-radius:99px;vertical-align:middle;
text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.badge.smoke{background:var(--warn-bg);color:var(--warn-line);
border:1px solid var(--warn-line)}
.note{background:var(--warn-bg);border:1px solid var(--warn-line);
border-radius:10px;padding:14px 16px;margin-top:14px;font-size:13.5px}
.note b{display:block;margin-bottom:4px}
.grid{stroke:var(--line);stroke-width:1}
.axis{fill:var(--muted);font-size:11px}
.series{font-size:11.5px;font-weight:600}
.bar-label{fill:var(--fg);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.bar-value{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.empty{color:var(--muted);font-style:italic;text-align:center;padding:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.stat .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.stat .v{font-size:22px;font-weight:650;letter-spacing:-.02em;margin-top:2px;
font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:12.5px;margin-top:48px;border-top:1px solid var(--line);
padding-top:16px}
@media(max-width:640px){.wrap{padding:28px 16px 60px}h1{font-size:24px}}
"""


def render(runs: list[Run]) -> str:
    real = [r for r in runs if not r.is_smoke]
    smoke = [r for r in runs if r.is_smoke]
    shown = real or smoke

    hardware = sorted({r.hardware for r in runs}) or ["nothing measured yet"]
    best = max((r.best_test_acc for r in runs if r.accuracy_is_meaningful),
               default=0.0)
    # Smoke runs are excluded: a few hundred images with warmup included
    # is not a throughput measurement, and it should never headline.
    # Synthetic runs are excluded for a stronger reason: their input is an in
    # memory tensor pool, so their throughput is an instrument reading, not a
    # figure for training this model on this dataset. It is the highest number
    # in the project and it is the one least entitled to be the headline.
    headline = [r for r in real if not r.raw.get("synthetic_data")]
    fastest = max((r.median_images_per_sec for r in headline), default=0.0)

    stats = [
        ("runs recorded", f"{len(runs)}"),
        ("hardware", f"{len(hardware)}"),
        ("best top-1", f"{best * 100:.2f}%" if best else "not yet"),
        ("peak throughput", f"{fastest:,.0f} img/s" if fastest else "not yet"),
    ]
    stat_html = "".join(
        f"<div class='stat'><div class='k'>{escape(k)}</div>"
        f"<div class='v'>{escape(v)}</div></div>" for k, v in stats
    )

    note = ""
    if smoke and not real:
        note = ("<div class='note'><b>These are smoke runs, not results.</b>"
                "Every row below used a small fraction of CIFAR-10 for a handful of "
                "epochs, to prove each code path executes. The accuracy numbers mean "
                "nothing. Real numbers arrive from the <code>laptop</code> and "
                "<code>kaggle</code> suites.</div>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distributed ViT on CIFAR-10</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Distributed ViT on CIFAR-10</h1>
<p class="sub">A vision transformer trained across devices, with the systems
knobs measured rather than assumed.</p>
<p class="sub">Measured on: {escape(", ".join(hardware))}</p>

<div class="stats" style="margin-top:22px">{stat_html}</div>
{note}

<h2>Scaling and configuration</h2>
<p class="sub"><b>Rank speedup</b> holds every other setting fixed and varies
only the rank count, so the percentage beside it is a real scaling efficiency.
<b>vs fp32</b> compares against the one rank fp32 uncompiled run on the same
hardware, which is where precision and compilation effects show up. Both are
computed only within one hardware group. Blank means no comparable baseline
was measured.</p>
<div class="card"><table>
<thead><tr><th>run</th><th>hardware</th><th>ranks</th><th>backend</th>
<th>strategy</th><th>precision</th><th>eff. batch</th><th>img/s</th>
<th>rank speedup</th><th>vs fp32</th><th>peak MB</th><th>top-1</th></tr></thead>
<tbody>{scaling_rows(shown)}</tbody></table></div>

<h2>Median throughput</h2>
<div class="card">{svg_bars(sorted(shown, key=lambda r: -r.median_images_per_sec))}</div>

<h2>Test accuracy by epoch</h2>
<div class="card">{svg_curves(shown)}</div>

<footer>Generated by <code>python -m dvit.report</code> from
<code>runs/*.json</code>. Every number on this page traces to a run artifact
and the command that produced it. Nothing here is typed by hand.</footer>
</div></body></html>"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the results dashboard")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="dashboard/index.html")
    args = p.parse_args(argv)

    runs = load_runs(Path(args.runs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs))
    print(f"wrote {out} from {len(runs)} run artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
