"""Fail if a number with a unit attached does not trace to a run artifact.

`CLAUDE.md` says never report a number the code did not measure, and
`RESULTS.md` says every figure traces to `runs/`. Until now both were
promises kept by diligence, and three separate bugs this project documents
were wrong numbers that survived precisely because nobody diffed prose
against artifacts.

Scope is deliberately narrow: only numbers carrying a measurement unit are
checked, because those are the ones making a claim about reality.

    4,706 img/s     throughput
    347 MB          peak memory
    1.96x           a ratio between two runs
    98 percent      an accuracy, an efficiency, or a relative change

Epoch counts, batch sizes, section numbers and years are not checked. They
are configuration or prose, not measurements.

A claim passes if some artifact value, or some ratio between two artifact
values on the same hardware, rounds to it at the precision the claim was
written with. Writing 1.96x when the artifacts give 1.963 is fine. Writing
1.8x is not.

    python -m dvit.verify_claims
    python -m dvit.verify_claims --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCES = ["RESULTS.md", "README.md", "CLAUDE.md"]

PATTERNS = {
    "throughput": re.compile(r"([\d,]+(?:\.\d+)?)\s*img/s"),
    "memory": re.compile(r"\b([\d,]+(?:\.\d+)?)\s*MB\b"),
    "ratio": re.compile(r"\b(\d+\.\d+)x\b"),
    # No trailing \b after the alternation: a percent sign followed by a
    # space is two non word characters, so \b never matches there and every
    # figure written as 98% was silently skipped. Only the spelled out form
    # was ever being checked.
    "percent": re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)"),
}


@dataclass(frozen=True)
class Claim:
    source: str
    line_no: int
    kind: str
    text: str
    value: float
    line: str


def load_runs(runs_dir: Path, pattern: str = "*.json") -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob(pattern)):
        d = json.loads(p.read_text())
        if d.get("finished"):
            out.append(d)
    return out


def measured(runs: list[dict]) -> dict[str, set[float]]:
    """Every value an artifact states, plus every ratio the docs could form."""
    tput = {r["median_images_per_sec"] for r in runs}
    mem = {r["peak_memory_mb"] for r in runs if r["peak_memory_mb"]}
    # Gradient payload per all reduce, derived from a recorded field rather
    # than asserted. The docs say "about 7 MB"; this is where that comes from.
    mem |= {r["model_params"] * 4 / 1024 ** 2 for r in runs}
    pct = {r["best_test_acc"] * 100 for r in runs}
    ratios: set[float] = set()

    for a in runs:
        for b in runs:
            if a is b or a["hardware"] != b["hardware"]:
                continue
            if b["median_images_per_sec"]:
                r = a["median_images_per_sec"] / b["median_images_per_sec"]
                ratios.add(r)
                # Efficiency, and the two ways a relative change gets written.
                pct.add(r / max(1, a["world_size"]) * 100)
                pct.add(r * 100)
                pct.add(abs(100 - r * 100))
            if a["peak_memory_mb"] and b["peak_memory_mb"]:
                pct.add(abs(100 - a["peak_memory_mb"] / b["peak_memory_mb"] * 100))
    return {"throughput": tput, "memory": mem, "ratio": ratios, "percent": pct}


def rounds_to(value: float, claim: str) -> bool:
    """Does `value` round to the claim, at the precision the claim used?"""
    decimals = len(claim.split(".")[1]) if "." in claim else 0
    target = float(claim.replace(",", ""))
    return abs(round(value, decimals) - target) < 10 ** (-decimals) / 2


def extract(path: Path) -> list[Claim]:
    claims = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if line.lstrip().startswith(("|---", "```")):
            continue
        for kind, pat in PATTERNS.items():
            for m in pat.finditer(line):
                raw = m.group(1)
                claims.append(Claim(path.name, i, kind, raw,
                                    float(raw.replace(",", "")), line.strip()))
    return claims


def extract_captions(path: Path) -> list[Claim]:
    if not path.exists():
        return []
    out = []
    for c in json.loads(path.read_text()):
        text = re.sub(r"<[^>]+>", "", c["text"])
        for kind, pat in PATTERNS.items():
            for m in pat.finditer(text):
                raw = m.group(1)
                out.append(Claim(path.name, 0, kind, raw,
                                 float(raw.replace(",", "")), text))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    p.add_argument("--captions", default=None,
                   help="a brag captions.json to check as well")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    runs = load_runs(Path(args.runs))
    if not runs:
        print("no finished run artifacts, nothing to verify against", file=sys.stderr)
        return 2
    pool = measured(runs)

    # Quarantined measurements really happened, they were just discarded as
    # invalid. Prose that describes a discarded number is telling the truth,
    # so it passes, but the gate says so out loud rather than staying silent.
    discarded = load_runs(Path(args.runs) / "invalid")
    pool_discarded = measured(discarded) if discarded else {k: set() for k in pool}

    claims: list[Claim] = []
    for s in args.sources:
        path = Path(s)
        if path.exists():
            claims.extend(extract(path))
    if args.captions:
        claims.extend(extract_captions(Path(args.captions)))

    unsupported, from_discarded = [], []
    for c in claims:
        if any(rounds_to(v, c.text) for v in pool[c.kind]):
            continue
        if any(rounds_to(v, c.text) for v in pool_discarded[c.kind]):
            from_discarded.append(c)
        else:
            unsupported.append(c)

    by_kind: dict[str, int] = {}
    for c in claims:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    print(f"checked {len(claims)} unit bearing numbers against {len(runs)} artifacts "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(by_kind.items()))})")

    if args.verbose:
        for c in claims:
            ok = any(rounds_to(v, c.text) for v in pool[c.kind])
            print(f"  {'ok  ' if ok else 'FAIL'} {c.source}:{c.line_no} {c.kind} {c.text}")

    if from_discarded:
        print(f"\n{len(from_discarded)} claim(s) cite a DISCARDED measurement "
              f"(runs/invalid). Allowed, and worth seeing:\n")
        for c in from_discarded:
            print(f"  {c.source}:{c.line_no}  {c.kind} = {c.text}")

    if unsupported:
        print(f"\n{len(unsupported)} claim(s) with no matching artifact:\n")
        for c in unsupported:
            print(f"  {c.source}:{c.line_no}  {c.kind} = {c.text}")
            print(f"    {c.line[:105]}")
        return 1

    print("every unit bearing number traces to an artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
