"""Render each caption to a transparent PNG through Chromium.

This ffmpeg build has neither freetype nor libass, so `drawtext` and the
`subtitles` filter both fail. Rendering through a browser is not a workaround
so much as an upgrade: it buys real typography, and identifiers can be set in
monospace so `fp32` and `amp` read as code rather than prose.

Timings live beside the text here, and they were sampled from the rendered
video, not from the capture script's intended waits. agg compresses idle gaps,
so the two timelines do not match.
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROMIUM = ("/Users/tanay/Library/Caches/ms-playwright/chromium-1243/"
            "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/"
            "Google Chrome for Testing")
W, H = 1280, 170

# (id, start, end, html). Every claim below is on screen in the same shot or
# the one before it, and no figure appears that runs/*.json does not contain.
CAPTIONS = [
    ("c1", 0.20, 3.60, "A speedup you cannot verify is not a result."),
    ("c2", 3.90, 6.00, "Two ranks must equal one process."),
    ("c3", 6.20, 9.30, "Every number traces to a run artifact."),
    ("c4", 9.80, 12.40, "<b>2x Tesla T4</b> on Kaggle. Every row measured."),
    # c5 and c6 both sit inside the 14.89 to 20.29 window where the T4 table
    # is held, so every figure they state is printed on screen beneath them.
    ("c5", 14.95, 17.60,
     "<m>fp32</m> scales at <b>98%</b>. <m>amp</m> drops to <b>57%</b>."),
    ("c6", 17.85, 20.25, "Remove the dataloader: back to <b>96%</b>."),
    ("c7", 21.40, 23.40, "It was never the GPUs."),
]

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  html,body{{margin:0;padding:0;background:transparent;width:{w}px;height:{h}px}}
  .band{{width:{w}px;height:{h}px;display:flex;align-items:center;
         justify-content:center;padding:0 90px;box-sizing:border-box}}
  p{{margin:0;text-align:center;color:#e8eaed;
     font:400 37px/1.35 Inter,-apple-system,"Segoe UI",system-ui,sans-serif;
     letter-spacing:-0.012em;
     text-shadow:0 2px 18px rgba(0,0,0,.85),0 0 3px rgba(0,0,0,.6)}}
  b{{font-weight:600;color:#ffffff}}
  m{{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
     font-size:33px;font-weight:500;color:#7fb0ff}}
</style></head>
<body><div class="band"><p>{text}</p></div></body></html>"""


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM, headless=True)
        pg = b.new_page(viewport={"width": W, "height": H},
                        device_scale_factor=2)
        for cid, start, end, text in CAPTIONS:
            pg.set_content(PAGE.format(w=W, h=H, text=text))
            pg.wait_for_timeout(450)          # let the webfont land
            path = out / f"{cid}.png"
            pg.screenshot(path=str(path), omit_background=True)
            manifest.append({"id": cid, "start": start, "end": end,
                             "png": str(path), "text": text})
            print(f"  {cid}  {start:5.2f}-{end:5.2f}s  {text}")
        b.close()
    (out / "captions.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
