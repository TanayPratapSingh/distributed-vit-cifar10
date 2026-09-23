"""Record the real dashboard in real Chromium.

Not a mockup and not a screen grab. Playwright drives the actual page served
from dashboard/index.html, so every pixel is the generated artifact. Screen
capture via avfoundation was ruled out because it would record the whole
desktop.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# Playwright's bundled build ships as "Google Chrome for Testing.app", not
# Chromium.app. Pointing at it directly avoids any version matching between
# the pip package and the cached browser revision.
CHROMIUM = ("/Users/tanay/Library/Caches/ms-playwright/chromium-1243/"
            "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/"
            "Google Chrome for Testing")
# Port is passed in. A previous run pointed at a port that another process
# had taken over, and a 200 status code did not catch it. The caller now
# verifies the served <title> before recording.
URL = f"http://127.0.0.1:{sys.argv[2]}/index.html"
W, H = 1280, 550


def glide(page, to: int, ms: int = 900, steps: int = 45) -> None:
    """Scroll with eased motion. Instant jumps read as cuts, not camera moves."""
    page.evaluate(
        """([to, ms, steps]) => new Promise(res => {
            const from = window.scrollY, d = to - from;
            let i = 0;
            const tick = () => {
                i++;
                const t = i / steps;
                const e = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
                window.scrollTo(0, from + d * e);
                if (i < steps) setTimeout(tick, ms / steps); else res();
            };
            tick();
        })""",
        [to, ms, steps],
    )


def main(out_dir: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True)
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=out_dir,
            record_video_size={"width": W, "height": H},
            color_scheme="dark",          # matches the terminal half
            device_scale_factor=2,        # retina text, downsampled on encode
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("tbody tr")

        # Anchor on the section headings rather than pixel offsets, so the
        # shot survives the page changing when new runs land.
        def top_of(text: str) -> int:
            return page.evaluate(
                """(t) => {
                    const h = Array.from(document.querySelectorAll('h2'))
                        .find(x => x.innerText.toLowerCase().includes(t));
                    return h ? Math.round(h.getBoundingClientRect().top + window.scrollY - 24) : 0;
                }""", text)

        page.wait_for_timeout(2200)                  # header and stat tiles
        glide(page, top_of("scaling"), 850)
        page.wait_for_timeout(1500)                  # table header in view
        glide(page, top_of("scaling") + 300, 900)
        # The table is held long enough for two captions, because the claims
        # they make are printed in these rows. A caption asserting a figure
        # that is not on screen is the thing this whole project is against.
        page.wait_for_timeout(5400)                  # the T4 rows
        glide(page, top_of("throughput"), 900)
        page.wait_for_timeout(2600)                  # throughput bars, outro

        ctx.close()
        browser.close()
    vids = sorted(Path(out_dir).glob("*.webm"))
    if not vids:
        print("no video produced", file=sys.stderr)
        return 1
    print(vids[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
