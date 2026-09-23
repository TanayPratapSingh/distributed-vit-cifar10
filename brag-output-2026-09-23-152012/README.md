# brag output

brag.mp4 is 23.47s, 1280x720, 30fps. brag.jpg is the poster, and it is also
baked as frame 0 of the video, because players and platforms ignore embedded
cover art and take the first frame regardless.

## Why this is capture, not composition

The /brag skill routes through Hyperframes, which builds an HTML composition
and renders it. That produces a montage: every frame is drawn, not observed.
For a project whose entire argument is "measure it, do not assume it", a
recreated terminal would undercut the thing being claimed.

So every frame here is the software actually running.

| Half | How |
|---|---|
| Terminal | expect drives a real interactive bash session with human paced typing. asciinema records the PTY. agg renders the cast. |
| Dashboard | Playwright drives real Chromium against the page served from dashboard/. |

Screen capture through ffmpeg avfoundation was ruled out. It would have
recorded the whole desktop.

## Layout

1280x720. Content occupies the top 1280x550. A 170px band is reserved
underneath for captions so they never cover live output.

## Captions

This ffmpeg build has neither freetype nor libass, so drawtext and the
subtitles filter both fail. Each caption is rendered to a transparent PNG
through Chromium and composited with the overlay filter on a time window.
That is not only a workaround: it buys real typography, and identifiers like
fp32 and amp are set in monospace so they read as code.

Caption timings were sampled from the rendered video, not taken from the
capture script's intended waits. agg compresses idle gaps, and the two
timelines disagree badly: the pytest command is typed at 4.27s in the
recording and at 1.72s in the render.

## What the camera caught

Pointing a recorder at the dashboard surfaced three real defects that had been
sitting on the page unnoticed. All three are fixed, committed and pushed:

1. Rank speedup divided a real data run by a synthetic data baseline, printing
   1.11x where the matched comparison is 1.15x.
2. The headline stat read 7,580 img/s, which is a synthetic input run. It now
   reads 4,706, the fastest real run.
3. The throughput chart did not mark synthetic runs, so its tallest bar read
   as the best real throughput in the project.

## Honesty

Caption text is verified by the same gate the docs are:

    python -m dvit.verify_claims --captions captions/captions.json

It fails if a number carrying a measurement unit does not round to some
artifact value, or to a ratio between two artifact values on the same
hardware.


No fixture data exists in this project, so no fixture caption was needed.
Every figure on screen comes from runs/*.json, and the caption naming the
hardware appears before any T4 number, because the terminal half was recorded
on the laptop and the T4 rows were not.

Every numeric caption is also printed on screen in the frame it appears over.
The first cut failed this: caption 6 claimed 96 percent while the page had
already scrolled past the row containing it. The table hold was extended so
captions 5 and 6 both sit on the evidence.

## Files

    brag.mp4                final, poster baked as frame 0
    brag.jpg                poster
    brag-plan.md            storyboard and planning rubric
    share-copy.txt          post copy
    capture/session.exp     the expect script that drove the shell
    capture/terminal.cast   the raw asciinema recording
    capture/record_dashboard.py   the Playwright recorder
    captions/*.png          caption plates
    captions/captions.json  caption text and sampled timings
    composition/            base and captioned intermediates
