# brag plan: distributed-vit-cifar10

## Invocation

`/brag` with an explicit override: capture first, not composed. The Hyperframes
route was rejected on a previous attempt for producing a musical montage, so
every frame here is the software actually running.

tone: deadpan. format: landscape. duration target: 22s. music: none.
voice: off (no `--voice` in the invocation).

## Planning rubric

1. **What is it?** A distributed training pipeline for a vision transformer on
   CIFAR-10, built so its distributed systems claims are checkable.
2. **Who is it for?** Engineers who have read a scaling number and had no way
   to know whether it was real.
3. **What is the single strongest claim?** The project caught its own scaling
   loss and proved what caused it.
4. **What is the real UI?** A terminal running the test suite, and a dashboard
   generated only from run artifacts.
5. **What is the hook?** A speedup you cannot verify is a number, not a result.
6. **What is genuinely surprising?** Turning on mixed precision made a 2 GPU
   job scale worse, from 98 percent to 57 percent, on identical hardware.
7. **What is the punchline?** It was never the GPUs. It was a 4 core host.
8. **What must NOT be claimed?** Nothing measured on Kaggle may appear as if
   the laptop produced it. Terminal frames are local; T4 numbers are labelled.
9. **What would make it dishonest?** Recreating output, or letting a caption
   assert a number no artifact contains.

## Creative angle

Deadpan and evidential. No music, no swooshes. The video is a short argument
with the receipts on screen, and the tension is that the project's most
interesting result is a failure it found in itself.

## Honesty constraints (non negotiable)

* The terminal segment is a real PTY session recorded with asciinema. The
  tests genuinely run and genuinely pass.
* The dashboard segment is real Chromium driving the actual generated page.
* Every number shown comes from `runs/*.json`. 24 artifacts exist.
* Captions name the hardware for every figure. The T4 rows are captioned as
  2x Tesla T4 on Kaggle, because they were not produced by the machine in the
  terminal segment.
* No fixture data appears anywhere in this project, so no fixture caption is
  needed. If that changes, it gets an on screen caption.

## Storyboard

| # | t (s) | Source | On screen | Caption band |
|---|---|---|---|---|
| 1 | 0.0-3.2 | terminal | prompt, command typed at human pace | A speedup you cannot verify is not a result. |
| 2 | 3.2-9.5 | terminal | `pytest test_ddp_equivalence.py` runs, 2 passed | 2 ranks must produce the gradient of 1 process over the same batch. Proven, not assumed. |
| 3 | 9.5-12.5 | terminal | `python -m dvit.report` -> 24 artifacts | Every number on the dashboard traces to a run artifact. |
| 4 | 12.5-17.0 | browser | dashboard scaling table, T4 rows | 2x Tesla T4, Kaggle. fp32 scales at 98 percent. |
| 5 | 17.0-21.0 | browser | dashboard, amp rows in view | Turn on mixed precision and the same cluster drops to 57 percent. |
| 6 | 21.0-24.0 | browser | throughput chart | Remove the dataloader and it returns to 96 percent. It was never the GPUs. It was a 4 core host. |

Total 24.0s. Slightly over the 15-25 band's midpoint, justified because two
captions are full sentences and the readability law outranks the duration law.

## Layout

Canvas 1280x720. Content occupies the top 1280x550. A 170px band is reserved
at the bottom for captions so they never cover live output. This band exists
because a first pass buried the line the viewer most needed to read.

## Pipeline

1. `expect` drives a real interactive bash session with human paced typing.
2. `asciinema` records the PTY.
3. `agg` renders the cast to video.
4. Playwright drives real Chromium against the live dashboard on :8777.
5. Captions rendered to transparent PNGs through Chromium, because this
   ffmpeg build has neither freetype nor libass, so `drawtext` and the
   `subtitles` filter both fail.
6. Composited with `overlay` on time windows.
7. Caption timings sampled from the finished video at 1fps and read back,
   because `agg` compresses idle gaps and the script's intended waits do not
   match the rendered timeline.
8. Best settled frame baked as frame 0, since players ignore embedded cover
   art and grab the first frame regardless.

Screen capture via `ffmpeg -f avfoundation` was ruled out: it would record the
whole desktop, including this conversation.
