# 2026-09-23 — Claude — Revisit detector and gap buckets (issue #10)

## What

- `src/distance_decayed_memory/data/revisit_detector.py` labels every frame
  from poses alone: `novel`, `recent`, or `revisit` with a gap and a source
  frame, plus `visit_age` (how long the current view has been on screen). It
  also maps gaps to power-of-two buckets, `[16,32)` through `[2048,4096)`.
- `scripts/data/spot_check_revisits.py` writes a JSON report and a picture
  grid of matched pairs per bucket, with near misses for contrast.

## Why

Evaluation must not rely on the script's own bookkeeping. Natural revisits
(walking back down a corridor) count too, and every frame needs a gap to be
scored in the headline plot.

## What I learned

- The plan's wording would turn a 20-frame pause into a "16-frame revisit"
  although the view never left the screen. The detector now requires the
  agent to leave the view first (D-009).
- Natural revisits are plentiful: 46% of frames in 20 local episodes, and the
  thin `[16,32)` bucket from issue #9 now has 377 frames.
- Pixel checks confirm the tolerances: matched pairs differ by a median 7–8
  (on 0–255) in every bucket, against 26 for random pairs. The sweep shows
  what tighter or looser tolerances would trade.

## How to verify

- `pytest` (84 tests, including a line-by-line reference implementation of
  the definition compared on random walks), `ruff check`, `black --check`.
- `python scripts/data/spot_check_revisits.py <episode dirs> --output-dir
  outputs/revisit-spot-check`, then open `spot_check.png`.

## Plain-language explanation

The detector looks at where the agent is and which way it faces. It marks
every moment when the agent is back looking at something it saw earlier, and
counts how long ago that was. It ignores moments when the agent never looked
away. The picture grid shows that those matched moments really do look the
same, whether the gap is 20 frames or 2,000.

## Next

Issue #11 (sharded writer and loader with node-local staging), then the
200-episode pilot (#18). No cluster jobs were run.
