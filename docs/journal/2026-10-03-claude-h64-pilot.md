# 2026-10-03 — Claude — The H 64 pilot: first evidence for distance decay

## What

The owner proposed keeping every memory policy within the 64 frames A1 was
trained on, with the `full` oracle seeing all 64. Step 1 checked that perfect
memory helps recall at that range; step 2 trained and evaluated all policies.

- Step 1 (jobs `69462`, `69463`): A1 with `full` at H 64 vs `window`.
- Step 2: 11 A2 runs, 3,000 steps each (`configs/track_a/a2_pilot_h64/`),
  then P1 on the first 100 val episodes (jobs `69531`–`69541`).
- New: `scripts/eval/compare_gaps.py` (paired per-gap comparison against a
  reference run, bootstrap over episodes); `ModelCache.from_policies` builds
  the batched cache directly from blocks (one copy fewer, bit-identical),
  which let `full` train with 8 streams.

## Why

The long-horizon pilot showed the model could not use memory beyond its
64-frame training context, so it could not test the hypothesis. Within that
context it can.

## Results (LPIPS, lower is better; revisits by frames since last seen)

| Policy | B-mid 32–47 | B-mid 48–63 | B-low 32–47 | B-low 48–63 |
|---|---|---|---|---|
| `window` | 0.174 | 0.175 | 0.222 | 0.217 |
| `window_sink` | 0.189 | 0.187 | 0.235 | 0.241 |
| `uniform_subsample` | 0.187 | 0.155 | 0.226 | 0.203 |
| `relic_discrete` | 0.172 | 0.151 | 0.200 | 0.189 |
| **`decay_continuous`** | **0.159** | **0.148** | **0.176** | **0.153** |
| `full` (trained oracle) | 0.155 | 0.146 | 0.155 | 0.146 |

- `decay_continuous` beats `relic_discrete` and `uniform_subsample` in nearly
  every bucket at both budgets; at B-low the difference is significant
  everywhere (for example −0.035 against relic at 48–63).
- At B-mid it is statistically indistinguishable from the trained oracle,
  with a quarter of its tokens (plus the shared window).
- It is also the best budgeted policy on novel views, so part of the gain is
  better general context, not only recall.
- `window_sink` is worst: its sink frame lies far beyond the horizon.

## Caveats

One seed; val, not test; 100 episodes. Only short revisits (a few seconds).
`decay_continuous` used a decay length of 16 frames chosen in advance;
`relic_discrete` and `window_sink` ran with default knobs, so the plan's
fairness tuning has not been done. The automatic window-cliff check is
ill-conditioned at H 64 (it divides by a near-zero span beyond the horizon).

## How to verify

```
python scripts/eval/compare_gaps.py outputs/eval-h64/eval-a2-h64-*-bmid-*/ \
  outputs/eval-h64/eval-a2-h64-full-*/ \
  --reference outputs/eval-h64/eval-a2-h64-window-bmid-*/
```
after fetching the runs listed in `docs/EXPERIMENTS.md`.

## Plain-language explanation

When every memory gets the same small storage, the one that keeps every past
frame but blurs it more the older it is did best: about as well as perfect
memory at the medium budget, and clearly better than keeping a few old frames
sharp. This is the project's hypothesis, seen for the first time, but only
for memories a few seconds old and on one training run. Repeating it with
more seeds, tuned competitors, and the test set would make it a result.

## Next

Owner decision: run the full A3 design at H 64, and/or extend A1 to longer
clips to test the long-horizon claim.
