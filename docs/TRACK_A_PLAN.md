# Track A — Detailed Plan (Memory Maze testbed)

**Status:** active. Owner decision 2026-09-22: Track A only; Track B deferred.
**Parent doc:** `docs/PROJECT_PLAN.md` (process rules, git, HPC skills, communication all still apply).
**Goal of this file:** give an agent everything needed to go from an empty repo to the headline figure, without guessing.

Items marked *(verify)* are assumptions the first agent to touch them must check and then correct in this file via PR.

---

## 1. What Track A must produce

One figure and one table decide the project's next step.

- **Headline figure:** revisit error (LPIPS, plus PSNR) vs. **revisit gap** (log-scale buckets of frames since the place was last seen). There is one curve per memory policy, all at the same cache budget, with 95% confidence bands over seeds.
- **Headline table:** the same results averaged over gap ranges (short, medium, long), at three budgets, plus non-revisit quality. The non-revisit column shows that no policy wins memory by sacrificing ordinary prediction.

Together these answer H1 (continuous vs. discrete), H2 (graded vs. window + sinks), and, through ablations, H3 (revisit training is necessary) and H4 (content-aware vs. pure distance) at small scale. That is enough for a workshop paper on its own.

---

## 2. Why Memory Maze

Memory Maze is a family of randomly generated 3D mazes (DeepMind, 2022) built specifically to test long-term memory.

- **Low resolution (64×64)**, so models are small and runs are fast.
- **A new random layout every episode.** The model can't memorize the mazes. On a return, the only way to know what a corridor looks like is to have remembered it from earlier in the same episode.
- **Known agent position and heading.** The verified global-observation variant
  exposes `maze_layout`, `agent_pos`, and the unit heading vector `agent_dir`,
  along with the image and target fields. That gives exact revisit detection
  and exact ground truth.
- **Unlimited data.** We generate our own trajectories, so we control the revisit gaps precisely.

Minecraft is the optional second environment (§11, milestone A5), only after the Memory Maze results are in.

---

## 3. Data generation

### 3.1 Environment setup

- Verified stack: `memory-maze==1.0.3`, `dm-control==1.0.47`,
  `mujoco==3.14.0`, and `gym==0.26.2` on Python 3.11.16.
- Headless CPU rendering on **`main`** uses EGL with
  `LIBGL_ALWAYS_SOFTWARE=1`. Job `68321` confirmed Mesa llvmpipe rather than a
  GPU. The CPU nodes do not provide OSMesa or Xvfb, and current conda-forge
  `mesalib` does not include `libOSMesa`, so the planned OSMesa path is not
  available. The one-core 64×64 benchmark measured a median **23.85 frames/s**
  over three 2,000-frame repeats (range 23.43–23.97).
- Start with the **9×9** maze size, then move to 15×15 for harder, longer-range runs.
- Action space: six verified discrete actions in order: no-op, forward, left,
  right, forward+left, and forward+right. Actions are inputs, not an observation
  key; record the integer action index per frame. Convert `agent_dir` to a
  scalar heading with `atan2(dir_y, dir_x)` when writing `pose.npy`.

### 3.2 Scripted revisit trajectories

The generator produces episodes in which **the agent deliberately returns to earlier places after controlled delays**.

1. Read the maze layout. Pick a random start.
2. **Explore** with a coverage-seeking policy (for example, go to the nearest unvisited cell using A* on the grid) for a random duration.
3. For each **revisit event**: choose an earlier *anchor* (a pose recorded at time `t_a`), sample a target gap `g` **log-uniformly in [16, 2048] frames**, keep exploring or wandering until roughly `t_a + g`, then navigate back to the anchor cell and turn to the anchor heading. Record the event `(t_a, t_return, cell, heading)`.
4. Mix in natural behaviour: random pauses, detours, partial returns, and episodes with **no scripted revisit** (about 30%). This stops the model from learning "a return always comes after X".
5. Navigation uses A* over grid cells plus a simple turn-then-move controller on the discrete actions. Add small action noise (about 5%) so trajectories aren't perfectly robotic.

**Implemented (issue #9).** `src/distance_decayed_memory/data/navigation.py`
(A*, closed-loop controller), `revisit_script.py` (the planner), and
`scripts/data/generate_revisit_episodes.py` (writer plus previews). Measured
Memory Maze dynamics at 4 Hz: one turn command from rest rotates about 17–18°
once momentum decays, and forward motion reaches 0.25 cell/step after about
three steps, coasting about 0.18 cell after release. The controller therefore
aligns heading to within 10° (15° fallback after six alignment turns). In a
20-episode local check (seeds 0–19, 40,960 frames), 81 returns completed, all within
0.20 cell and 7.9° of the anchor pose, with a median arrival 3 frames after
the target. Realized gaps covered 16–2,047 frames, but the `[16, 32)` bucket
was thin (3 of 81); the A0 pilot must confirm per-bucket coverage, counting
natural revisits from the detector as well. Conventions are in `D-008`.

Target episode length: **2,048 frames** for training and **4,096 frames** for evaluation episodes. The longer evaluation episodes test whether memory generalizes past the training length.

### 3.3 Revisit detection for evaluation (independent of the script)

A frame at time `t` counts as a **revisit with gap `g`** if there is an earlier frame `t' < t − w_min` with position within 0.3 cells and heading within 15° *(tune thresholds in the pilot using visual spot-checks)*, where `g = t − t'` uses the most recent such `t'`. Revisits found naturally, not only scripted ones, are counted too. Frames with no match are **novel views**, which are the control condition.

Gap buckets (frames): `[16,32), [32,64), … , [2048,4096)`. These are powers of two, which gives 8 buckets.

### 3.4 Storage format

Per episode, write to `/mnt/hpc/tmp/$USER/dd-memory/data/memmaze9/<split>/<episode_id>/`:

- `frames.npy`: uint8 `[T, 64, 64, 3]`, about 12 KB per frame, so about 25 MB per 2,048-frame episode.
- `actions.npy`: int8 `[T]`.
- `pose.npy`: float32 `[T, 3]` (x, y, heading).
- `meta.json`: seed, maze size, scripted revisit events, generator git SHA.

Group episodes into shards of about 256 per shard for fast loading, and stage the shards on node-local `/tmp` at job start.

| Split | Episodes | Approx size | Purpose |
|---|---|---|---|
| pilot | 200 | 5 GB | debug and visual checks |
| train | 20,000 | ~500 GB | A1 and A2 training |
| val | 500 | ~12 GB | model selection and early stopping |
| test | 1,000 (4,096 frames) | ~50 GB | **frozen**; hash recorded in repo; touched only for final numbers |

Splits use disjoint environment seeds. Record `sha256` of the test manifest in `docs/DATASETS.md` before any final evaluation.

---

## 4. Model

A frame-causal **diffusion transformer** working directly on pixels. No VAE is needed at 64×64.

- **Tokens.** 64×64×3 frames, 4×4 patches, **256 tokens per frame**. 3D RoPE over (t, x, y).
- **Chunks.** The model generates **4 frames per chunk**. Tokens inside a chunk attend to each other bidirectionally. Across chunks, attention is causal.
- **Conditioning.** Action embedding plus diffusion-time embedding, added per frame via adaLN.
- **Objective.** Flow matching (velocity prediction), with independent noise levels per chunk (Diffusion-Forcing style). Sampling uses 16–32 Euler steps (tune for speed vs. quality at A1).
- **Size ladder** (pick in A1 by a quick scaling check):

| Name | Layers | Width | Heads | Params (approx) |
|---|---|---|---|---|
| S | 12 | 512 | 8 | ~40M |
| M | 16 | 768 | 12 | ~115M |
| L | 24 | 1024 | 16 | ~300M |

**Default: M.** Go to L only if M is clearly under-fitting *and* the sweep still fits in the compute budget (§9).

A useful fact: for M, a full-fidelity cache of a whole 4,096-frame episode is about 1M tokens × 16 layers × 2 × 768 × 2 bytes, roughly 50 GB. That **fits on one 96 GB card**, so the **full-cache oracle is runnable at every horizon in Track A.** The budget constraint in Track A is imposed deliberately to mirror what the large model would face (§6), and the oracle gives a true upper bound.

---

## 5. Training: two stages

### Stage A1 — base model (one run, shared by every policy)

- Short-context training on 64-frame windows with **full** attention over the window (block-causal mask). This teaches the model how the maze looks and moves: rendering and dynamics.
- Uses DDP on 2 pro6000 cards, the most one job may hold (§9).
- Stop when validation loss plateaus. Pick the checkpoint by validation loss, plus a visual check of 64-frame rollouts.

This checkpoint is **frozen as the common starting point**. Every policy in A2 starts from exactly the same weights, which keeps the comparison fair and saves a lot of compute.

### Stage A2 — memory fine-tuning (one run per policy × budget × seed)

A2 teaches the model to read its compressed memory. It uses **streaming training**, which reproduces inference exactly:

1. Hold `N` episodes in parallel (the batch). Each has its own live cache managed by the policy under test.
2. At each optimizer step, advance every episode by one chunk. The model gets the **noisy** next chunk plus the action sequence, attends to the cache, and receives the flow-matching loss against the true chunk.
3. Then run one **clean** pass on the true chunk to write its K/V into the cache, and call `policy.compact(t)`.
4. When an episode ends, replace it with a new one. Episodes start staggered, so every batch mixes short and long histories.
5. **Gradient window.** K/V from the last `k = 2` chunks stay in the autograd graph. Older cache entries are detached. This lets the model learn to *write* useful memory for the near future while keeping memory use bounded.

Why streaming rather than one big teacher-forced pass: compressing K/V properly needs the K/V of every past token in every layer. A single pass over a 2,048-frame history (about 520k tokens) per training example is far more expensive. Streaming produces exactly the cache the model will see at test time, at a cost linear in episode length.

Known issue: **stale cache entries.** K/V written hundreds of steps ago came from slightly older weights. Mitigations: A2 starts from a converged A1 checkpoint and uses a low learning rate (about 0.1–0.3× A1's). Optionally, refresh the cache by restarting episodes on a fixed schedule. Log a staleness diagnostic: the difference between cached K/V and freshly recomputed K/V on a held-out episode, every N steps.

Other A2 details:
- Teacher-forced frames get light noise augmentation (a small random noise level on the clean pass) so the model tolerates its own imperfect frames at test time.
- Optional loss up-weighting on scripted revisit frames. Test it as an ablation, off by default.
- A2 length: fixed number of optimizer steps, identical for every policy. Set it after the A2 pilot (§11).

---

## 6. Budgets and horizon

Horizon `H = 2,048` frames for training. Evaluation runs out to 4,096 frames: policies are still capped at `H`, and the 2,048–4,096 gap buckets test what happens past the horizon.

Budgets are chosen as a **fraction of horizon tokens**, to match what the large model would face. For a 1.3B model with a ~25k-token budget and a 10-minute horizon, the fraction is about 0.7%.

| Budget | Frame-equivalents | Tokens | Fraction of H·256 |
|---|---|---|---|
| B-low | 8 | 2,048 | 0.4% |
| B-mid (default) | 16 | 4,096 | 0.8% |
| B-high | 48 | 12,288 | 2.3% |

The policies are exactly those in `PROJECT_PLAN.md` §4.1. For each policy × budget, the `stats()` output (tokens per distance bucket) is logged and checked against the budget with an assertion at every compaction.

---

## 7. Evaluation protocol

Two protocols, both run on the frozen test split.

**P1 — Observe then predict (the headline memory test).** The cache is built from **true** frames up to a point shortly before a scripted return (`t_return − 16`). The model then generates the next 32 frames from actions only, and we score them against ground truth. This isolates memory retrieval from compounding generation errors.

**P2 — Full autoregressive rollout.** Given only the first 16 true frames and the action sequence, the model generates all 4,096 frames. Score every revisit and novel frame. This measures memory together with drift, and is closer to real use.

Metrics per frame: LPIPS (primary), PSNR, SSIM. Aggregate by gap bucket and by condition (revisit vs. novel). Also report:

- **Memory gain** = error(`window` policy) − error(policy), per bucket. It reads as "how much better than forgetting".
- **Oracle gap** = error(policy) − error(`full`), per bucket. It reads as "how much worse than perfect memory".
- Efficiency: cache bytes, attention FLOPs per chunk, and wall-clock time per generated frame.

Statistics: 3 seeds per configuration (A2 seeds; A1 is shared). Report bootstrap 95% confidence intervals over episodes × seeds, and paired comparisons on the same episodes when comparing two policies.

---

## 8. Experiment matrix

### A3 — main sweep

| Policy | B-low | B-mid | B-high |
|---|---|---|---|
| `window` | 3 seeds | 3 | 3 |
| `window_sink` | 3 | 3 | 3 |
| `relic_discrete` | 3 | 3 | 3 |
| `uniform_subsample` | 3 | 3 | 3 |
| `decay_continuous` (exp, τ tuned on val) | 3 | 3 | 3 |
| `full` (oracle, no budget) | 3 seeds, one column | | |

That is 5 × 3 × 3 + 3 = **48 A2 runs**.

Fairness rule: `decay_continuous` gets its τ (decay length) chosen on **val** with a small sweep (τ in {16, 64, 256} frames). The same number of tuning runs goes to the knobs of `relic_discrete` (the window size and where the level boundaries fall) and `window_sink` (the number of sinks).

### A4 — ablations (at B-mid, 2 seeds each)

| Question | Variants |
|---|---|
| Decay shape | exponential, power law, linear, log-steps |
| Cell type | spatial-only pooling, temporal-only merging, spacetime cells |
| Positional encoding | RoPE at cell centroid vs. averaged post-RoPE keys |
| Proportional attention | with vs. without `log v` bias |
| **H3: revisit training** | A2 on data with vs. without scripted revisits (for `window_sink`, `relic_discrete`, `decay_continuous`) |
| **H4: content-aware** | `decay_content` vs. `decay_continuous` |
| Revisit loss weighting | off vs. 2× on revisit frames |

That is roughly 30–35 more runs.

---

## 9. Compute plan

Every number below is an **estimate to replace with measurements** after the
A0/A1 pilots.

### 9.1 The GPU budget (measured, 2026-09-22/23)

Slurm caps this project at **2 pro6000 GPUs per job** and **4 pro6000 GPUs
concurrently per user** (two jobs on `mixed`, two on `gpu`). Seven cards exist;
we may use four at a time. Evidence and the exact QOS limits are in
[docs/hpc/ddp-and-requeue.md](hpc/ddp-and-requeue.md).

Consequences that shape this plan:

- **A1 is a 2-GPU DDP run,** not a 4-7 GPU run.
- **A2 and evaluation are throughput problems, not latency problems.** Each run
  is single-GPU, and four run side by side.
- **The sweep is queue-bound.** About 80-90 runs cannot be launched at once, so
  they need a runner that keeps four slots full and resumes interrupted runs
  (checkpoint/requeue is validated).
- **Data generation is unaffected:** it is CPU work on `main`.

### 9.2 Estimated cost

- **Data generation:** CPU job arrays on `main`. 20k episodes × 2,048 frames is
  about 41M frames. The A0 environment-only measurement is **23.85 frames/s per
  core**, or about 480 core-hours before navigation and I/O. At ideal 32-core
  scaling that is roughly 15 hours; measure end-to-end speed and scaling with
  the 200-episode pilot before scheduling the full split.
- **A1:** 1 run, M model, about 1-2 days on 7 cards in the original estimate;
  on 2 cards assume **3-5 days**, and re-estimate from the A0/A1 pilot.
- **A2 runs:** single-GPU jobs. At an assumed 8-16 GPU-hours each, about 80 runs
  is roughly 650-1,300 GPU-hours. With **4 cards in parallel that is about
  7-14 days** of wall clock.
- **Evaluation:** P2 rollouts of 4,096 frames are the expensive part. Batch many
  episodes per GPU. Budget about 20% of A2 compute, so **8-17 days** for A2 plus
  evaluation together.

### 9.3 Staging, because the budget is now the binding constraint

Run the sweep in priority order rather than all at once, so that stopping early
still yields a publishable result:

1. **Core (protected):** B-mid column, 3 seeds, every policy, plus the `full`
   oracle and the fairness tuning. This alone answers the headline question.
2. **Second budget column:** B-low, for the budget-sensitivity claim.
3. **A4 ablations** at B-mid.
4. **B-high column** and `uniform_subsample`.

If this runs over budget, cut from the bottom of that list. **Never** cut: the
`full` oracle, 3 seeds on the B-mid column, or the fairness tuning.

Before launching the sweep, replace the per-run estimate with a measured one
from the A2 pilot and recompute this section. If a measured run costs much more
than 16 GPU-hours, cut the A2 step count or drop the model from M to S rather
than dropping seeds.

---

## 10. Sanity checks (all must pass before A3 results are trusted)

These are cheap and catch the bugs that would otherwise produce a fake result.

1. **Window cliff.** With `window` of `w` frames, revisit error must be close to `full` for gaps `< w` and jump to about novel-view error for gaps `> w`. If there's no cliff, revisit detection or evaluation is broken.
2. **Oracle is best.** `full` should match or beat every budgeted policy in every bucket (within noise). A budgeted policy clearly beating the oracle means there's a bug to find, not a result.
3. **Equivalence.** `decay_continuous` with a budget large enough to keep everything at full detail must reproduce `full` numerically (unit test plus one eval run).
4. **Budget assertion** never fires, and the logged `stats()` totals are identical across policies.
5. **Memory is actually used.** Shuffling or zeroing cache entries older than the window must hurt revisit error but leave novel-view error roughly unchanged.
6. **Streaming = inference.** Cache contents during A2 training and during evaluation, for the same true frames, match up to noise augmentation.

---

## 11. Milestones and gates

| ID | Milestone | Done when | Owner check-in |
|---|---|---|---|
| A0 | Environment + generator pilot | 200 pilot episodes; revisit detector validated on a grid of visual spot-checks; generation speed measured | Journal entry with example frames of a revisit pair |
| A1 | Base model | Size chosen (S/M/L); converged checkpoint; 64-frame rollouts look right | Journal with rollout GIFs and loss curves |
| A2-pilot | Streaming trainer + policies | Sanity checks 1–6 pass on a short run; cost per A2 run measured; A2 step count fixed | **Gate: owner approves the sweep before launch** |
| A3 | Main sweep | 48 runs done; headline figure and table drafted | **Gate 2 from the master plan: decide next steps together** |
| A4 | Ablations | Ablation table | Journal |
| A5 | *(optional)* Minecraft replication | Key comparison (3 policies, B-mid) reproduces or not | Journal |
| A6 | Workshop draft | `paper/` compiles with all figures generated by scripts | Owner review |

---

## 12. Issue list to open now (replaces master plan Appendix B items 7 and 10)

Infrastructure (from the master plan, still first):
1. `infra`: repo scaffold, CI, pre-commit, gitignore, `AGENTS.md`, `CLAUDE.md`.
2. `infra`: HPC scripts and skills.
3. `infra`: verify the pro6000 layout, wall time, and compute-node internet; update the HPC doc.
4. `infra`: conda env in scratch; Blackwell kernel check (SDPA, FlexAttention, bf16).
5. `infra`: DDP smoke test; checkpoint, resume, and requeue test.
6. `paper`: literature refresh.

Track A:
7. `track-a`: install `memory-maze` headless on `main`; measure render speed; confirm the observation keys for layout and pose.
8. `track-a`: scripted revisit trajectory generator (A* navigation, log-uniform gaps, anti-shortcut mixing) + unit tests on toy grids.
9. `track-a`: revisit detector + gap bucketing + visual spot-check script.
10. `track-a`: dataset writer (sharded) + loader with node-local staging; generate the pilot split.
11. `phase-1`: `MemoryPolicy` library (all policies) + tests, as in the master plan Phase 1.
12. `track-a`: pixel DiT model (patchify, 3D RoPE, adaLN actions, block-causal attention) + flow-matching loss + sampler.
13. `track-a`: A1 trainer (DDP, auto-resume) + short scaling check S/M/L.
14. `track-a`: A2 streaming trainer (parallel episodes, live caches, `k`-chunk gradient window, staleness diagnostic).
15. `track-a`: evaluation harness P1/P2, metrics, bootstrap CIs, headline plotting script.
16. `track-a`: sanity-check suite (§10) as runnable scripts.
17. `track-a`: generate the full train/val/test splits; freeze and hash test.

Dependencies: 7 → 8 → 9 → 10 → 17. 11 and 12 can run in parallel with 7–10. 13 needs 10 + 12. 14 needs 11 + 13. 15 and 16 need 14.

---

## 13. Decisions recorded by this plan

Copy these into `docs/DECISIONS.md` as the first entries.

- **D-001** Track A only; Track B deferred until the owner reopens it after Gate 2.
- **D-002** Memory Maze 9×9 first. Pixels at 64×64 with 4×4 patches, no VAE.
- **D-003** Two-stage training: a shared A1 base, then one A2 memory fine-tune per policy from the same checkpoint.
- **D-004** A2 uses streaming training with live caches and a 2-chunk gradient window, chosen over single-pass teacher forcing (exact train/test match, linear cost).
- **D-005** Budgets are defined as fractions of horizon tokens (0.4 / 0.8 / 2.3%) to mirror large-model conditions. The full-cache oracle is always included.
- **D-006** Equal tuning effort for every policy. Test split frozen and hashed before final evaluation.
- **D-007** Plan around the measured Slurm ceiling: 2 pro6000 GPUs per job, 4 concurrent per user. A1 is a 2-GPU DDP run; the A2 sweep is staged and queued four at a time (§9).
