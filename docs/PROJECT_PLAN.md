# Project Plan — Distance-Decayed Memory for Autoregressive Video Prediction

**Audience:** the Codex / Claude agents working on this repo, and the project owner.
**Status:** v1.1 planning document. Agents: treat this as the source of truth for scope and process. If you need to change it, open a PR against this file and explain why.

> **Owner decision (2026-09-22): Track A only for now.** Track B (§6, Phase 3–4 scaled work) is **deferred**. Do not download Wan checkpoints, build Track B data, or spend GPU time on Track B until the owner reopens it after Gate 2. The detailed, authoritative spec for the current work is **`docs/TRACK_A_PLAN.md`**; where it is more specific than this file, it wins. Phase 0 items that exist only for Track B (base-checkpoint evaluation, Wan KV sizing) are dropped from Phase 0.

Starting files in the repo: `README.md`, `PROJECT_PLAN.md` (this file) and `TRACK_A_PLAN.md`.

The original design proposal and the Bowdoin HPC reference are **embedded in this file** as Appendix C and Appendix D. **First bootstrap task:** move `PROJECT_PLAN.md` and `TRACK_A_PLAN.md` into `docs/`, extract Appendix C to `docs/proposal.md` and Appendix D to `docs/hpc/bowdoin-hpc.md` (verbatim), then replace the two appendices here with links to those files. Every other reference in these plans uses the `docs/...` paths.

---

## 0. TL;DR

We want to test one claim: **inside a fixed memory budget, a causal video model remembers the past better if its cache compresses old frames on a smooth curve by distance, instead of keeping a sliding window, a window plus anchors, or a RELIC-style fixed discrete schedule.**

We test it on two tracks.

- **Track A is a controlled testbed.** We train small action-conditioned video models from scratch on environments where we have ground truth for every revisit (Memory Maze first, Minecraft second). This is cheap, fast, and statistically clean, and it is where most of the science happens.
- **Track B is the scaled demonstration.** We fine-tune an open causal video model built on Wan2.1-T2V-1.3B (Self-Forcing / LongLive family checkpoints) with each memory policy and test at the minute scale. This is the "as powerful as we can" model.

With 7 × RTX PRO 6000 Blackwell (96 GB), the realistic ceiling is **fine-tuning a roughly 1–5B causal video model and training roughly 100–500M models from scratch**. Pretraining a video foundation model from scratch is out of scope. The paper's contribution is the memory mechanism plus a revisit-consistency benchmark, not a new foundation model.

---

## 1. Hypotheses

The paper is framed around these hypotheses. Every experiment should map to one of them.

| ID | Hypothesis | Primary evidence |
|---|---|---|
| **H1** | At equal cache token budget, continuous distance decay beats a RELIC-style discrete schedule on revisit consistency. | Track A revisit error vs. revisit gap; Track B identity consistency |
| **H2** | Graded compression beats a hard window plus sinks (LongLive-style) at equal budget, and the gap grows with horizon. | Same metrics, stratified by horizon |
| **H3** | The benefit requires training pressure. A model fine-tuned without revisit augmentation does not use compressed distant memory, whatever the cache policy is. | Ablation: revisit augmentation on vs. off |
| **H4** | Pure distance decay is beaten by a content-aware allocation at the same budget. This is a risk from the proposal, stated as a hypothesis so we test it instead of hiding it. | Content-aware variant vs. pure decay |
| **H5** | Memory and compute stay flat with video length, and quality degrades more slowly than with a sliding window. | Efficiency benchmarks, drift curves |

A negative result on H1 is still publishable if the benchmark and analysis are solid, for example "smooth decay ≈ discrete schedule; what matters is budget allocation and training". Agents must record honest results and must not tune only the proposed method.

---

## 2. Why the budget matters (back-of-envelope)

Agents: verify these numbers against the actual model config in Phase 0 and update this table.

Wan2.1-T2V-1.3B at 480×832 has 30 layers and hidden size 1536. The VAE downsamples 8× spatially and 4× temporally, and the patch size is 2×2. That gives about **1,560 tokens per latent frame** and about **4 latent frames per second** at 16 fps.

KV cache per token is roughly 30 layers × 2 (K,V) × 1536 × 2 bytes ≈ **0.18 MB**.

| Horizon | Latent frames | Full KV cache (approx.) |
|---|---|---|
| 5 s | 20 | ~5.8 GB |
| 1 min | 240 | ~69 GB |
| 4 min | 960 | ~276 GB |
| 10 min | 2,400 | ~690 GB |

A full cache stops fitting on one 96 GB card at around one minute. That is the whole motivation.

The design consequence is that **spatial pooling alone cannot reach 10 minutes.** Take a realistic budget of about 16 full-frame equivalents (~25k tokens, ~4.6 GB). If 4 frames stay at full fidelity, the remaining ~18.7k tokens have to cover about 2,400 frames, which is under 8 tokens per frame. So the design has to treat compression as **spacetime cells whose volume grows with distance**. Near `t`, a cell is one patch of one frame. Far back, a cell averages many patches across many frames. Spatial-only and temporal-only compression become special cases of one mechanism (see §4.2).

---

## 3. Scope

### In scope

1. A model-agnostic **memory policy library** (the core contribution), with unit tests.
2. **Track A:** small causal, action-conditioned video world models trained from scratch on Memory Maze and then Minecraft, with all memory policies compared at matched budgets.
3. **Track B:** fine-tuning an open causal Wan-based model with the top policies, plus minute-scale evaluation.
4. A **revisit-consistency benchmark** with metrics stratified by revisit gap. This is also a reusable contribution.
5. Efficiency benchmarks covering memory, throughput, and latency vs. video length.
6. A paper draft.

### Out of scope unless Phase 3 goes well

- Pretraining any video model from scratch at more than about 1B parameters.
- The full 10-minute horizon in Track B as a guaranteed deliverable. It is a stretch goal. The minimum bar is clear gains at 2–5 minutes.
- New VAEs, new samplers, or text encoders.
- Real-time deployment work beyond measuring throughput.

---

## 4. Method design (what gets built)

### 4.1 The `MemoryPolicy` interface

Every policy implements the same interface so that the model code never changes between experiments.

```python
class MemoryPolicy(Protocol):
    budget_tokens: int                     # hard cap, identical across compared policies
    def append(self, k, v, pos, t): ...     # add newly generated frame(s) at time t
    def compact(self, t): ...               # re-compress aged entries; enforce budget and horizon
    def kv(self) -> tuple[K, V, pos, weight]: ...  # tensors for attention; weight = cell volume
    def stats(self) -> dict: ...            # tokens per distance bucket, memory bytes
```

Policies to implement:

| Policy | Description |
|---|---|
| `full` | Exact cache with no eviction. An oracle, only feasible at short horizons. |
| `window` | Last `w` frames at full fidelity. |
| `window_sink` | LongLive-style. Last `w` frames plus the first `s` frames as sinks. |
| `relic_discrete` | Rolling window plus a remainder at fixed 1×/2×/4× spatial downsampling, following the RELIC paper's schedule, extended to our horizon at equal budget. |
| `uniform_subsample` | Keeps every k-th frame at full fidelity. A naive control. |
| `decay_continuous` | **The proposal.** Cell volume `v(d)` grows smoothly with distance `d`, with a hard cutoff at horizon `H`. |
| `decay_content` | Same budget, but cell allocation is modulated by a salience score (attention mass received, or feature novelty). Tests H4. |

### 4.2 The continuous decay policy, concretely

Define a token density `ρ(d)` in tokens per latent frame at distance `d` (in latent frames). Its integral over `[0, H]` equals the budget `B`. For each distance the cell volume is `v(d) = N_frame / ρ(d)`, the number of original patch-tokens that one cached token summarizes.

- **Full-fidelity window.** `v(d) = 1` for `d < w`.
- **Decay region.** `ρ(d)` follows a family we sweep: exponential `ρ0·exp(-(d-w)/τ)`, power law `ρ0·(d-w+1)^(-α)`, linear, and log-spaced steps (the last one is close to the discrete baseline, which makes it a useful bridge).
- **Spacetime cells.** While `v(d) ≤ N_frame`, pool spatially within a frame using an adaptive average pool to the nearest grid. Once `v(d) > N_frame`, also merge adjacent frames temporally. So a far-past cell spans several frames.
- **Aging.** As `t` advances, entries move to larger `d` and are re-pooled from their current cells. This is cheap because it is average pooling of already-pooled tokens. Compaction is amortized, running every `c` frames, not on every step.
- **Hard cutoff.** Anything with `d > H` is dropped.

Design decisions to make explicitly and record in `docs/DECISIONS.md`:

1. **What gets pooled.** Pool K and V directly, which is cheap. The alternatives are pooling hidden states and re-projecting, or learned pooling. Start with direct K/V average pooling. A small learned compressor is an ablation.
2. **Positional encoding.** Wan uses 3D RoPE. Store **pre-RoPE** keys and apply RoPE at the cell centroid (fractional coordinates) at attention time. The alternative is averaging post-RoPE keys. This is a real correctness issue and should be ablated.
3. **Proportional attention.** A pooled token stands in for `v` tokens, so add `log v` to its attention logits (as in Token Merging). Ablate it on and off.
4. **Temporal position growth.** Absolute time indices over 10 minutes go far past anything seen in training. Options are relative re-indexing (positions relative to `t`) or clamping. Investigate what Self-Forcing / LongLive do and document it.
5. **Budget matching.** Compare policies at equal **cached token count** and report equal attention FLOPs alongside. Log `stats()` for every run to prove the budgets match.

### 4.3 Attention kernel

Queries come from the current chunk and keys come from the cache, so a plain concatenated-KV `scaled_dot_product_attention` is sufficient, with an additive bias for `log v`. Use PyTorch **FlexAttention** if a block mask or score modification is needed. Verify on Blackwell (sm_120) which kernels actually work. FlashAttention-3 is Hopper-only and should not be assumed. FA2 and SDPA backends must be tested in Phase 0.

---

## 5. Datasets

Agents must check the license and download terms of every dataset before use and record them in `docs/DATASETS.md`. Anything marked *(verify)* has not been checked.

### Track A (controlled, with ground-truth revisits)

| Dataset | Why | Notes |
|---|---|---|
| **Memory Maze** (DeepMind, 2022) | **Primary.** 3D mazes built specifically to test long-term memory. Long episodes, low resolution (64×64), actions available, and agent position and orientation known, so revisits can be detected exactly. Offline datasets exist, and the environment can generate unlimited new trajectories, including scripted loops that force revisits. | Cheap enough to sweep many policies and seeds. *(verify dataset sizes and episode lengths)* |
| **Minecraft** (MineDojo / MineRL-generated trajectories, or VPT contractor data) | **Secondary.** Richer visuals and an established setting for memory in video world models (for example WorldMem). Scripted loop trajectories give ground-truth revisits. | Rendering our own trajectories gives control over revisit gaps. *(verify tooling works headless on the cluster; CPU rendering on `main` may be the way)* |
| **Habitat + HM3D** *(optional)* | Photorealistic indoor scenes with arbitrary camera paths and exact revisits. | Requires a license agreement for HM3D. Only use if Track A needs a realism step before Track B. |

### Track B (open-domain, long video)

| Dataset | Use |
|---|---|
| **Sekai** | Long first-person world-exploration videos with camera annotations. Good for long-horizon fine-tuning and loop-style revisits. *(verify availability/license)* |
| **SpatialVID** | Large video dataset with camera poses. Useful if we add camera conditioning. *(verify)* |
| **MiraData** | Long clips with structured captions, for long-duration text-to-video tuning. *(verify)* |
| **DL3DV-10K / RealEstate10K** | Scene walkthroughs with poses. Useful for loop-closure evaluation. |
| **Ego4D** *(optional)* | Very long egocentric video. Requires a signed agreement. |

**Revisit augmentation** (needed for H3) follows RELIC's lesson. Build training clips that come back to earlier content, either by time-reversing segments (A→B→A) or by selecting clips whose camera trajectories loop. Without this pressure the model has no reason to use distant memory.

---

## 6. Models

### Track A: small model from scratch

- Frame-level causal diffusion transformer (Diffusion-Forcing-style per-frame noise levels, or flow matching), conditioned on actions.
- Input is either raw 64×64 pixels with patch size 4 (256 tokens per frame) or a small pretrained image VAE. Start with pixels for simplicity.
- Size ladder: about 50M, 150M, and 400M parameters. Pick the largest that trains in under about 24 h on 1–2 GPUs, so that sweeps fit.
- **Compressed-context teacher forcing.** During training, build the memory from ground-truth past frames (with noise augmentation) using the policy under test, then predict the next chunk. This trains the model to read compressed memory without costly rollouts. Rollout-based fine-tuning is a later refinement.

### Track B: scaled model

- Start from an open causal checkpoint on **Wan2.1-T2V-1.3B**: **Self-Forcing** (causal distillation) and/or **LongLive** (long streaming tuning, window plus frame sinks, demonstrated at 240 s). Phase 0 decides which one based on code quality, license, and how cleanly its KV cache can be swapped out.
- Stretch option: a larger base such as Wan2.2-TI2V-5B, whose VAE compresses more and so needs fewer tokens per frame. Only if a causal variant is available and Phase 3 is on schedule.
- Fine-tune one copy per memory policy with the same recipe, data, steps, and seed schedule.
- The 1.3B model fits comfortably on a single 96 GB card, so use **DDP** across GPUs. FSDP is only needed for 5B or larger. The RTX PRO 6000 has no NVLink, so all communication goes over PCIe and possibly the network. Measure it in Phase 0 and prefer larger per-GPU batches or gradient accumulation to keep communication low.

---

## 7. Evaluation

### Track A metrics (ground truth available)

- **Revisit error vs. revisit gap.** For every generated frame whose agent pose matches a pose visited `g` steps earlier, compute PSNR, SSIM, and LPIPS against ground truth, then bucket by `g`. **This curve per policy is the headline figure of the paper.**
- Non-revisit prediction quality, to show that a policy is not buying memory by sacrificing short-term quality.
- Optionally, a probe task: decode the maze layout or object positions from the cache.

### Track B metrics (no pixel ground truth)

- **Re-entry identity consistency.** Use prompt-switching scenarios in the style of LongLive's interactive prompts. Segment 1 introduces a subject or scene, segment 2 moves away, segment 3 returns. Score with DINOv2 / CLIP feature similarity (and a face-embedding model for people) between segment 1 and segment 3. Also compare against a no-memory control.
- **Loop-closure consistency** if camera conditioning is added. A camera path returns to its start, and the start and return frames are compared.
- **Drift over time.** Per-segment quality over the video (VBench-style imaging and aesthetic quality, subject and background consistency) plotted against time.
- **Efficiency.** Peak GPU memory, frames per second, and per-frame latency vs. video length (30 s, 1, 2, 5, 10 min).

### Analysis

- **Attention mass vs. distance** in trained models, across policies. This directly checks the PackForcing risk from the proposal: does attention to distant content concentrate predictably or not?
- Decay-shape and `τ` sensitivity curves.

### Rigor

- At least 3 seeds for Track A comparisons. Report means with bootstrap 95% CIs.
- Budgets matched and logged for every run.
- Hyperparameters tuned **equally** for all policies. If the proposed method gets a sweep, the baselines get the same sweep.
- The evaluation set is frozen before the final comparisons and stored with a hash in the repo.

---

## 8. Phases and gates

Each phase ends with a **gate**. The agent reports gate results to the owner in plain language and waits for a go-ahead before starting expensive jobs in the next phase.

### Phase 0 — Foundations

1. Repo scaffold, CI, pre-commit, `AGENTS.md`, `CLAUDE.md` (§9–10).
2. HPC scripts and skills (§11).
3. **Verify the new cluster reality.** Find out which nodes host the 7 pro6000 cards and how many are on each node. Check max wall time on `gpu`, interconnect between nodes, whether compute nodes have internet (this matters for W&B and HF downloads), and scratch quota. Update `docs/hpc/bowdoin-hpc.md`.
4. Build the conda env in scratch. Confirm the PyTorch cu128 build, SDPA/FlexAttention backends on sm_120, and bf16.
5. Run a DDP smoke test on 1 → max GPUs per node → multi-node if applicable, and measure all-reduce bandwidth.
6. Slurm checkpoint/resume test: kill a job mid-run, requeue it, and confirm it resumes.
7. **Literature refresh.** The proposal's survey is from mid-2026. Search again for work combining causal video generation with graded or distance-based KV compression. Summarize in `docs/related_work.md` and flag anything that overlaps our claim.
8. ~~Pick the Track B base checkpoint.~~ *Deferred with Track B (owner decision 2026-09-22).*

**Gate 0:** a documented throughput number (Track B model: training samples/s per GPU), a working resume, a finished literature refresh, and a chosen base model.

### Phase 1 — Memory policy library

1. Implement all policies from §4.1 in `src/memory/` as pure PyTorch, CPU-testable.
2. Unit tests covering: budget never exceeded; horizon cutoff; aging is monotone (a cell never gets finer as it ages); `full` policy equals exact attention; RoPE-at-centroid correctness; `stats()` accuracy.
3. Visualizer showing tokens per distance for each policy, used as a figure in the paper.

**Gate 1:** all tests pass in CI, and a figure shows all policies at identical budget.

### Phase 2 — Track A (Memory Maze, then Minecraft)

1. Data pipeline, including scripted revisit trajectories with controlled gap distributions.
2. Train the base model with `full` memory at a short horizon to establish the ceiling.
3. Sweep policies × budgets (for example 3 budgets) × 3 seeds, one job per GPU.
4. Ablations: decay shape, `τ`; spatial-only vs. temporal-only vs. spacetime cells; RoPE handling; proportional attention; revisit augmentation on/off (H3); content-aware (H4).
5. Repeat the key comparisons on Minecraft.

**Gate 2 (decision point):** revisit-error-vs-gap curves. The owner and agents decide together.
- If H1 holds, go to Phase 3 with the best config.
- If it is a tie, go to Phase 3 anyway with the main policies, and reframe the paper around budget allocation and training.
- If it loses clearly, investigate why before spending Track B compute. The content-aware variant or the benchmark may become the paper.

### Phase 3 — Track B (scaled)

1. Integrate `MemoryPolicy` into the chosen causal Wan-based codebase behind a config flag.
2. Build the revisit-augmented long-clip data.
3. Fine-tune 3–4 policies with an identical recipe: `window_sink`, `relic_discrete`, `decay_continuous`, and optionally `decay_content`.
4. Evaluate at 1, 2, and 5 minutes.

**Gate 3:** Track B results table plus sample videos.

### Phase 4 — Push the horizon and analyze

1. Extend to 10 minutes with the best policy (the stretch goal).
2. Efficiency benchmarks and attention analysis.
3. Failure-case gallery.

### Phase 5 — Paper

See §13.

---

## 9. Repository layout

```
.
├── AGENTS.md                 # rules for all agents (Codex reads this)
├── CLAUDE.md                 # points Claude Code to AGENTS.md + skills
├── README.md
├── pyproject.toml            # ruff, black, pytest config
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml  # CPU tests + lint on every PR
├── .gitignore                # MUST include .env.hpc.local, outputs/, *.ckpt, wandb/
├── .claude/skills/           # Claude Code skills (see §11)
├── configs/                  # YAML configs; one file per experiment
│   ├── track_a/
│   └── track_b/
├── docs/
│   ├── PROJECT_PLAN.md       # this file
│   ├── proposal.md
│   ├── STATUS.md             # live handoff: current state, next steps, blockers
│   ├── DECISIONS.md          # decision log (ADR-style, append-only)
│   ├── EXPERIMENTS.md        # run registry
│   ├── DATASETS.md           # sources, licenses, locations on scratch
│   ├── related_work.md
│   ├── journal/              # dated plain-language updates for the owner
│   └── hpc/bowdoin-hpc.md
├── scripts/
│   ├── hpc/                  # canonical HPC scripts (skills wrap these)
│   └── data/
├── slurm/                    # sbatch templates
├── src/
│   ├── memory/               # MemoryPolicy implementations (core contribution)
│   ├── models/
│   ├── data/
│   ├── train/
│   └── eval/
├── tests/
└── paper/                    # LaTeX, figures generated by scripts only
```

Configs use plain YAML plus dataclasses (or Hydra if an agent makes the case in `DECISIONS.md`). Every run saves its **resolved config, git SHA, and Slurm job ID** next to its outputs.

---

## 10. Agent operating rules

These rules exist so that **no progress is lost when an agent's usage limit runs out mid-task**, and so that the owner can follow along.

### 10.1 Session protocol

At the **start** of every session:
1. `git pull`, then read `docs/STATUS.md`, the latest `docs/journal/` entry, and any open issues assigned to you.
2. Check running jobs with the `hpc-monitor` skill before submitting anything new.

During the session:
3. Work on a branch. **Commit small and often** (at least every meaningful step, and at most about 30 minutes of work between commits). **Push after every commit.** Unpushed work is lost work.
4. Before submitting any job longer than 30 minutes, push the exact commit the job runs, and register the run in `docs/EXPERIMENTS.md`.

At the **end** of every session, or when you sense you are close to a limit, do this first rather than last:
5. Update `docs/STATUS.md` with what is done, what is in progress (including branch names and running job IDs), what comes next, and any blockers.
6. Write a `docs/journal/YYYY-MM-DD-<agent>-<topic>.md` entry for the owner (§10.3).
7. Commit and push. Leave draft PRs open rather than unpushed branches.

### 10.2 Git conventions

- `main` is protected. Changes go through PRs, and CI must pass.
- Branch names: `phase<N>/<short-topic>`, for example `phase1/decay-policy`, `phase0/hpc-skills`.
- Commit messages follow Conventional Commits: `feat(memory): add continuous decay policy`, `fix(hpc): tolerate empty sacct reply`, `exp(track-a): launch policy sweep budget=16`.
- Open a PR as a **draft early**, then mark it ready when done. The PR description has four sections: **What**, **Why**, **How to verify**, and **Plain-language explanation** (written for the owner, as in §10.3).
- No force-pushing to shared branches and no history rewriting on `main`.
- One agent per branch. Use `git worktree` if you work on several branches.
- Tasks live in **GitHub Issues** with labels `phase-N`, `track-a`, `track-b`, `infra`, `paper`. An agent claims an issue by commenting before starting, so parallel agents do not duplicate work.

### 10.3 Keeping the owner informed

The owner wants to understand the work, not only see it done.

- Each journal entry and each PR's plain-language section explains **what you did, why it matters for the hypotheses, what you learned, and what is next**. Explain any new concept you relied on in two or three sentences (for example what proportional attention is, or why RoPE needs care when pooling).
- Report **results honestly**, including failures, surprises, and things you are unsure of.
- When a decision is non-obvious, add an entry to `docs/DECISIONS.md` covering context, options, choice, and consequences.
- At each phase gate, write a short summary and **ask the owner for a go-ahead** before launching the next phase's expensive jobs.

### 10.4 Code quality

- Python 3.11, type hints, `ruff` + `black` via pre-commit.
- Memory policies must have unit tests before they are used in any experiment.
- No notebooks in `src/`. Analysis notebooks go in `notebooks/` and cannot be imported by training code.
- Every figure in `paper/` is produced by a script in `scripts/` from logged results. No hand-made figures.
- Seeds are set and logged. Evaluation is deterministic where possible.

### 10.5 Hard rules

- **Never print, log, commit, or paste the HPC password or any token.** `.env.hpc.local` stays gitignored. If a secret is ever committed, stop and tell the owner immediately.
- Never run compute on the login node (`moosehead`).
- Never write caches, logs, checkpoints, or datasets under `$HOME` on the cluster (see the HPC reference, §3).
- Never delete another run's outputs on scratch.
- Do not request more GPUs than the current phase plan calls for, and follow cluster etiquette. The cards may be shared.

---

## 11. HPC skills

Canonical logic lives in `scripts/hpc/` as bash/Python scripts, so **both Codex and Claude** can use it. Claude Code skills in `.claude/skills/<name>/SKILL.md` wrap these scripts and document when to use them. `AGENTS.md` lists the same scripts for Codex (and Codex skills can mirror them if the Codex setup supports skills).

Build these in Phase 0, starting from the `expect`-based pattern in `docs/hpc/bowdoin-hpc.md` §1:

| Skill | Script | Does |
|---|---|---|
| `hpc-run` | `scripts/hpc/remote.sh "<cmd>"` | Runs one remote command on `moosehead`, propagates the exit code, never echoes the password. Detects exit 255 and says "VPN?". |
| `hpc-status` | `scripts/hpc/status.sh` | `sinfo` for GPU availability (pro6000 free/used per node), `squeue` for our jobs, `quota -s`, scratch usage. |
| `hpc-sync` | `scripts/hpc/checkout.sh --git-ref <ref>` | Clones the exact ref into a fresh scratch dir and links persistent data, weights, and env paths, since gitignored files are missing from a fresh clone. |
| `hpc-submit` | `scripts/hpc/submit.py --config <yaml> --gres --gpus --time ...` | Renders an sbatch from `slurm/` templates with all cache env vars set and output on scratch, submits with `--parsable`, and appends to `docs/EXPERIMENTS.md`. |
| `hpc-monitor` | `scripts/hpc/monitor.sh <jobid>` | Polls `sacct` (not `squeue`), tolerates empty replies, and tails the log. |
| `hpc-fetch` | `scripts/hpc/fetch.sh <run_id>` | `scp`s small artifacts (metrics, figures, sample videos) back. Heavy outputs stay on scratch. |
| `hpc-env` | `scripts/hpc/build_env.sh` | Builds or updates the conda env at `/mnt/hpc/tmp/$USER/envs/<name>` and records `pip freeze` in the repo. |
| `handoff` | none (process skill) | The end-of-session checklist from §10.1, steps 5–7. |

Every Slurm template must include:

- `--output=/mnt/hpc/tmp/%u/<project>/logs/%x-%j.out`
- The `TORCH_HOME`, `HF_HOME`, `PIP_CACHE_DIR`, and `XDG_CACHE_HOME` exports pointing to scratch.
- `--requeue` and `--signal=B:USR1@300`, with a trap that saves a checkpoint.
- A preamble printing `nvidia-smi --query-gpu=name,compute_mode --format=csv`, the git SHA, and the resolved config.

Checkpointing policy: save every about 30 minutes plus on SIGUSR1 to `/mnt/hpc/tmp/$USER/<project>/ckpt/<run_id>/`. Keep the last two checkpoints and the best one. Training scripts **auto-resume** from the latest checkpoint when it exists.

Experiment tracking: TensorBoard logs on scratch plus a CSV summary committed to the repo. W&B only if compute nodes have internet, or in offline mode synced from the login node. Phase 0 determines which.

Security note for the owner: storing the password in a file that agents can read is convenient but risky. If the cluster allows it, switch to an SSH key (the reference notes `ssh-copy-id` is untested) together with `ControlMaster` connection reuse. The skills should support both methods.

---

## 12. Compute plan

Total: 7 × RTX PRO 6000 Blackwell, 96 GB each, in Default compute mode, so multi-process CUDA works. Node layout is **to be verified in Phase 0**.

| Phase | Typical allocation | Pattern |
|---|---|---|
| 0 | 1 GPU, briefly all GPUs for the DDP test | smoke tests |
| 2 (Track A) | 7 independent 1-GPU jobs | Slurm job arrays: policy × budget × seed |
| 3 (Track B) | 1 policy on all GPUs of a node at a time, or policies in parallel with fewer GPUs each | DDP fine-tuning |
| 4 | 1–2 GPUs per eval job | long rollouts are memory-bound; parallelize across videos |

Rough budgeting, to be replaced with measured numbers after Gate 0:

- **Track A.** If one 150M-model run takes about 12–24 GPU-hours, a sweep of 7 policies × 3 budgets × 3 seeds is about 63 runs, or roughly 750–1,500 GPU-hours. Running 7 in parallel, that is about 4–9 days. Ablations add about 50%. Cut budgets to 2 or seeds to 2 if needed, but never below 2 seeds.
- **Track B.** Each policy's fine-tune is likely several days on the full set of cards. Plan for 3–4 policies. This is why ablations happen in Track A, and Track B only confirms the top configurations.

If the cards are shared and queues are long, Track A takes priority because it carries the scientific claim.

---

## 13. Paper plan

**Working title:** *Graceful Forgetting: Distance-Decayed Memory for Long-Horizon Autoregressive Video Generation*

Outline:
1. Introduction. The memory problem in causal video, the budget arithmetic from §2, and our claim.
2. Related work. Radial Attention, RELIC, LongLive, Sparse Forcing, PackForcing, VideoMLA, linear-attention streaming, Self-Forcing / CausVid, and memory in world models. Refreshed in Phase 0.
3. Method. The spacetime-cell view that unifies spatial and temporal compression, the continuous density `ρ(d)`, and the RoPE and proportional-attention details.
4. Revisit-consistency benchmark. Protocol and metrics stratified by revisit gap.
5. Controlled experiments (Track A). Headline curves and ablations.
6. Scaled experiments (Track B). Minute-scale results and efficiency.
7. Analysis. Attention vs. distance, when decay fails, and content-aware results.
8. Limitations, including an honest account of horizon and scale.

Venue strategy: a workshop paper once Track A results exist (for example a NeurIPS or ICLR workshop on video, world models, or efficient ML), then a main-conference submission with Track B. Agents keep `paper/` compiling from Phase 2 onward, so writing happens alongside the experiments rather than at the end. Code and the benchmark are released alongside the paper.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pure distance decay loses to content-aware selection (PackForcing evidence) | Test it explicitly as H4. A hybrid (distance prior × salience) is a natural follow-up and could be the stronger contribution. |
| The model ignores compressed memory | Revisit augmentation plus compressed-context teacher forcing, verified by the H3 ablation and attention analysis. |
| The Track B codebase makes the KV cache hard to swap | Choose the base model in Phase 0 partly on this criterion. Track A carries the claim either way. |
| RoPE or position extrapolation breaks at long horizons | Relative re-indexing, documented and ablated. |
| Cluster contention or wall-time limits | Checkpoint and auto-resume everywhere. Job arrays of short jobs. Prioritize Track A. |
| Someone publishes the same idea | Refresh the literature at every gate. The benchmark and analysis still differentiate us. Get the workshop version out early. |
| Blackwell kernel incompatibilities (sm_120) | Verify in Phase 0. Fall back to PyTorch SDPA / FlexAttention. |
| Lost progress when agent limits run out | §10.1 protocol: push often, `STATUS.md`, draft PRs, auto-resuming jobs. |
| Secret leakage | Gitignore, never echo, pre-commit secret scan (for example `detect-secrets` or `gitleaks`). |

---

## Appendix A — Starter `AGENTS.md`

Agents: create this file in Phase 0 and keep it short. Details belong in this plan.

```markdown
# AGENTS.md

Project: distance-decayed KV memory for causal video generation.
Read first: docs/PROJECT_PLAN.md, docs/STATUS.md, latest docs/journal/ entry.

## Every session
1. git pull; read STATUS.md; check running jobs (scripts/hpc/monitor.sh / status.sh).
2. Work on a branch `phase<N>/<topic>`; commit + push small and often.
3. Before any job >30 min: push the commit it runs, register it in docs/EXPERIMENTS.md.
4. Before stopping (or when near your limit — do this FIRST): update STATUS.md,
   write docs/journal/<date>-<agent>-<topic>.md in plain language, commit, push,
   leave a draft PR.

## HPC (details: docs/hpc/bowdoin-hpc.md)
- Remote commands: scripts/hpc/remote.sh "<cmd>"   (requires VPN; exit 255 = VPN down)
- Submit: scripts/hpc/submit.py ; monitor with sacct via scripts/hpc/monitor.sh
- Use --gres=gpu:pro6000:N. Never compute on moosehead. Never write to $HOME on the cluster.
- Never print or commit credentials (.env.hpc.local is gitignored).

## Code
Python 3.11, ruff+black, pytest. Memory policies need tests before experiments.
Figures only from scripts. Every run stores config + git SHA + job id.

## Communication
PRs and journal entries: What / Why / How to verify / Plain-language explanation.
Stop and ask the owner at every phase gate before launching expensive jobs.
```

## Appendix B — First issues to open

1. `infra`: Move plans into `docs/`, extract Appendix C and D into their own files, then repo scaffold, CI, pre-commit, gitignore, AGENTS.md, CLAUDE.md.
2. `infra`: HPC scripts and Claude skills (`hpc-run`, `hpc-status`, `hpc-sync`, `hpc-submit`, `hpc-monitor`, `hpc-fetch`, `hpc-env`, `handoff`).
3. `infra`: Verify the pro6000 node layout, wall-time limits, compute-node internet, and interconnect. Update the HPC reference.
4. `infra`: Conda env in scratch plus a Blackwell kernel check (SDPA, FlexAttention, bf16).
5. `infra`: DDP smoke test and checkpoint/resume/requeue test.
6. `paper`: Literature refresh (post-mid-2026) into `docs/related_work.md`.
7. ~~`track-b`: Evaluate Self-Forcing vs. LongLive codebases.~~ *Deferred.* Replaced by the Track A issue list in `docs/TRACK_A_PLAN.md` §12.
8. `phase-1`: `MemoryPolicy` interface, the `full`/`window`/`window_sink` policies, and tests.
9. `phase-1`: `relic_discrete` and `decay_continuous` with spacetime cells, RoPE-at-centroid, and tests.
10. `track-a`: Memory Maze data pipeline with scripted revisit trajectories.

---

## Appendix C — Original design proposal

[Read Appendix C: original design proposal](proposal.md).

---

## Appendix D — Bowdoin HPC reference

[Read Appendix D: Bowdoin HPC reference](hpc/bowdoin-hpc.md).
