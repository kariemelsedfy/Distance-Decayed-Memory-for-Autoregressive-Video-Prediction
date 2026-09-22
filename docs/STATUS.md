# Project status

**Updated:** 2026-09-22
**Active phase:** Phase 0 — foundations
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase0/ddp-requeue`
**PR:** #20 (draft; issue #6)

## Done

- Read `PROJECT_PLAN.md` and `TRACK_A_PLAN.md` in full.
- Moved both plans into `docs/`.
- Extracted Appendices C and D verbatim and replaced them with links.
- Added the initial repository documentation and quality scaffold.
- Added canonical HPC run/status/sync/submit/monitor/fetch/environment scripts
  and matching Claude skills.
- Tested `hpc-run` and `hpc-status` successfully against `moosehead`.
- Verified the seven pro6000 cards (3 on `moose68`, 4 on `moose69`), 30-day
  partition maximum, outbound compute-node internet, bonded network links, and
  scratch capacity. Short Slurm probes `68124`–`68126` completed.
- Added and dry-ran `scripts/github/create_track_a_issues.py`, containing all 17
  issues, labels, and dependency links from `TRACK_A_PLAN.md` §12.
- Created and verified all 17 GitHub issues as #2–#18, with `phase-*`,
  `track-a`, `infra`, and `paper` labels and explicit dependency references.
- Initialized the empty GitHub repository with the original three-file baseline,
  then pushed the bootstrap work in small commits on `phase0/bootstrap`.
- Opened draft PR #1 and linked issues #2, #3, and #4 to the work.
- Marked PR #1 ready for review after its `quality` CI job passed.
- Passed 3 repository tests, Ruff, Black, shell syntax checks, TOML/YAML parsing,
  appendix verification, and a dry render of the Slurm template.
- Merged the initial bootstrap as PR #1.
- Built the Python 3.11 conda environment entirely in scratch through a CPU
  Slurm job, with conda, pip, build, and compiler caches also on scratch.
- Recorded the exact environment in `environment/hpc-dd-memory.txt`: PyTorch
  2.11.0+cu128, CUDA runtime 12.8, Triton 3.6.0, cuDNN 9.19.0.56, NCCL 2.28.9,
  and NumPy 2.4.6.
- Added a reproducible Blackwell probe and 15-minute Slurm job. Final job
  `68171` ran on `moose68` at commit `e97171c` and returned finite bf16 results
  from matrix multiplication, all four forced PyTorch SDPA backends, and
  compiled FlexAttention on `sm_120`.
- Documented the result and its limits in `docs/hpc/blackwell-kernels.md` and
  updated the general Bowdoin HPC reference and experiment registry.
- Merged PR #19 (closes issue #5).
- On `phase0/ddp-requeue` (draft PR #20): added DDP/NCCL all-reduce and
  checkpoint/resume probes, a distributed launcher, bounded Slurm scripts, and
  `scripts/hpc/submit_phase0_smoke.sh` (modes `one`, `node`, `multi`, `requeue`).
- Fixed the submit script passing literal quotes into `DD_MEMORY_CHECKOUT`
  (cause of failed job `68178`); commit `8fe148f`.
- One-GPU DDP smoke job `68179` (moose68, `8fe148f`) passed: zero parameter
  delta, exact all-reduce, NCCL 2.28.9.

## In progress

- Four-GPU single-node job `68180` (moose69, run
  `phase0-node-20260922T170816Z`) was submitted at `8fe148f`; the VPN dropped
  before its result could be read. It is bounded to 10 minutes.

## Next

1. Reconnect the VPN; read `68180` with `scripts/hpc/monitor.sh 68180` and
   `runs/phase0-node-20260922T170816Z/results.json` on scratch.
2. Submit `--mode multi` (7 GPUs, heterogeneous job across both nodes) and
   `--mode requeue`, from checkout
   `/mnt/hpc/tmp/kelsedfy/dd-memory/checkouts/phase0-ddp-requeue-20260922T170118Z`.
   The heterogeneous launch path (`--het-group=0,1`, `WORLD_SIZE` from
   `SLURM_NTASKS`) has not run yet and is the most likely to need a fix.
3. Record results in `docs/EXPERIMENTS.md` and a `docs/hpc/` note, then mark
   PR #20 ready.
4. Issue #7 (literature refresh) remains independent.

## Blockers

HPC home is at its 25,600 MB hard limit; keep everything on scratch.
VPN access was lost at about 17:15 UTC on 2026-09-22.

## Running jobs

- `68180` (ddp-node, ≤10 min) — status unconfirmed because of the VPN drop.
