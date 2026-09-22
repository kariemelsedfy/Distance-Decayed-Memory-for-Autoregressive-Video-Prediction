# Phase 0 bootstrap — 2026-09-22

## What changed

I read both project plans in full and kept Track A as the only active scope. I
moved the plans into `docs/`, extracted the proposal and Bowdoin reference
verbatim, replaced the embedded appendices with links, and added the initial
README, CI, pre-commit, ignore rules, agent instructions, status, decisions,
experiment registry, and dataset registry.

I also added one canonical set of HPC scripts under `scripts/hpc/` and thin
Claude skills that point to those scripts. `hpc-run` and `hpc-status` were tested
against the real cluster without exposing the password.

The complete GitHub issue set is encoded in
`scripts/github/create_track_a_issues.py`. It created all 17 planned issues as
#2–#18, with labels and concrete issue-number dependency links. The work is on
`phase0/bootstrap` in draft PR #1.

## Why this matters

The scaffold makes the later policy comparison auditable: secrets and large
outputs are excluded, CPU checks run in CI, decisions D-001 through D-006 are
fixed in writing, and every future cluster run has a common submission and
monitoring path. Centralizing the cluster behavior also means humans, Codex,
and Claude do not quietly use different commands.

The live audit replaces planning assumptions with facts. The seven pro6000
cards are split across two nodes, so A1 can use at most four same-node GPUs
without crossing the network; any seven-GPU DDP run is necessarily multi-node.

## What we learned

- `moose68` has 3 pro6000 cards and `moose69` has 4. A sampled card reported
  97,887 MiB and Default compute mode.
- The GPU partition defaults to 14 days and permits at most 30 days. It rejects
  jobs that do not request a GPU GRES.
- Both pro6000 nodes can reach Hugging Face over HTTPS. The W&B API endpoint is
  also reachable, so online tracking is technically possible (authentication
  and project policy still need to be configured).
- Both nodes use an 802.3ad bonded interface with two 100-Gbps full-duplex member
  links. This is link configuration, not a measured all-reduce bandwidth.
- Scratch has about 34 TB available. The home directory is nearly at its hard
  quota, so scratch-only caches, environments, logs, and checkpoints are not
  optional.

The only GPU allocations were jobs `68124`–`68126`, each capped at two minutes;
they completed during the session.

## How to verify

- Run `scripts/hpc/remote.sh "hostname && command -v sbatch"`.
- Run `scripts/hpc/status.sh`.
- Run `bash -n scripts/hpc/*.sh` and
  `python scripts/hpc/submit.py --help` locally.
- Run `python scripts/github/create_track_a_issues.py --repo OWNER/REPO --dry-run`.
- Direct pytest, Ruff, and Black checks pass in the temporary test environment.
  The repository is now initialized, so CI and pre-commit can repeat them.

## Handoff and next steps

The Git and GitHub prerequisites were resolved later in the session. I created
an initial `main` commit matching the original three-file starting state, pushed
each Phase 0 commit immediately, opened draft PR #1, and added claim/progress
comments to issues #2–#4.

There are no running jobs and no blocker for this bootstrap scope. After PR #1
is reviewed and merged, Phase 0 continues with issue #5 (scratch environment and
Blackwell kernels), issue #6 (DDP plus checkpoint/requeue), and issue #7
(literature refresh). Track B remains deferred.
