# Project status

**Updated:** 2026-09-22
**Active phase:** Phase 0 — foundations
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase0/bootstrap`
**PR:** #1 (ready for review; CI green)

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

## In progress

- PR #1 is awaiting owner review and merge.

## Next

1. Review and merge PR #1.
2. Continue Phase 0 with issue #5: build the scratch conda environment and run
   Blackwell kernel checks.
3. Then address issue #6 (DDP/checkpoint/requeue) and issue #7 (literature refresh).

## Blockers

None for the requested bootstrap scope. The nearly full HPC home quota remains
an operational risk; keep caches, environments, logs, data, and checkpoints on
scratch.

## Running jobs

None. Short verification jobs `68124`, `68125`, and `68126` completed.
