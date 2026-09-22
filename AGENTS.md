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
