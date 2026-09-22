# Claude Code project instructions

Read and follow [AGENTS.md](AGENTS.md) first. The full process and scientific
scope are in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md); the active technical
specification is [docs/TRACK_A_PLAN.md](docs/TRACK_A_PLAN.md).

Track A is active. Track B is deferred until the owner explicitly reopens it
after Gate 2.

Reusable cluster workflows live in `.claude/skills/`. Those skills wrap the
canonical scripts in `scripts/hpc/`; update the script first when behavior
changes so Codex, Claude, and humans all run the same implementation.
