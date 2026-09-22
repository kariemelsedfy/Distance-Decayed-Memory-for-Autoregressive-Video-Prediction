---
name: handoff
description: Preserve project state before stopping, changing agents, or nearing a usage limit.
---

# Handoff

Before stopping:

1. Update `docs/STATUS.md` with completed work, active branches, running job IDs,
   next steps, and blockers.
2. Write `docs/journal/YYYY-MM-DD-<agent>-<topic>.md` explaining what changed,
   why it matters, how to verify it, what was learned, and what comes next.
3. Commit with a Conventional Commit message, push immediately, and leave a
   draft PR. Never leave the only copy of useful work unpushed.
