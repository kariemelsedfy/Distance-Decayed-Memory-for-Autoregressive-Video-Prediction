---
name: hpc-sync
description: Clone an exact Git ref into a fresh Bowdoin scratch checkout and link persistent project storage.
---

# HPC Sync

Run:

```bash
scripts/hpc/checkout.sh --git-ref <commit-or-branch> --repo <git-url>
```

Prefer a pushed commit SHA. The script creates a fresh checkout under
`/mnt/hpc/tmp/$USER/dd-memory/checkouts/` and links persistent data, weights,
environment, log, and checkpoint locations. A fresh checkout never contains
gitignored credentials or assets; reference those through persistent scratch.
