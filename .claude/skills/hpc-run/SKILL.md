---
name: hpc-run
description: Run one safe, non-interactive command on Bowdoin HPC through the canonical credential-protecting wrapper.
---

# HPC Run

Use from the repository root:

```bash
scripts/hpc/remote.sh "hostname && command -v sbatch"
```

The script reads `.env.hpc.local`, suppresses the password, propagates the
remote exit code, and explains that exit 255 usually means the VPN is down.
Never put a credential in the command string. Never run compute on the login
node; use this skill for inspection and Slurm orchestration only.
