---
name: hpc-monitor
description: Monitor a Bowdoin Slurm job reliably with sacct and its scratch log.
---

# HPC Monitor

Run `scripts/hpc/monitor.sh <job-id>`. It polls `sacct`, not `squeue`, shows the
Slurm log tail when available, and retries empty replies instead of treating
them as completion. Set `HPC_POLL_SECONDS` or `HPC_MONITOR_TIMEOUT` when a
different polling cadence or bounded local wait is needed.
