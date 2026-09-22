---
name: hpc-status
description: Inspect GPU availability, the current user's Slurm jobs, home quota, and scratch capacity on Bowdoin HPC.
---

# HPC Status

Run `scripts/hpc/status.sh` before submitting work. It reports configured and
allocated GRES per GPU node, the current user's queue, home quota, scratch
filesystem capacity, and the user's scratch usage. Treat an SSH exit of 255 as
a likely VPN problem.
