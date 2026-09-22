---
name: hpc-submit
description: Render, submit, and register a reproducible Slurm job for this project.
---

# HPC Submit

Before any job over 30 minutes, push its exact commit and add or verify the run
configuration. Then use `scripts/hpc/submit.py --help`. Required inputs include
`--config` and `--remote-workdir`; resource controls include `--gres`, `--gpus`,
`--time`, `--cpus`, and `--mem`.

The renderer uses `slurm/job.sbatch.tmpl`, puts logs and caches on scratch,
requests requeue with a five-minute USR1 warning, prints the Git SHA and config,
submits with `--parsable`, and appends the job to `docs/EXPERIMENTS.md`. Use
`--dry-run` to inspect the script without submitting.
