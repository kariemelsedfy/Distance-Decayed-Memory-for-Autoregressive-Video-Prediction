---
name: hpc-env
description: Build or update the project's Python 3.11 and PyTorch cu128 environment in Bowdoin scratch.
---

# HPC Environment

After syncing a pushed checkout, run:

```bash
scripts/hpc/build_env.sh --workdir <remote-checkout>
```

The environment lives under `/mnt/hpc/tmp/$USER/envs/`, all package caches use
scratch, and `pip freeze` is written into the checkout's `environment/`
directory. Commit that lock snapshot after fetching it. Do not build a large
environment in the home directory.
