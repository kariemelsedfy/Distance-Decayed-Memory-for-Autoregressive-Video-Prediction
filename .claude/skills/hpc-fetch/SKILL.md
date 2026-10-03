---
name: hpc-fetch
description: Fetch small metrics, figures, and sample-video artifacts from a Bowdoin scratch run.
---

# HPC Fetch

Run `scripts/hpc/fetch.sh <run-id> [local-directory]`. The script packages only
small result formats such as JSON, CSV, figures, GIFs, MP4 samples, and
NumPy archives under 50 MB (evaluation records), then
copies and extracts that archive locally. Checkpoints, datasets, and other heavy
outputs remain on scratch.
