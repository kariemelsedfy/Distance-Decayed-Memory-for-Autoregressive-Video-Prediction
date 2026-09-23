# Memory Maze on Bowdoin `main`

## Verified result

Job `68321` completed on `moose17` at commit `6794684` with one CPU, 8 GB of
memory, and no GPU allocation. The environment was Python 3.11.16 with
`memory-maze==1.0.3`, `dm-control==1.0.47`, `mujoco==3.14.0`, and
`gym==0.26.2`.

The working headless configuration is:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_SHADER_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/mesa
```

The runtime reported `Mesa` and `llvmpipe (LLVM 21.1.8, 256 bits)`, proving
that EGL was using CPU software rendering rather than a GPU. Three timed runs
of 2,000 frames measured 23.85, 23.97, and 23.43 frames/s. Median throughput was
**23.85 frames/s per core**; mean throughput was 23.75 frames/s.

The verified observation dictionary contains:

| Key | Shape | dtype | Meaning |
|---|---:|---|---|
| `image` | 64×64×3 | uint8 | first-person RGB frame |
| `maze_layout` | 9×9 | uint8 | binary map; 1 is traversable and 0 is wall |
| `agent_pos` | 2 | float64 | position in maze-grid coordinates |
| `agent_dir` | 2 | float64 | unit heading vector |
| `target_color` | 3 | float64 | current target RGB |
| `target_pos` | 2 | float64 | current target in global grid coordinates |
| `target_vec` | 2 | float64 | current target relative to the agent |
| `targets_pos` | 3×2 | float64 | all target positions |
| `targets_vec` | 3×2 | float64 | all target-relative vectors |

There is no action observation key. The action spec is a six-valued discrete
input in this order: no-op, forward, left, right, forward+left, forward+right.
The underlying control vectors are `[0,0]`, `[-1,0]`, `[0,-1]`, `[0,1]`,
`[-1,-1]`, and `[-1,1]`. A scalar heading for the dataset can be derived as
`atan2(agent_dir[1], agent_dir[0])`.

## What did not work

- Job `68314` exited before logging because `/etc/profile` was sourced after
  strict shell mode. The Slurm script now sources it first.
- Job `68315` tried the planned OSMesa route. Bowdoin's CPU nodes do not have
  system OSMesa or Xvfb, and current conda-forge `mesalib` did not provide
  `libOSMesa.so`; PyOpenGL therefore failed at import. Short diagnostics
  `68316`–`68320` established that forced software EGL works on `main`.
- Gym prints its upstream maintenance warning under NumPy 2.x, but Memory Maze
  uses its `dm_env` interface for this project. Environment construction,
  resets, observations, and 6,200 rendered steps all passed.

## Reproduce

Push the desired commit, create a fresh scratch checkout, submit, and monitor:

```bash
scripts/hpc/checkout.sh --git-ref <ref> --repo <repository-url>
scripts/hpc/submit_memory_maze_probe.sh --workdir <printed-checkout-path>
scripts/hpc/monitor.sh <job-id>
```

The job writes `results.json`, `pip-freeze.txt`, `conda-explicit.txt`, and
`git-sha.txt` under `/mnt/hpc/tmp/$USER/dd-memory/runs/<run-id>/`.
