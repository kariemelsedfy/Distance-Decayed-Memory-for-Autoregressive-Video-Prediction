
# Bowdoin HPC — general reference

Cluster facts only; nothing project-specific. Everything here was verified in
practice during 2026 unless marked *(unverified)*. Copy this file into any new
project and it should still be true.

---

## Phase 0 live verification — 2026-09-22

The initial Track A bootstrap rechecked the current cluster rather than relying
on the older notes below.

| Item | Verified result |
|---|---|
| RTX PRO 6000 placement | **7 total:** 3 on `moose68`, 4 on `moose69` (`scontrol` also reports `gres/gpu:pro6000=7`). |
| Card identity | `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97,887 MiB reported, Default compute mode (sampled on `moose68`). |
| `gpu` wall time | Default **14 days**; maximum **30 days**; partition QoS is `maxgpu`. |
| GPU-partition policy | Jobs on `gpu` must request a GPU GRES; Slurm rejects CPU-only allocations on this partition. |
| Compute-node internet | Outbound DNS/TLS/HTTPS works on both `moose68` and `moose69`; Hugging Face returned HTTP 200 on both, and the W&B API endpoint was reachable from `moose68`. |
| Inter-node links | Both nodes report `bond0` using IEEE 802.3ad link aggregation with two up, full-duplex **100,000 Mbps** member links. This identifies the link configuration, not measured training throughput. |
| Scratch filesystem | Gluster filesystem is **35 TB total, 34 TB available, 2% used**; this user's scratch usage was 8.2 GB after building the Phase 0 environment. No per-user scratch quota was exposed by `quota -s`, so confirm allocation and purge policy with HPC staff. |
| Home quota warning | This account reported 25,600 MB used against a 25,600 MB hard limit after the environment build. Keep all caches, logs, data, checkpoints, and environments on scratch. |
| Blackwell software probe | On `moose68`, PyTorch 2.11.0+cu128 recognized compute capability 12.0 and bf16. Forced math, Flash, efficient, and cuDNN SDPA backends plus compiled FlexAttention all returned finite outputs. See [`blackwell-kernels.md`](blackwell-kernels.md). |

Evidence came from read-only Slurm/storage queries, two-minute jobs
`68124`–`68126`, and the 15-minute-capped kernel check `68171`. No benchmark or
long-running GPU workload was launched.

---

## 1. Getting on

| Host | Role |
|---|---|
| `moosehead.bowdoin.edu` | login + Slurm submission (the one you want) |
| `dover.bowdoin.edu` | interactive shell host |
| `hpcweb.bowdoin.edu` | web portal |

- **Off campus, you must be on the Bowdoin VPN.** Without it `ssh` just exits
  `255` with no message — that's the signature, not a broken account.
- **Password auth only** on a fresh account; no key is installed by default.
  (`ssh-copy-id` to install a key is *(unverified)* — never tried it.)
- **The login node will not run compute.** `python3` on `moosehead` refuses with
  a "does not have the resources" warning telling you to use Slurm or an
  interactive host. Treat `moosehead` as orchestration only: `sbatch`, `squeue`,
  `sacct`, file ops.

### Automating password SSH

Store credentials in a gitignored `.env.hpc.local` in the project root:

```bash
BOWDOIN_HPC_HOST=moosehead.bowdoin.edu
BOWDOIN_HPC_INTERACTIVE_HOST=dover.bowdoin.edu
BOWDOIN_HPC_USER=<username>
BOWDOIN_HPC_PASSWORD=<password>
```

Then drive it with `expect` (present on macOS by default). The pattern that
works — it never echoes the password, and it propagates the remote exit code:

```bash
#!/usr/bin/env bash
set -euo pipefail
set -a; source .env.hpc.local; set +a
export RC="$1"                       # the remote command

expect <<'EOF'
set timeout -1
log_user 0
set sent 0
eval spawn -noecho [list ssh \
  -o StrictHostKeyChecking=accept-new \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o NumberOfPasswordPrompts=1 \
  -o ConnectTimeout=20 \
  "$env(BOWDOIN_HPC_USER)@$env(BOWDOIN_HPC_HOST)" $env(RC)]
expect {
  -re "(?i)yes/no"   { send "yes\r"; exp_continue }
  -re "(?i)password:" { send "$env(BOWDOIN_HPC_PASSWORD)\r"; set sent 1; log_user 1; exp_continue }
  -re {Permission denied|[Aa]uthentication failed} { exit 2 }
  timeout { exit 124 }
  eof { if {!$sent} { log_user 1 }; catch wait r; exit [lindex $r 3] }
}
EOF
```

The same `expect` block with `scp -r` instead of `ssh` handles uploads. Key
options: `PubkeyAuthentication=no` (otherwise ssh burns the prompt budget on
keys), `NumberOfPasswordPrompts=1`, `log_user 0` until the password is sent.

---

## 2. Hardware

Two partitions: **`gpu`** and **`main`** (CPU-only).

GPU nodes and their GRES names — this is what I've actually landed jobs on, not
necessarily the full cluster:

| `--gres` | Card | VRAM | Node(s) | Compute mode |
|---|---|---|---|---|
| `gpu:pro6000:1` | RTX PRO 6000 Blackwell Server Ed. | 96 GB (97,887 MiB reported) | `moose68` (3), `moose69` (4) | **Default** |
| `gpu:a100:1` | A100 | 40/80 GB *(unverified which)* | `moose63`, `moose66` | Exclusive_Process |
| `gpu:rtx5090:1` | RTX 5090 | 32 GB | `moose63`, `moose66` | Exclusive_Process |
| `gpu:rtx2080:1` | RTX 2080 | 11 GB | `moose54`, `moose59`–`moose62`, `moose64`, `moose65` | Exclusive_Process |
| `gpu:rtx3080:1` | RTX 3080 | 10 GB | `moose63` (3), `moose66` (3) | Exclusive_Process |

CPU nodes seen on `main`: `moose12`, `moose21`.

Always check live rather than trusting the table:

```bash
sinfo -p gpu -o '%P %N %G %D %t'      # what exists and what's idle
sinfo -p gpu -O nodehost,gres,gresused,statecompact
```

Ask for a card explicitly (`--gres=gpu:pro6000:1`); a bare `--gres=gpu:1` gives
you whatever is free, which is usually the 2080.

### The Exclusive_Process trap — the single most important fact

**Only the pro6000 nodes are in Default compute mode. Every other GPU is
`Exclusive_Process`: exactly one CUDA context per GPU, cluster-wide.**

Any workload where **two processes** touch the GPU fails on those nodes — a
parent that initializes CUDA and then spawns a CUDA child, a subprocess-based
renderer/inference tool, a DataLoader worker that touches CUDA, two ranks pinned
to one device. The failure is at the first `.to(device)`:

```
CUDA error: CUDA-capable device(s) is/are busy or unavailable
(cudaErrorDevicesUnavailable)
```

It reads like a broken driver or an occupied GPU; it is neither. Options:

1. Request `pro6000` (Default mode — parent and child coexist fine).
2. Keep exactly one process on CUDA — push the lighter stage to CPU.
3. Force any ONNX/secondary runtime to `CPUExecutionProvider`.

Check before you debug anything else:
`nvidia-smi --query-gpu=name,compute_mode --format=csv`

### Node quirks

- **`moose63`'s A100 is flaky** — Slurm reports it idle, then torch CUDA init
  fails. Prefer `-w moose66` for A100 work.
- `onnxruntime-gpu` logs a missing `libcublasLt.so.11` CUDA-provider error on the
  older cards; it falls back to CPU and jobs still complete.
- `moose68` reported pro6000 driver `615.71.09`; PyTorch 2.11.0+cu128 and its
  CUDA 12.8 runtime work on `sm_120` there.

---

## 3. Storage — plan around the home quota

| Path | Size | Use |
|---|---|---|
| `/home/<user>` | **20 G soft / 25.6 G hard** | code and conda env only. A conda env alone is ~8–9 G, so this fills fast. |
| `/mnt/hpc/tmp/<user>` | **35 TB total / 34 TB free** on 2026-09-22, `gluster1.bowdoin.edu:/gv0` | durable scratch — data, weights, checkpoints, logs |
| node-local `/tmp` | per-node | runtime staging; fastest, disappears |

Home fills up and then **everything fails in confusing ways**:

- HF / torch-hub model downloads die with
  `OSError: [Errno 122] Disk quota exceeded`
- A Slurm `--output` path in `$HOME` can kill a job **mid-run**, hours in, when
  the log grows past the quota.

So on every job, always:

```bash
export TORCH_HOME=/mnt/hpc/tmp/$USER/model_cache/torch
export HF_HOME=/mnt/hpc/tmp/$USER/model_cache/hf
export HUGGINGFACE_HUB_CACHE=/mnt/hpc/tmp/$USER/model_cache/hf
export PIP_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/pip
export XDG_CACHE_HOME=/mnt/hpc/tmp/$USER/cache
```

and set `#SBATCH --output=/mnt/hpc/tmp/%u/<project>/%x-%j.out` (or node-local
`/tmp/%x-%j.out` and copy it back at the end).

Check quota / usage: `quota -s`, `du -sh ~/.cache ~/.conda`,
`df -h /mnt/hpc/tmp`. *(Scratch purge policy on `/mnt/hpc/tmp` is unverified —
I've never had anything deleted, but don't assume it's permanent.)*

---

## 4. Software environment

- Modules: `miniconda3`, `python3.11`, `python3.11.8`, `cuda-12.8.1`,
  `cuda-12.9.1`, `cuda-13.1` (`module avail`, `module load cuda-12.8.1`)
- `/usr/bin/ffmpeg` and `ffprobe` exist system-wide
- Build conda envs **in scratch**, not home, if they're large:
  `conda create -p /mnt/hpc/tmp/$USER/envs/<name> python=3.11`
- Call the env's interpreter by absolute path in jobs (`$ENV/bin/python`) rather
  than relying on `conda activate` inside a non-interactive shell
- Verified on `moose68`: Python 3.11.16, PyTorch 2.11.0+cu128, CUDA runtime
  12.8, Triton 3.6.0, bf16, all four PyTorch SDPA backends, and compiled
  FlexAttention. This was a correctness smoke test, not a benchmark; see
  [`blackwell-kernels.md`](blackwell-kernels.md).
- `huggingface-cli download` is deprecated — the working form is
  `hf download <repo> --local-dir <path>`

---

## 5. Slurm patterns

The `gpu` partition currently has a 14-day default and 30-day maximum wall
time. It requires a GPU GRES request even for an otherwise CPU-only probe.

### Job header

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=gpu            # or: main (CPU only)
#SBATCH --gres=gpu:pro6000:1       # name the card
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/hpc/tmp/%u/myproject/%x-%j.out
```

### Everyday commands

```bash
srun -p gpu --gres=gpu:rtx2080:1 -c 4 --mem=16G -t 00:30:00 --pty bash   # interactive
sbatch --parsable job.sbatch                                             # → bare job id
squeue -u $USER -o '%i %j %T %M %N'
sacct -j <id> -o JobID,JobName,Elapsed,State,AllocTRES%60,NodeList -P    # survives job exit
scancel <id>
sbatch -w moose66 ...            # pin a node
sbatch -p gpu --gres=gpu:a100:1 --exclude=moose63 ...
```

**Poll with `sacct`, not `squeue`.** `squeue` returns empty both for "finished"
and for a transient blip / dropped VPN; `sacct` is authoritative and still
answers after the job ends. A polling loop should tolerate empty replies and
retry rather than declaring the job done.

### Round-trip workflow that works well

For driving the cluster from a laptop over a flaky VPN:

1. push code to git; on the cluster, **clone the ref into a fresh scratch dir**
   (reproducible, no rsync drift — but remember a fresh clone has **no gitignored
   files**: external checkouts, weights and data must be pointed at a persistent
   path or re-staged)
2. `sbatch --parsable` with resource overrides as flags
3. poll `sacct` until terminal state
4. `scp` artifacts back; keep heavy outputs on scratch
5. stage runtime-hot files on node-local `/tmp`, persist results to
   `/mnt/hpc/tmp/<user>/`

Wrap each stage as one script taking `--git-ref --partition --gres --cpus --mem
--time --timeout --poll-seconds`, so a rerun is one command.

---

## 6. Failure cheat-sheet

| Symptom | Real cause | Fix |
|---|---|---|
| `ssh` exits `255`, silent | not on the VPN | connect VPN |
| "does not have the resources" running python | you're on the login node | `srun` / `sbatch` |
| `cudaErrorDevicesUnavailable` at first `.to(device)` | Exclusive_Process GPU, second CUDA context | use `pro6000`, or keep one process on CUDA |
| `Errno 122 Disk quota exceeded` | home full (caches or Slurm log) | caches + `--output` → `/mnt/hpc/tmp` |
| Job dies hours in, no traceback | Slurm log hit the home quota | log to scratch or node-local `/tmp` |
| GPU "idle" but torch init fails | `moose63` A100 flakiness | `-w moose66` / `--exclude=moose63` |
| Fresh scratch clone fails immediately | gitignored deps (weights, external checkouts) absent | point env vars at persistent scratch copies |
| `libcublasLt.so.11` not found (onnxruntime) | CUDA-provider mismatch on older cards | harmless; or pin CPU provider |
| Job pends forever on `--gres=gpu:pro6000:1` | both pro6000 nodes busy | `sinfo` first; fall back to another card |
| Job pends forever, `Reason=QOSMaxCpuPerUserLimit` | asked for more than the per-user QOS ceiling (2 pro6000; 4 CPUs/40G on `gpu`) | shrink the request, or use `mixed` for CPU/memory; see [ddp-and-requeue.md](ddp-and-requeue.md) |
| NCCL `invalid device ordinal` in `shm` transport | `--gpu-bind=single` hid each rank's peer GPU | allocate `--gres=gpu:pro6000:N` per node; pick the device by `SLURM_LOCALID` |
| Job killed by its own preemption signal | `SIGUSR1` handler installed after the torch import | register the handler before any heavy import |
