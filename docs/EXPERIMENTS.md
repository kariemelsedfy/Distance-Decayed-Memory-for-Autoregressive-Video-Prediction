# Experiment registry

No research experiment has been launched. The Phase 0 infrastructure smoke test
is recorded below so its GPU use and result remain auditable.

Register every run before submission when its requested wall time exceeds 30
minutes. Do not record credentials or private tokens.

| Run ID | Date | Git SHA | Config | Slurm job | Resources | Output | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| `blackwell-kernels-20260922` | 2026-09-22 | `e97171c` | bf16; B=1, H=4, T=128, D=64 | `68171` | 1× pro6000, 4 CPU, 16 GB, 15 min | `runs/blackwell-check-68171/results.json` on scratch | completed | All forced SDPA backends, compiled FlexAttention, and bf16 matmul passed with finite outputs. Environment refresh job `68170` also completed. |
