#!/usr/bin/env python3
"""Render and submit a reproducible Slurm job through the canonical SSH wrapper."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "slurm" / "job.sbatch.tmpl"
EXPERIMENTS = ROOT / "docs" / "EXPERIMENTS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--gres", default="pro6000")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--cpus", default=4, type=int)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--job-name", default="dd-memory")
    parser.add_argument("--run-id")
    parser.add_argument("--remote-workdir", required=True)
    parser.add_argument(
        "--command",
        default="python -m src.train --config {config}",
        help="Shell command; {config} is replaced with the config path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    if args.gpus < 1 or args.cpus < 1:
        raise SystemExit("--gpus and --cpus must be positive")

    now = dt.datetime.now(dt.UTC)
    run_id = args.run_id or f"{args.job_name}-{now:%Y%m%dT%H%M%SZ}"
    if not all(char.isalnum() or char in "._-" for char in run_id):
        raise SystemExit("Run ID may contain only letters, numbers, '.', '_', and '-'")

    gres = args.gres if args.gres.startswith("gpu:") else f"gpu:{args.gres}:{args.gpus}"
    remote_config = f"{args.remote_workdir}/{args.config.as_posix()}"
    command = args.command.format(config=shlex.quote(remote_config))
    rendered = TEMPLATE.read_text(encoding="utf-8").format(
        job_name=args.job_name,
        partition=args.partition,
        gres=gres,
        cpus=args.cpus,
        memory=args.mem,
        wall_time=args.time,
        project="dd-memory",
        run_id=run_id,
        workdir=args.remote_workdir,
        config=remote_config,
        command=command,
    )

    if args.dry_run:
        sys.stdout.write(rendered)
        return 0

    encoded = base64.b64encode(rendered.encode()).decode()
    remote_file = f"/mnt/hpc/tmp/$USER/dd-memory/jobs/{run_id}.sbatch"
    remote_command = (
        "set -euo pipefail; mkdir -p /mnt/hpc/tmp/$USER/dd-memory/jobs; "
        f"printf %s {shlex.quote(encoded)} | base64 -d > {remote_file}; "
        f"sbatch --parsable {remote_file}"
    )
    result = subprocess.run(
        [str(ROOT / "scripts" / "hpc" / "remote.sh"), remote_command],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().splitlines()[-1]
    sha = git_sha()
    row = (
        f"| {run_id} | {now:%Y-%m-%d} | `{sha}` | `{args.config}` | {job_id} | "
        f"{args.partition}, {gres}, {args.time} | "
        f"`/mnt/hpc/tmp/$USER/dd-memory/runs/{run_id}` | submitted | |\n"
    )
    with EXPERIMENTS.open("a", encoding="utf-8") as handle:
        handle.write(row)
    print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
