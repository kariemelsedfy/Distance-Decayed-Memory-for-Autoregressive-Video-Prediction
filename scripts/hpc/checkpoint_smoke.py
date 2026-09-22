#!/usr/bin/env python3
"""Exercise atomic CUDA checkpoint save and resume across a Slurm requeue."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

STOP_REQUESTED = False


def request_stop(_signal_number: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--total-steps", default=20, type=int)
    parser.add_argument("--step-seconds", default=0.5, type=float)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> bool:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.total_steps < 2 or args.step_seconds < 0:
        raise ValueError(
            "total-steps must be at least two and step-seconds nonnegative"
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "checkpoint.pt"
    result_path = args.run_dir / "results.json"
    attempt_path = args.run_dir / "attempts.jsonl"
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(20260922)
    model = torch.nn.Linear(16, 16, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step = 0
    resumed_from_step = 0

    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        resumed_from_step = step

    with attempt_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "restart_count": restart_count,
                    "resumed_from_step": resumed_from_step,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                },
                sort_keys=True,
            )
            + "\n"
        )

    signal.signal(signal.SIGUSR1, request_stop)
    while step < args.total_steps:
        generator = torch.Generator(device=device).manual_seed(100_000 + step)
        inputs = torch.randn((32, 16), generator=generator, device=device)
        targets = torch.randn((32, 16), generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        step += 1

        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": float(loss.detach().item()),
                "restart_count": restart_count,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        if STOP_REQUESTED:
            print(f"Checkpointed step {step} after SIGUSR1", flush=True)
            return False
        time.sleep(args.step_seconds)

    passed = restart_count >= 1 and resumed_from_step > 0 and step == args.total_steps
    report = {
        "passed": passed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "restart_count": restart_count,
        "resumed_from_step": resumed_from_step,
        "final_step": step,
        "total_steps": args.total_steps,
        "device_name": torch.cuda.get_device_name(device),
        "checkpoint": str(checkpoint_path),
    }
    write_json_atomic(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return passed


def main() -> int:
    try:
        return 0 if run(parse_args()) else 99
    except Exception as error:  # noqa: BLE001 - report rank failures to Slurm
        print(f"Checkpoint smoke failed: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
