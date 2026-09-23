#!/usr/bin/env python3
"""Check CUDA DDP correctness and measure a simple NCCL all-reduce."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-mib", default=128, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    return parser.parse_args()


def effective_bus_bandwidth_gbps(
    algorithm_bandwidth_gbps: float, world_size: int
) -> float:
    """Convert ring all-reduce algorithm bandwidth to standard bus bandwidth."""
    if world_size <= 1:
        return 0.0
    return algorithm_bandwidth_gbps * 2 * (world_size - 1) / world_size


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> bool:
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    # Every rank on a node sees every GPU of that node, so NCCL can open a
    # peer's device for its shared-memory transport. Each rank owns the device
    # matching its node-local rank.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Local rank {local_rank} has no GPU; this node exposes "
            f"{torch.cuda.device_count()}"
        )

    distributed.init_process_group(backend="nccl", timeout=dt.timedelta(seconds=180))
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rank_info = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "slurm_local_id": int(os.environ.get("SLURM_LOCALID", "0")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device),
    }
    topology: list[dict[str, Any] | None] = [None] * world_size
    distributed.all_gather_object(topology, rank_info)

    torch.manual_seed(20260922)
    model = torch.nn.Linear(32, 16, bias=True, device=device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    inputs = torch.randn((8, 32), generator=generator, device=device)
    targets = torch.randn((8, 16), generator=generator, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(wrapped(inputs), targets)
    loss.backward()
    optimizer.step()

    flattened = torch.cat(
        [parameter.detach().flatten() for parameter in model.parameters()]
    )
    reference = flattened.clone()
    distributed.broadcast(reference, src=0)
    parameter_delta = (flattened - reference).abs().max()
    distributed.all_reduce(parameter_delta, op=distributed.ReduceOp.MAX)
    max_parameter_delta = float(parameter_delta.item())

    if args.tensor_mib < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError(
            "tensor-mib and iterations must be positive; warmup may be zero"
        )
    element_count = args.tensor_mib * 1024 * 1024 // 4
    collective = torch.empty(element_count, dtype=torch.float32, device=device)
    expected_sum = world_size * (world_size + 1) / 2

    for _ in range(args.warmup):
        collective.fill_(rank + 1)
        distributed.all_reduce(collective)
    torch.cuda.synchronize()
    distributed.barrier()

    started = time.perf_counter()
    for _ in range(args.iterations):
        collective.fill_(rank + 1)
        distributed.all_reduce(collective)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
    distributed.all_reduce(elapsed_tensor, op=distributed.ReduceOp.MAX)
    max_elapsed = float(elapsed_tensor.item())
    collective_error = (collective - expected_sum).abs().max()
    distributed.all_reduce(collective_error, op=distributed.ReduceOp.MAX)
    max_collective_error = float(collective_error.item())

    bytes_per_iteration = collective.numel() * collective.element_size()
    algorithm_bandwidth = (
        bytes_per_iteration * args.iterations / max_elapsed / 1_000_000_000
    )
    bus_bandwidth = effective_bus_bandwidth_gbps(algorithm_bandwidth, world_size)
    passed = max_parameter_delta <= 1e-6 and max_collective_error == 0.0

    if rank == 0:
        report: dict[str, Any] = {
            "passed": passed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "backend": distributed.get_backend(),
            "world_size": world_size,
            "node_count": len({entry["hostname"] for entry in topology if entry}),
            "topology": topology,
            "ddp": {
                "loss": float(loss.detach().item()),
                "max_parameter_delta": max_parameter_delta,
            },
            "all_reduce": {
                "dtype": str(collective.dtype),
                "tensor_mib": args.tensor_mib,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "max_elapsed_seconds": max_elapsed,
                "max_absolute_error": max_collective_error,
                "algorithm_bandwidth_gbps": algorithm_bandwidth,
                "bus_bandwidth_gbps": bus_bandwidth,
            },
        }
        write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))

    distributed.barrier()
    distributed.destroy_process_group()
    return passed


def main() -> int:
    try:
        return 0 if run(parse_args()) else 1
    except Exception as error:  # noqa: BLE001 - report rank failures to Slurm
        print(f"DDP smoke failed: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
