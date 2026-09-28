#!/usr/bin/env python3
"""Evaluate a checkpoint under one memory policy with protocol P1 or P2.

    python scripts/eval/evaluate.py --checkpoint runs/a2-.../checkpoints/latest.pt \
        --split-dir data/memmaze9/val --protocol p1 --output-dir runs/eval-...

The policy comes from an A2 checkpoint (which records it) unless ``--policy``
and ``--policy-config`` override it; A1 checkpoints need ``--policy``. The
frozen test split is refused unless ``--final`` is given (D-006). Writes
``records.npz`` and ``run.json`` (see ``distance_decayed_memory.eval.results``).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

from distance_decayed_memory.data.shards import SplitReader
from distance_decayed_memory.eval.metrics import FrameMetrics
from distance_decayed_memory.eval.protocols import (
    ProtocolConfig,
    evaluate_p1,
    evaluate_p2,
)
from distance_decayed_memory.eval.results import save_run
from distance_decayed_memory.memory import Geometry, make_policy
from distance_decayed_memory.models.dit import DiTConfig, PixelDiT


def parse_args() -> argparse.Namespace:
    """Arguments from the command line, or from ``--args-file FILE`` (a JSON list).

    The file form lets batch jobs pass JSON policy configs, whose commas cannot
    travel through ``sbatch --export``.
    """
    argv = sys.argv[1:]
    if argv[:1] == ["--args-file"]:
        argv = json.loads(pathlib.Path(argv[1]).read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--split-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--protocol", choices=("p1", "p2"), default="p1")
    parser.add_argument("--policy")
    parser.add_argument("--policy-config", type=json.loads, default=None)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--episodes", type=int, help="first N episodes only")
    parser.add_argument("--batch", type=int, default=8, help="P2 episodes per batch")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--local-chunks", type=int, default=2)
    parser.add_argument("--p2-frames", type=int, help="P2 rollout length")
    parser.add_argument("--ablation", choices=("drop", "shuffle"))
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--final", action="store_true", help="allow the frozen test split"
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.split_dir / "manifest.json").read_text())
    if manifest.get("split") == "test" and not args.final:
        raise SystemExit("The test split is frozen (D-006); pass --final to use it.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = DiTConfig(**state["model_config"])
    model = PixelDiT(model_config)
    model.load_state_dict(state[args.weights])
    model.to(device).eval()

    policy_name = args.policy or state.get("policy")
    if policy_name is None:
        raise SystemExit("This checkpoint records no policy; pass --policy.")
    policy_config = (
        args.policy_config
        if args.policy_config is not None
        else (state.get("policy_config", {}) if not args.policy else {})
    )
    geometry = Geometry(model_config.grid)

    def build():
        return make_policy(policy_name, geometry=geometry, **policy_config)

    build()  # fail fast on a bad config
    config = ProtocolConfig(
        steps=args.steps,
        local_chunks=args.local_chunks,
        ablation=args.ablation,
        seed=args.seed,
    )
    reader = SplitReader(args.split_dir)
    count = min(args.episodes or len(reader), len(reader))
    names = ("frames", "actions", "revisit_kind", "revisit_gap", "visit_age")
    metrics = FrameMetrics(device, use_lpips=not args.no_lpips)
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"
    )
    started = time.perf_counter()
    if args.protocol == "p1":
        episodes = ((i, reader.episode(i, names)) for i in range(count))
        records, stats = evaluate_p1(
            model, episodes, build, config, metrics, device, autocast
        )
    else:
        parts, stats = [], {"generated_frames": 0, "max_policy_tokens": 0}
        token_means = []
        for first in range(0, count, args.batch):
            batch = [
                (i, reader.episode(i, names))
                for i in range(first, min(first + args.batch, count))
            ]
            part, part_stats = evaluate_p2(
                model,
                batch,
                build,
                config,
                metrics,
                device,
                autocast,
                total_frames=args.p2_frames,
            )
            parts.append(part)
            stats["generated_frames"] += part_stats["generated_frames"]
            stats["max_policy_tokens"] = max(
                stats["max_policy_tokens"], part_stats["max_policy_tokens"]
            )
            token_means.append(part_stats["mean_policy_tokens"])
        stats["mean_policy_tokens"] = float(sum(token_means) / len(token_means))
        records = {
            k: torch.cat([torch.as_tensor(p[k]) for p in parts]).numpy()
            for k in parts[0]
        }
    seconds = time.perf_counter() - started
    run = {
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "train_seed": int(state.get("seed", 0)),
        "split": manifest.get("split"),
        "split_manifest": str(args.split_dir / "manifest.json"),
        "protocol": args.protocol,
        "policy": policy_name,
        "policy_config": policy_config,
        "protocol_config": config.__dict__,
        "episodes": count,
        "cache": stats,
        "seconds": seconds,
        "seconds_per_generated_frame": seconds / max(1, stats["generated_frames"]),
        "git_sha": os.environ.get("DD_MEMORY_GIT_SHA", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    save_run(args.output_dir, records, run)
    print(json.dumps({k: v for k, v in run.items() if k != "protocol_config"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
