#!/usr/bin/env python3
"""Sanity check 6: A2's training cache equals the inference cache.

Runs both cache-building paths on one episode with the given checkpoint. The
``full`` policy must match exactly; other policies report their discrepancy
(expected from D-011's local window when older chunks are compressed).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from distance_decayed_memory.data.shards import SplitReader
from distance_decayed_memory.eval.sanity import streaming_matches_inference
from distance_decayed_memory.memory import Geometry, make_policy
from distance_decayed_memory.models.dit import DiTConfig, PixelDiT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--split-dir", required=True, type=pathlib.Path)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--policy-config", type=json.loads, default=None)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, help="writes streaming_check.json there"
    )
    argv = sys.argv[1:]
    if argv[:1] == ["--args-file"]:  # as written by scripts/hpc/submit_eval.sh
        argv = json.loads(pathlib.Path(argv[1]).read_text())
    args = parser.parse_args(argv)
    if args.output_dir and not args.output:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.output = args.output_dir / "streaming_check.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = DiTConfig(**state["model_config"])
    model = PixelDiT(config)
    model.load_state_dict(state["ema"])
    model.to(device).float().eval()
    episode = SplitReader(args.split_dir).episode(args.episode, ("frames", "actions"))
    frames = torch.from_numpy(np.asarray(episode["frames"][: args.frames]))
    actions = torch.from_numpy(np.asarray(episode["actions"][: args.frames]))
    geometry = Geometry(config.grid)
    results = {}
    policies = {"full": {}}
    if state.get("policy") and state["policy"] != "full":
        policies[state["policy"]] = args.policy_config or state.get("policy_config", {})
    for name, policy_config in policies.items():
        check = streaming_matches_inference(
            model,
            frames,
            actions,
            lambda n=name, c=policy_config: make_policy(n, geometry=geometry, **c),
        )
        results[name] = check.as_dict()
        print(f"{name}: {check.details}")
    passed = results["full"]["passed"]
    if args.output:
        args.output.write_text(
            json.dumps({"passed": passed, "results": results}, indent=2) + "\n"
        )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
