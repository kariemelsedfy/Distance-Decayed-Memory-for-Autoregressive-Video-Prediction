#!/usr/bin/env python3
"""Write and verify a split's ``manifest.json`` once all its shards exist."""

from __future__ import annotations

import argparse
import json
import pathlib

from distance_decayed_memory.data.shards import build_manifest, verify_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=pathlib.Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--episodes-per-shard", type=int, required=True)
    args = parser.parse_args()

    shards = sorted(args.split_dir.glob("shard-*[0-9]"))
    if len(shards) != args.expected_shards:
        raise SystemExit(
            f"found {len(shards)} finished shards, expected {args.expected_shards}"
        )
    records = [json.loads((s / "shard.json").read_text()) for s in shards]
    manifest = build_manifest(
        args.split_dir,
        {
            "split": args.split,
            "first_seed": args.first_seed,
            "episodes_per_shard": args.episodes_per_shard,
            "maze_size": records[0]["maze_size"],
            "frames_per_episode": records[0]["frames_per_episode"],
            "detector": records[0]["detector"],
            "generator_git": sorted({r["generator_git"]["sha"] for r in records}),
        },
    )
    problems = verify_split(args.split_dir)
    if problems:
        raise SystemExit(f"verification failed: {problems[:10]}")
    digest = (args.split_dir / "manifest.sha256").read_text().split()[0]
    print(
        json.dumps(
            {
                "split": args.split,
                "episodes": manifest["episodes"],
                "frames": manifest["frames"],
                "manifest_sha256": digest,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
