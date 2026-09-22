#!/usr/bin/env python3
"""Create the Track A issue set from docs/TRACK_A_PLAN.md section 12."""

# The exact issue wording is intentionally kept as readable single-line strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Issue:
    title: str
    labels: tuple[str, ...]
    scope: str
    dependencies: tuple[int, ...] = ()


LABELS = {
    "phase-0": ("1d76db", "Phase 0 foundations"),
    "phase-1": ("5319e7", "Phase 1 memory policy library"),
    "phase-2": ("a2eeef", "Phase 2 Track A experiments"),
    "track-a": ("0e8a16", "Active controlled Memory Maze track"),
    "track-b": ("c5def5", "Deferred scaled-video track"),
    "infra": ("d4c5f9", "Infrastructure and developer workflow"),
    "paper": ("f9d0c4", "Paper and related-work tasks"),
}

ISSUES = (
    Issue(
        "Repo scaffold, CI, pre-commit, gitignore, AGENTS.md, and CLAUDE.md",
        ("phase-0", "infra"),
        "Build the repository and documentation scaffold specified by the project plan.",
    ),
    Issue(
        "HPC scripts and Claude skills",
        ("phase-0", "infra"),
        "Implement hpc-run, status, sync, submit, monitor, fetch, env, and handoff workflows.",
    ),
    Issue(
        "Verify pro6000 layout, wall time, and compute-node internet",
        ("phase-0", "infra"),
        "Audit live cluster resources and update docs/hpc/bowdoin-hpc.md.",
    ),
    Issue(
        "Build scratch conda environment and check Blackwell kernels",
        ("phase-0", "infra"),
        "Create the Python 3.11 environment in scratch and verify SDPA, FlexAttention, and bf16 on sm_120.",
        (2, 3),
    ),
    Issue(
        "DDP smoke test and checkpoint/requeue test",
        ("phase-0", "infra"),
        "Test 1 GPU, maximum GPUs per node, multi-node all-reduce, auto-resume, and Slurm requeue.",
        (4,),
    ),
    Issue(
        "Refresh literature on graded causal video memory",
        ("phase-0", "paper"),
        "Search post-mid-2026 work, summarize overlaps in docs/related_work.md, and flag claim risks.",
    ),
    Issue(
        "Install Memory Maze headlessly and verify observations",
        ("phase-2", "track-a"),
        "Run on main, measure rendering speed, and confirm layout, pose, heading, and action observations.",
    ),
    Issue(
        "Implement scripted revisit trajectory generator",
        ("phase-2", "track-a"),
        "Add A* navigation, log-uniform revisit gaps, anti-shortcut mixing, and toy-grid unit tests.",
        (7,),
    ),
    Issue(
        "Implement revisit detector and gap bucketing",
        ("phase-2", "track-a"),
        "Detect pose-matched revisits independently of the script and add visual spot-check tooling.",
        (8,),
    ),
    Issue(
        "Implement sharded dataset writer and staged loader",
        ("phase-2", "track-a"),
        "Write/load episode shards, stage them on node-local storage, and generate the pilot split.",
        (9,),
    ),
    Issue(
        "Implement MemoryPolicy library and tests",
        ("phase-1", "track-a"),
        "Implement every policy and the budget, horizon, aging, equivalence, RoPE, and stats tests from Phase 1.",
    ),
    Issue(
        "Implement pixel DiT model, flow loss, and sampler",
        ("phase-2", "track-a"),
        "Add patchification, 3D RoPE, adaLN action conditioning, block-causal attention, flow matching, and sampling.",
    ),
    Issue(
        "Implement A1 DDP trainer and size scaling check",
        ("phase-2", "track-a"),
        "Add auto-resuming A1 training and compare S/M/L enough to choose the shared base model.",
        (10, 12),
    ),
    Issue(
        "Implement A2 streaming memory trainer",
        ("phase-2", "track-a"),
        "Add parallel live caches, a two-chunk gradient window, auto-resume, and the staleness diagnostic.",
        (11, 13),
    ),
    Issue(
        "Build P1/P2 evaluation and headline plotting harness",
        ("phase-2", "track-a"),
        "Compute LPIPS/PSNR/SSIM, bootstrap confidence intervals, efficiency metrics, and headline plots.",
        (14,),
    ),
    Issue(
        "Build runnable Track A sanity-check suite",
        ("phase-2", "track-a"),
        "Automate the six mandatory checks in TRACK_A_PLAN.md section 10.",
        (14,),
    ),
    Issue(
        "Generate full splits and freeze the Track A test manifest",
        ("phase-2", "track-a"),
        "Generate train/validation/test data, freeze the test manifest, and record its SHA-256.",
        (10,),
    ),
)


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True).strip()


def issue_body(issue: Issue, numbers: dict[int, int]) -> str:
    if issue.dependencies:
        dependencies = "\n".join(
            f"- Depends on #{numbers[index]}" for index in issue.dependencies
        )
    else:
        dependencies = "- None"
    return (
        "## Scope\n\n"
        f"{issue.scope}\n\n"
        "## Dependencies\n\n"
        f"{dependencies}\n\n"
        "## Source\n\n"
        "`docs/TRACK_A_PLAN.md` §12. Track B remains deferred.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="GitHub OWNER/REPO")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        for index, issue in enumerate(ISSUES, start=1):
            deps = ", ".join(map(str, issue.dependencies)) or "none"
            print(
                f"{index:02d}. {issue.title} [{', '.join(issue.labels)}]; deps={deps}"
            )
        return 0

    for name, (color, description) in LABELS.items():
        subprocess.run(
            [
                "gh",
                "label",
                "create",
                name,
                "--repo",
                args.repo,
                "--color",
                color,
                "--description",
                description,
                "--force",
            ],
            check=True,
        )

    existing_data = json.loads(
        gh(
            "issue",
            "list",
            "--repo",
            args.repo,
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,title",
        )
    )
    existing = {item["title"]: item["number"] for item in existing_data}
    numbers: dict[int, int] = {}

    for index, issue in enumerate(ISSUES, start=1):
        if issue.title in existing:
            numbers[index] = existing[issue.title]
            print(f"exists #{numbers[index]}: {issue.title}")
            continue
        command = [
            "issue",
            "create",
            "--repo",
            args.repo,
            "--title",
            issue.title,
            "--body",
            issue_body(issue, numbers),
        ]
        for label in issue.labels:
            command.extend(("--label", label))
        url = gh(*command)
        numbers[index] = int(url.rstrip("/").rsplit("/", 1)[-1])
        print(f"created #{numbers[index]}: {issue.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
