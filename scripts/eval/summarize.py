#!/usr/bin/env python3
"""Aggregate evaluation runs and run the sanity checks (issues #16 and #17).

    python scripts/eval/summarize.py runs/eval-* --output runs/summary.json \
        --equivalence runs/eval-decay-ample runs/eval-full

Groups runs by policy and budget (seeds pooled), reports per-condition means
with 95% bootstrap intervals, memory gain over ``window`` and gap to ``full``
at each budget, and the sanity checks that the given runs allow:

1. window cliff (needs ``window`` and ``full``), 2. oracle is best (needs
``full``), 3. equivalence (``--equivalence``), 4. budgets, 5. memory is used
(runs evaluated with ``--ablation`` next to the same run without it). Check 6
runs on the model: ``scripts/eval/check_streaming.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from distance_decayed_memory.eval import sanity
from distance_decayed_memory.eval.results import (
    load_run,
    reference_gaps,
    summarize,
)

LOCAL_WINDOW_FRAMES = 8  # D-011


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--metric", default="lpips")
    parser.add_argument("--max-visit-age", type=int)
    parser.add_argument("--equivalence", nargs=2, type=pathlib.Path)
    parser.add_argument("--resamples", type=int, default=2_000)
    args = parser.parse_args()

    runs = [load_run(path) for path in args.runs]
    normal = [run for run in runs if not run.info["protocol_config"].get("ablation")]
    ablated = [run for run in runs if run.info["protocol_config"].get("ablation")]
    summary = summarize(
        normal, max_visit_age=args.max_visit_age, resamples=args.resamples
    )
    metric = args.metric
    if not any(metric in value for value in summary.values()):
        metric = "psnr"  # runs evaluated with --no-lpips

    labels = {run.label: run for run in normal}
    full_label = next((label for label in labels if label.startswith("full")), None)
    budgets = sorted(
        {
            run.info["policy_config"].get("budget_tokens")
            for run in normal
            if run.info["policy_config"].get("budget_tokens")
        }
    )
    gaps = {}
    for budget in budgets:
        window = f"window@{budget}"
        gaps[budget] = reference_gaps(
            {
                k: v
                for k, v in summary.items()
                if k.endswith(f"@{budget}") or k == full_label
            },
            metric,
            window if window in summary else None,
            full_label,
        )

    checks = [sanity.budgets(normal)]
    if full_label:
        checks.append(sanity.oracle_is_best(summary, full_label, metric))
        for budget in budgets:
            window = f"window@{budget}"
            if window in summary:
                per_frame = (
                    labels[window].info["policy_config"].get("window", budget // 256)
                )
                checks.append(
                    sanity.window_cliff(
                        summary,
                        window,
                        full_label,
                        per_frame + LOCAL_WINDOW_FRAMES,
                        metric,
                    )
                )
    if args.equivalence:
        checks.append(
            sanity.equivalence(
                load_run(args.equivalence[0]), load_run(args.equivalence[1])
            )
        )
    for run in ablated:
        partner = next(
            (
                other
                for other in normal
                if other.info["checkpoint"] == run.info["checkpoint"]
                and other.label == run.label
                and other.info["protocol"] == run.info["protocol"]
            ),
            None,
        )
        if partner is None:
            continue
        base = summarize([partner], resamples=args.resamples)[partner.label]
        hurt = summarize([run], resamples=args.resamples)[run.label]
        check = sanity.memory_is_used(base, hurt, metric)
        check.details["run"] = str(run.directory)
        checks.append(check)

    result = {
        "metric": metric,
        "runs": [str(run.directory) for run in runs],
        "summary": summary,
        "reference_gaps": gaps,
        "sanity": sanity.summarize_checks(checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
