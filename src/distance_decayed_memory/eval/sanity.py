"""The six Track A sanity checks (TRACK_A_PLAN.md §10).

Checks 1–5 read evaluation summaries or runs; check 6 runs the model. Each
returns a :class:`Check` with a pass flag and the numbers behind it, so a
failure explains itself. A result from the A3 sweep is trusted only when all
six pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from distance_decayed_memory.data.revisit_detector import bucket_label
from distance_decayed_memory.eval.results import LOWER_IS_BETTER, Run
from distance_decayed_memory.models.dit import (
    ModelCache,
    PixelDiT,
    conditioning_actions,
)
from distance_decayed_memory.models.flow import (
    StreamingCache,
    encode_chunk,
    to_model_range,
)


@dataclass
class Check:
    name: str
    passed: bool
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "details": self.details}


def _bucket_bounds(label: str) -> tuple[int, int]:
    low, high = label.strip("[)").split(",")
    return int(low), int(high)


def _worse(metric: str, a: float, b: float) -> float:
    """How much worse ``a`` is than ``b`` (positive = worse)."""
    return a - b if LOWER_IS_BETTER[metric] else b - a


def window_cliff(
    summary: dict,
    window_label: str,
    full_label: str,
    visible_frames: int,
    metric: str = "lpips",
    tolerance: float = 0.25,
) -> Check:
    """1. ``window`` ≈ ``full`` below its reach, ≈ novel-view error beyond it.

    ``visible_frames`` is everything the window policy can see: its own window
    plus the shared local window (D-011). Buckets entirely below that reach
    must sit within ``tolerance`` of the full-minus-novel span of ``full``;
    buckets entirely beyond it must sit within ``tolerance`` of novel error.
    """
    window, full = summary[window_label][metric], summary[full_label][metric]
    novel = window["novel"]["mean"]
    rows, passed = {}, True
    for condition, value in window.items():
        if condition == "novel" or condition not in full:
            continue
        low, high = _bucket_bounds(condition)
        span = abs(novel - full[condition]["mean"]) or 1e-12
        if high <= visible_frames:
            distance = abs(value["mean"] - full[condition]["mean"]) / span
            side = "inside"
        elif low >= visible_frames:
            distance = abs(value["mean"] - novel) / span
            side = "beyond"
        else:
            continue
        ok = distance <= tolerance
        passed &= ok
        rows[condition] = {"side": side, "relative_distance": distance, "ok": ok}
    passed &= any(r["side"] == "inside" for r in rows.values()) or not rows
    passed &= any(r["side"] == "beyond" for r in rows.values())
    return Check(
        "window_cliff",
        bool(passed),
        {"buckets": rows, "visible_frames": visible_frames},
    )


def oracle_is_best(summary: dict, full_label: str, metric: str = "lpips") -> Check:
    """2. ``full`` matches or beats every budgeted policy in every bucket.

    "Within noise" means the oracle's interval overlaps or lies on the better
    side of the policy's interval.
    """
    full = summary[full_label][metric]
    violations = {}
    for label, metrics in summary.items():
        if label == full_label or metric not in metrics:
            continue
        for condition, value in metrics[metric].items():
            if condition == "novel" or condition not in full:
                continue
            oracle = full[condition]
            if LOWER_IS_BETTER[metric]:
                clearly_better = value["high"] < oracle["low"]
            else:
                clearly_better = value["low"] > oracle["high"]
            if clearly_better:
                violations.setdefault(label, []).append(condition)
    return Check("oracle_is_best", not violations, {"violations": violations})


def equivalence(first: Run, second: Run, atol: float = 1e-3) -> Check:
    """3. ``decay_continuous`` with an ample budget reproduces ``full`` exactly."""
    a, b = first.records, second.records
    same_rows = (
        len(a["frame"]) == len(b["frame"])
        and np.array_equal(a["frame"], b["frame"])
        and np.array_equal(a["episode"], b["episode"])
    )
    details: dict = {"same_rows": bool(same_rows)}
    passed = bool(same_rows)
    if same_rows:
        for metric in ("psnr", "ssim", "lpips"):
            if metric in a and np.isfinite(a[metric]).any():
                gap = float(np.nanmax(np.abs(a[metric] - b[metric])))
                details[f"max_abs_{metric}_difference"] = gap
                passed &= gap <= atol * (100 if metric == "psnr" else 1)
    return Check("equivalence", passed, details)


def budgets(runs: list[Run], min_fill: float = 0.8) -> Check:
    """4. No policy exceeds its budget, and all fill it comparably.

    The per-step budget assertion lives in the A2 trainer; this checks the
    cache statistics every evaluation run records. ``min_fill`` flags a policy
    that leaves much of its budget unused, which would make it look worse for
    a reason unrelated to its allocation.
    """
    rows, passed = {}, True
    for run in runs:
        budget = run.info.get("policy_config", {}).get("budget_tokens")
        if budget is None:
            continue
        stats = run.info.get("cache", {})
        maximum = stats.get("max_policy_tokens", 0)
        fill = stats.get("mean_policy_tokens", 0) / budget
        ok = maximum <= budget
        passed &= ok
        rows[run.label] = {
            "budget": budget,
            "max_tokens": maximum,
            "mean_fill": fill,
            "within_budget": ok,
            "underfilled": fill < min_fill,
        }
    return Check("budgets", bool(passed), {"runs": rows})


def memory_is_used(
    normal: dict,
    ablated: dict,
    metric: str = "lpips",
    min_revisit_harm: float = 0.0,
    max_novel_change: float = 0.5,
) -> Check:
    """5. Removing old cache entries hurts revisits but barely changes novel views.

    ``normal`` and ``ablated`` are one label's ``summary[label]`` for the same
    run with and without ``--ablation drop`` (or ``shuffle``). Revisit harm must
    be positive with its interval above zero in at least one bucket; the novel-
    view change must be under ``max_novel_change`` of the largest revisit harm.
    """
    base, hurt = normal[metric], ablated[metric]
    harms = {}
    for condition in base:
        if condition == "novel" or condition not in hurt:
            continue
        harms[condition] = {
            "harm": _worse(metric, hurt[condition]["mean"], base[condition]["mean"]),
            "significant": (
                _worse(metric, hurt[condition]["low"], base[condition]["high"])
                > min_revisit_harm
                if LOWER_IS_BETTER[metric]
                else _worse(metric, hurt[condition]["high"], base[condition]["low"])
                > min_revisit_harm
            ),
        }
    largest = max((h["harm"] for h in harms.values()), default=0.0)
    novel_change = abs(hurt["novel"]["mean"] - base["novel"]["mean"])
    passed = (
        any(h["significant"] for h in harms.values())
        and largest > 0
        and novel_change <= max_novel_change * largest
    )
    return Check(
        "memory_is_used",
        bool(passed),
        {"revisit_harm": harms, "novel_change": novel_change},
    )


@torch.no_grad()
def streaming_matches_inference(
    model: PixelDiT,
    frames: torch.Tensor,
    actions: torch.Tensor,
    make_policy,
    local_chunks: int = 2,
) -> Check:
    """6. The cache A2 training builds equals the one inference builds.

    Inference encodes each chunk once when it is newest (``encode_chunk``
    against the current cache). A2 recomputes the local chunks in a forward
    over ``[local chunks | target]`` against the policy cache only (D-011).
    With no noise augmentation and the ``full`` policy, both must give the
    same keys for every chunk; other policies report their discrepancy.

    ``frames`` is ``[T, H, W, C]`` uint8, ``actions`` ``[T]``.
    """
    config = model.config
    chunk = config.chunk_frames
    parameter = next(model.parameters())
    device = parameter.device
    count = frames.shape[0] // chunk
    video = to_model_range(frames[: count * chunk].to(device))[None].to(parameter)
    prev = conditioning_actions(
        actions[None, : count * chunk].long(), torch.tensor([config.no_action])
    ).to(device)

    inference = StreamingCache([make_policy()], local_chunks)
    reference = []
    for index in range(count):
        span = slice(index * chunk, (index + 1) * chunk)
        k, _ = encode_chunk(
            model, video[:, span], prev[:, span], index * chunk, inference.model_cache()
        )
        inference.push(k, _, index * chunk)
        reference.append(k[0])

    training_policy = make_policy()
    differences = []
    for position in range(local_chunks, count):
        first = (position - local_chunks) * chunk
        stop = (position + 1) * chunk
        zero = torch.zeros(1, stop - first, device=device, dtype=parameter.dtype)
        cache = ModelCache.from_policies([training_policy])
        _, keys, values = model(
            video[:, first:stop],
            zero,
            prev[:, first:stop],
            first,
            cache,
            return_kv=True,
        )
        per_chunk = chunk * config.tokens_per_frame
        k_first, v_first = keys[0, ..., :per_chunk, :], values[0, ..., :per_chunk, :]
        stored = reference[position - local_chunks]
        differences.append(
            float((k_first - stored).norm() / stored.norm().clamp_min(1e-12))
        )
        training_policy.append(k_first, v_first, None, first)
        training_policy.compact()
    worst = max(differences, default=0.0)
    return Check(
        "streaming_matches_inference",
        worst <= 1e-4,
        {"max_relative_key_difference": worst, "chunks_compared": len(differences)},
    )


def summarize_checks(checks: list[Check]) -> dict:
    return {
        "all_passed": all(check.passed for check in checks),
        "checks": [check.as_dict() for check in checks],
    }


def bucket_name(index: int) -> str:
    return bucket_label(index)
