"""Storing and aggregating per-frame evaluation results (TRACK_A_PLAN.md §7).

An evaluation run writes ``records.npz`` (one row per scored frame, columns
from :data:`protocols.RECORD_FIELDS`) and ``run.json`` (policy, budget, seed,
checkpoint, protocol, cache statistics). Aggregation groups rows into
*conditions* — each revisit gap bucket, plus novel views — and reports means
with 95% bootstrap intervals. Resampling is over (seed, episode) clusters, so
frames of one episode are never treated as independent.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from distance_decayed_memory.data.revisit_detector import (
    NOVEL,
    REVISIT,
    bucket_label,
)

METRICS = ("lpips", "psnr", "ssim")
LOWER_IS_BETTER = {"lpips": True, "psnr": False, "ssim": False}


def save_run(
    directory: pathlib.Path, records: dict[str, np.ndarray], run: dict
) -> None:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "records.npz", **records)
    (directory / "run.json").write_text(json.dumps(run, indent=2) + "\n")


@dataclass
class Run:
    directory: pathlib.Path
    records: dict[str, np.ndarray]
    info: dict

    @property
    def label(self) -> str:
        policy = self.info["policy"]
        budget = self.info.get("policy_config", {}).get("budget_tokens")
        return policy if budget is None else f"{policy}@{budget}"

    @property
    def seed(self) -> int:
        return int(self.info.get("train_seed", 0))


def load_run(directory: pathlib.Path) -> Run:
    directory = pathlib.Path(directory)
    with np.load(directory / "records.npz") as data:
        records = {name: data[name] for name in data.files}
    info = json.loads((directory / "run.json").read_text())
    return Run(directory, records, info)


def condition_masks(
    records: dict[str, np.ndarray], max_visit_age: int | None = None
) -> dict[str, np.ndarray]:
    """Boolean masks for ``novel`` and every populated revisit bucket."""
    kind = records["kind"]
    masks = {"novel": kind == NOVEL}
    revisit = kind == REVISIT
    if max_visit_age is not None:
        revisit &= records["visit_age"] <= max_visit_age
    for bucket in np.unique(records["bucket"][revisit]).astype(int):
        masks[bucket_label(bucket)] = revisit & (records["bucket"] == bucket)
    return masks


def _cluster_means(
    values: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sums and counts, so resampled means weight frames equally."""
    keys, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys)).astype(float)
    return sums, counts


def bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    resamples: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Frame-weighted mean with a 95% percentile interval over clusters."""
    finite = np.isfinite(values)
    values, clusters = values[finite], clusters[finite]
    if not len(values):
        return {
            "mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "frames": 0,
            "clusters": 0,
        }
    sums, counts = _cluster_means(values, clusters)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "frames": int(len(values)),
        "clusters": int(len(sums)),
    }


def _clusters(run: Run) -> np.ndarray:
    return run.seed * 10_000_000 + run.records["episode"].astype(np.int64)


def summarize(
    runs: Sequence[Run],
    metrics: Sequence[str] = METRICS,
    max_visit_age: int | None = None,
    resamples: int = 2_000,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """``summary[label][metric][condition] = {mean, low, high, frames, clusters}``.

    Runs sharing a label (the same policy and budget, different seeds) are
    pooled; their episodes form separate clusters.
    """
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.label, []).append(run)
    summary: dict = {}
    for label, members in grouped.items():
        records = {
            name: np.concatenate([m.records[name] for m in members])
            for name in members[0].records
        }
        clusters = np.concatenate([_clusters(m) for m in members])
        masks = condition_masks(records, max_visit_age)
        summary[label] = {}
        for metric in metrics:
            if metric not in records or not np.isfinite(records[metric]).any():
                continue
            summary[label][metric] = {
                condition: bootstrap_mean(
                    records[metric][mask], clusters[mask], resamples
                )
                for condition, mask in masks.items()
            }
    return summary


def paired_difference(
    first: Sequence[Run],
    second: Sequence[Run],
    metric: str = "lpips",
    max_visit_age: int | None = None,
    resamples: int = 2_000,
) -> dict[str, dict[str, float]]:
    """``first − second`` per condition, paired on the same (episode, frame).

    Seeds are averaged within each run group first, then differences are
    resampled over episodes. Negative means ``first`` is lower (better for
    LPIPS).
    """

    def per_frame(runs: Sequence[Run]):
        """Seed-averaged value per (episode, frame) key, plus that frame's labels."""
        columns = {
            name: np.concatenate([run.records[name] for run in runs])
            for name in ("episode", "frame", "kind", "bucket", "visit_age", metric)
        }
        keys = columns["episode"].astype(np.int64) * 1_000_000 + columns["frame"]
        unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        sums = np.bincount(inverse, weights=columns[metric], minlength=len(unique))
        counts = np.bincount(inverse, minlength=len(unique))
        labels = {
            name: columns[name][first]
            for name in ("episode", "kind", "bucket", "visit_age")
        }
        return unique, sums / counts, labels

    keys_a, values_a, labels = per_frame(first)
    keys_b, values_b, _ = per_frame(second)
    shared, index_a, index_b = np.intersect1d(keys_a, keys_b, return_indices=True)
    if not len(shared):
        return {}
    records = {name: value[index_a] for name, value in labels.items()}
    difference = values_a[index_a] - values_b[index_b]
    masks = condition_masks(records, max_visit_age)
    return {
        condition: bootstrap_mean(difference[mask], records["episode"][mask], resamples)
        for condition, mask in masks.items()
    }


def reference_gaps(
    summary: dict,
    metric: str = "lpips",
    window: str | None = None,
    oracle: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Memory gain (window − policy) and oracle gap (policy − full) per condition.

    Signs follow LPIPS (lower is better): a positive memory gain and a small
    oracle gap are good. Only means are reported; use :func:`paired_difference`
    for intervals.
    """
    result: dict = {}
    for label, metrics in summary.items():
        if metric not in metrics:
            continue
        entry = {}
        for condition, value in metrics[metric].items():
            item = {}
            if window and window in summary and condition in summary[window][metric]:
                item["memory_gain"] = (
                    summary[window][metric][condition]["mean"] - value["mean"]
                )
            if oracle and oracle in summary and condition in summary[oracle][metric]:
                item["oracle_gap"] = (
                    value["mean"] - summary[oracle][metric][condition]["mean"]
                )
            entry[condition] = item
        result[label] = entry
    return result
