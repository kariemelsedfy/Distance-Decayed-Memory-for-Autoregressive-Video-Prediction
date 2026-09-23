#!/usr/bin/env python3
"""Validate and benchmark headless Memory Maze rendering on one CPU core."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import statistics
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

EXPECTED_ACTIONS = np.asarray(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=np.float64,
)
ACTION_NAMES = (
    "noop",
    "forward",
    "left",
    "right",
    "forward_left",
    "forward_right",
)
REQUIRED_OBSERVATIONS = {
    "agent_dir": ((2,), np.dtype(np.float64)),
    "agent_pos": ((2,), np.dtype(np.float64)),
    "image": ((64, 64, 3), np.dtype(np.uint8)),
    "maze_layout": ((9, 9), np.dtype(np.uint8)),
    "target_color": ((3,), np.dtype(np.float64)),
    "target_pos": ((2,), np.dtype(np.float64)),
    "target_vec": ((2,), np.dtype(np.float64)),
    "targets_pos": ((3, 2), np.dtype(np.float64)),
    "targets_vec": ((3, 2), np.dtype(np.float64)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-renderer", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0 or args.repeats < 1:
        parser.error(
            "steps and repeats must be positive; warmup steps cannot be negative"
        )
    return args


def package_versions() -> dict[str, str]:
    names = ("memory-maze", "dm-control", "mujoco", "gym", "numpy")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size:
        summary["min"] = float(array.min())
        summary["max"] = float(array.max())
    return summary


def validate_observation(observation: Mapping[str, np.ndarray]) -> None:
    if set(observation) != set(REQUIRED_OBSERVATIONS):
        raise AssertionError(
            "Unexpected observation keys: "
            f"got {sorted(observation)}, expected {sorted(REQUIRED_OBSERVATIONS)}"
        )
    for key, (shape, dtype) in REQUIRED_OBSERVATIONS.items():
        value = np.asarray(observation[key])
        if value.shape != shape:
            raise AssertionError(f"{key} shape is {value.shape}, expected {shape}")
        if value.dtype != dtype:
            raise AssertionError(f"{key} dtype is {value.dtype}, expected {dtype}")

    layout_values = set(np.unique(observation["maze_layout"]).tolist())
    if not layout_values <= {0, 1}:
        raise AssertionError(f"maze_layout is not binary: {sorted(layout_values)}")
    direction_norm = float(np.linalg.norm(observation["agent_dir"]))
    if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"agent_dir is not a unit vector: norm={direction_norm}")


def reset_if_last(env: Any, timestep: Any) -> tuple[Any, int]:
    if timestep.last():
        return env.reset(), 1
    return timestep, 0


def opengl_strings() -> dict[str, str]:
    from OpenGL import GL

    names = {
        "vendor": GL.GL_VENDOR,
        "renderer": GL.GL_RENDERER,
        "version": GL.GL_VERSION,
    }
    values = {}
    for name, enum in names.items():
        raw = GL.glGetString(enum)
        values[name] = raw.decode("utf-8", errors="replace") if raw else "unavailable"
    return values


def main() -> int:
    args = parse_args()
    backend = os.environ.get("MUJOCO_GL")
    if backend != args.expected_renderer:
        raise SystemExit(
            f"Set MUJOCO_GL={args.expected_renderer} before starting the process; "
            f"got {backend!r}. The backend is selected at import time."
        )
    pyopengl_backend = os.environ.get("PYOPENGL_PLATFORM")
    if pyopengl_backend not in (None, args.expected_renderer):
        raise SystemExit(
            "PYOPENGL_PLATFORM conflicts with requested renderer: "
            f"{pyopengl_backend!r}"
        )

    import memory_maze  # noqa: F401 -- import establishes the selected renderer
    from memory_maze import tasks

    started = time.perf_counter()
    env = tasks.memory_maze_9x9(
        global_observables=True,
        image_only_obs=False,
        top_camera=False,
        camera_resolution=64,
        control_freq=4.0,
        discrete_actions=True,
        seed=args.seed,
    )
    environment_creation_seconds = time.perf_counter() - started

    try:
        started = time.perf_counter()
        timestep = env.reset()
        initial_reset_seconds = time.perf_counter() - started
        validate_observation(timestep.observation)

        action_spec = env.action_spec()
        if getattr(action_spec, "num_values", None) != len(EXPECTED_ACTIONS):
            raise AssertionError(f"Unexpected action spec: {action_spec!r}")
        action_vectors = np.asarray(getattr(env, "action_set", ()))
        np.testing.assert_array_equal(action_vectors, EXPECTED_ACTIONS)

        rng = np.random.default_rng(args.seed)
        warmup_actions = rng.integers(0, len(EXPECTED_ACTIONS), args.warmup_steps)
        warmup_resets = 0
        for action in warmup_actions:
            timestep = env.step(int(action))
            timestep, reset_count = reset_if_last(env, timestep)
            warmup_resets += reset_count

        repeat_results = []
        total_resets = 0
        for repeat in range(args.repeats):
            actions = rng.integers(0, len(EXPECTED_ACTIONS), args.steps)
            reset_count = 0
            started = time.perf_counter()
            for action in actions:
                timestep = env.step(int(action))
                timestep, did_reset = reset_if_last(env, timestep)
                reset_count += did_reset
            elapsed = time.perf_counter() - started
            fps = args.steps / elapsed
            total_resets += reset_count
            repeat_results.append(
                {
                    "repeat": repeat,
                    "steps": args.steps,
                    "seconds": elapsed,
                    "frames_per_second": fps,
                    "episode_resets": reset_count,
                }
            )

        validate_observation(timestep.observation)
        fps_values = [item["frames_per_second"] for item in repeat_results]
        observation = timestep.observation
        result = {
            "passed": True,
            "renderer": {
                "mujoco_gl": backend,
                "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM", "unset"),
                "libgl_always_software": os.environ.get(
                    "LIBGL_ALWAYS_SOFTWARE", "unset"
                ),
                "opengl": opengl_strings(),
                "cpu_threads": {
                    name: os.environ.get(name, "unset")
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            },
            "platform": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "system": platform.platform(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
                "slurm_cpus_per_task": os.environ.get(
                    "SLURM_CPUS_PER_TASK", "not-under-slurm"
                ),
            },
            "packages": package_versions(),
            "environment": {
                "name": "Memory Maze 9x9 with global observables",
                "seed": args.seed,
                "control_frequency_hz": 4.0,
                "environment_creation_seconds": environment_creation_seconds,
                "initial_reset_seconds": initial_reset_seconds,
            },
            "observations": {
                key: array_summary(observation[key]) for key in sorted(observation)
            },
            "pose_example": {
                "agent_pos": np.asarray(observation["agent_pos"]).tolist(),
                "agent_dir": np.asarray(observation["agent_dir"]).tolist(),
                "heading_radians": math.atan2(
                    float(observation["agent_dir"][1]),
                    float(observation["agent_dir"][0]),
                ),
            },
            "actions": {
                "count": len(EXPECTED_ACTIONS),
                "names": list(ACTION_NAMES),
                "vectors": action_vectors.tolist(),
            },
            "benchmark": {
                "warmup_steps": args.warmup_steps,
                "warmup_resets": warmup_resets,
                "steps_per_repeat": args.steps,
                "repeats": repeat_results,
                "frames_per_second_mean": statistics.fmean(fps_values),
                "frames_per_second_median": statistics.median(fps_values),
                "frames_per_second_min": min(fps_values),
                "frames_per_second_max": max(fps_values),
                "episode_resets": total_resets,
            },
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
