"""Sharded episode storage, split manifests, node-local staging, and clip loading.

A split directory holds ``shard-NNNNN/`` directories and a ``manifest.json``.
Each shard concatenates whole episodes along the frame axis:

- ``frames.npy`` uint8 ``[F, 64, 64, 3]``, ``actions.npy`` int8 ``[F]``,
  ``pose.npy`` float32 ``[F, 3]``;
- revisit labels from :mod:`revisit_detector`: ``revisit_kind.npy`` int8,
  ``revisit_gap.npy`` int32, ``revisit_source.npy`` int32 (frame index within
  the same episode), and ``visit_age.npy`` int32;
- ``episodes.json``: per-episode offset, length, and generation metadata;
- ``shard.json``: shapes, dtypes, and SHA-256 of every file.

Arrays are standard ``.npy`` files opened with ``mmap_mode="r"``, so random
clip access reads only the bytes it needs. Shards are written into a
``.partial`` directory and renamed when complete, so a finished shard is never
half-written.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import pathlib
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from distance_decayed_memory.data.revisit_detector import (
    DetectorConfig,
    detect_revisits,
)

FORMAT_VERSION = 1
ARRAYS: dict[str, tuple[tuple[int, ...], str]] = {
    "frames": ((64, 64, 3), "uint8"),
    "actions": ((), "int8"),
    "pose": ((3,), "float32"),
    "revisit_kind": ((), "int8"),
    "revisit_gap": ((), "int32"),
    "revisit_source": ((), "int32"),
    "visit_age": ((), "int32"),
}


def shard_name(index: int) -> str:
    return f"shard-{index:05d}"


def sha256_file(path: pathlib.Path, block: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def detector_record(config: DetectorConfig) -> dict[str, float | int]:
    record = asdict(config)
    record["heading_tolerance_degrees"] = math.degrees(record.pop("heading_tolerance"))
    return record


class ShardWriter:
    """Append a known number of frames, then :meth:`finish` to publish the shard."""

    def __init__(
        self,
        directory: pathlib.Path,
        total_frames: int,
        info: dict[str, Any] | None = None,
        detector: DetectorConfig | None = None,
    ) -> None:
        self.directory = pathlib.Path(directory)
        self.partial = self.directory.with_name(self.directory.name + ".partial")
        if self.partial.exists():
            shutil.rmtree(self.partial)
        self.partial.mkdir(parents=True)
        self.total_frames = total_frames
        self.info = dict(info or {})
        self.detector = detector or DetectorConfig()
        self.arrays = {
            name: np.lib.format.open_memmap(
                self.partial / f"{name}.npy",
                mode="w+",
                dtype=np.dtype(dtype),
                shape=(total_frames, *shape),
            )
            for name, (shape, dtype) in ARRAYS.items()
        }
        self.episodes: list[dict[str, Any]] = []
        self.cursor = 0

    def add(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        poses: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        length = len(frames)
        if not len(actions) == len(poses) == length:
            raise ValueError("frames, actions, and poses must have equal length")
        if self.cursor + length > self.total_frames:
            raise ValueError("shard capacity exceeded")
        labels = detect_revisits(poses, self.detector)
        stop = self.cursor + length
        values = {
            "frames": frames,
            "actions": actions,
            "pose": poses,
            "revisit_kind": labels.kind,
            "revisit_gap": labels.gap,
            "revisit_source": labels.source,
            "visit_age": labels.visit_age,
        }
        for name, value in values.items():
            self.arrays[name][self.cursor : stop] = value
        self.episodes.append({**meta, "offset": self.cursor, "length": length})
        self.cursor = stop

    def finish(self) -> dict[str, Any]:
        if self.cursor != self.total_frames:
            raise ValueError(
                f"shard holds {self.cursor} of {self.total_frames} planned frames"
            )
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        _write_json(self.partial / "episodes.json", self.episodes)
        files = {}
        for path in sorted(self.partial.iterdir()):
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        record = {
            **self.info,
            "format_version": FORMAT_VERSION,
            "name": self.directory.name,
            "episodes": len(self.episodes),
            "frames": self.total_frames,
            "episode_ids": [episode["episode_id"] for episode in self.episodes],
            "detector": detector_record(self.detector),
            "arrays": {
                name: {"shape": [self.total_frames, *shape], "dtype": dtype}
                for name, (shape, dtype) in ARRAYS.items()
            },
            "files": files,
        }
        _write_json(self.partial / "shard.json", record)
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.partial.rename(self.directory)
        return record


def is_complete_shard(directory: pathlib.Path) -> bool:
    return (pathlib.Path(directory) / "shard.json").is_file()


def build_manifest(split_dir: pathlib.Path, info: dict[str, Any]) -> dict[str, Any]:
    """Collect every finished shard into ``manifest.json`` and hash it."""
    split_dir = pathlib.Path(split_dir)
    partial = sorted(split_dir.glob("shard-*.partial"))
    if partial:
        raise ValueError(f"unfinished shards: {[p.name for p in partial]}")
    shards = []
    episode_ids: list[str] = []
    for directory in sorted(split_dir.glob("shard-*")):
        record = json.loads((directory / "shard.json").read_text())
        episode_ids.extend(record["episode_ids"])
        shards.append(
            {
                "name": record["name"],
                "episodes": record["episodes"],
                "frames": record["frames"],
                "files": record["files"],
            }
        )
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode ids across shards")
    manifest = {
        **info,
        "format_version": FORMAT_VERSION,
        "shards": shards,
        "episodes": len(episode_ids),
        "frames": sum(shard["frames"] for shard in shards),
    }
    _write_json(split_dir / "manifest.json", manifest)
    digest = sha256_file(split_dir / "manifest.json")
    (split_dir / "manifest.sha256").write_text(f"{digest}  manifest.json\n")
    return manifest


def verify_split(split_dir: pathlib.Path, check_hashes: bool = True) -> list[str]:
    """Return a list of problems; empty means the split matches its manifest."""
    split_dir = pathlib.Path(split_dir)
    problems = []
    manifest = json.loads((split_dir / "manifest.json").read_text())
    for shard in manifest["shards"]:
        for name, expected in shard["files"].items():
            path = split_dir / shard["name"] / name
            if not path.is_file():
                problems.append(f"missing {path}")
            elif path.stat().st_size != expected["bytes"]:
                problems.append(f"size mismatch {path}")
            elif check_hashes and sha256_file(path) != expected["sha256"]:
                problems.append(f"hash mismatch {path}")
    return problems


@contextlib.contextmanager
def _locked(path: pathlib.Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stage_split(
    split_dir: pathlib.Path, destination: pathlib.Path, check_hashes: bool = True
) -> pathlib.Path:
    """Copy a split to node-local storage once per node, then reuse it.

    Safe when several ranks on one node call it at once: the first copies and
    verifies under a file lock, and the others wait and reuse the result. The
    copy is keyed by the manifest hash, so a changed split is copied afresh.
    """
    split_dir = pathlib.Path(split_dir)
    digest = (split_dir / "manifest.sha256").read_text().split()[0]
    target = pathlib.Path(destination) / f"{split_dir.name}-{digest[:16]}"
    marker = target / ".staged"
    with _locked(pathlib.Path(destination) / f".{target.name}.lock"):
        if marker.is_file():
            return target
        if target.exists():
            shutil.rmtree(target)
        partial = target.with_name(target.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        for name in ("manifest.json", "manifest.sha256"):
            shutil.copy2(split_dir / name, partial / name)
        manifest = json.loads((split_dir / "manifest.json").read_text())
        for shard in manifest["shards"]:
            shutil.copytree(split_dir / shard["name"], partial / shard["name"])
        problems = verify_split(partial, check_hashes=check_hashes)
        if problems:
            raise RuntimeError(f"staged copy failed verification: {problems[:5]}")
        partial.rename(target)
        marker.write_text(digest + "\n")
    return target


class Shard:
    def __init__(self, directory: pathlib.Path) -> None:
        self.directory = pathlib.Path(directory)
        self.record = json.loads((self.directory / "shard.json").read_text())
        self.episodes = json.loads((self.directory / "episodes.json").read_text())
        self._arrays: dict[str, np.ndarray] = {}

    def array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.directory / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def episode(self, index: int, names: Sequence[str] = tuple(ARRAYS)) -> dict:
        episode = self.episodes[index]
        start, stop = episode["offset"], episode["offset"] + episode["length"]
        return {name: self.array(name)[start:stop] for name in names} | {
            "meta": episode
        }


class SplitReader:
    """Read-only access to a split's episodes through their shards."""

    def __init__(self, split_dir: pathlib.Path) -> None:
        self.split_dir = pathlib.Path(split_dir)
        self.manifest = json.loads((self.split_dir / "manifest.json").read_text())
        self.shards = [
            Shard(self.split_dir / s["name"]) for s in self.manifest["shards"]
        ]
        self.index = [
            (shard_index, episode_index)
            for shard_index, shard in enumerate(self.shards)
            for episode_index in range(len(shard.episodes))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def episode(self, index: int, names: Sequence[str] = tuple(ARRAYS)) -> dict:
        shard_index, episode_index = self.index[index]
        return self.shards[shard_index].episode(episode_index, names)

    def episode_length(self, index: int) -> int:
        shard_index, episode_index = self.index[index]
        return self.shards[shard_index].episodes[episode_index]["length"]


class ClipDataset:
    """Fixed-length clips for A1 training; works directly with a torch DataLoader.

    Clips start every ``start_stride`` frames (a multiple of the 4-frame chunk
    by default) and never cross an episode boundary. Items are dictionaries of
    NumPy arrays, which the default torch collate turns into tensors.
    """

    def __init__(
        self,
        split_dir: pathlib.Path,
        clip_frames: int = 64,
        start_stride: int = 4,
        names: Sequence[str] = ("frames", "actions", "pose"),
    ) -> None:
        self.reader = SplitReader(split_dir)
        self.clip_frames = clip_frames
        self.names = tuple(names)
        starts = []
        for index in range(len(self.reader)):
            length = self.reader.episode_length(index)
            for start in range(0, length - clip_frames + 1, start_stride):
                starts.append((index, start))
        self.starts = np.asarray(starts, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        episode_index, start = (int(v) for v in self.starts[item])
        episode = self.reader.episode(episode_index, self.names)
        stop = start + self.clip_frames
        clip = {name: np.array(episode[name][start:stop]) for name in self.names}
        clip["episode"] = np.int64(episode_index)
        clip["start"] = np.int64(start)
        return clip


def default_stage_root() -> pathlib.Path:
    """Node-local scratch for staged splits (``$DD_MEMORY_STAGE`` or ``/tmp``)."""
    user = os.environ.get("USER", "user")
    return pathlib.Path(os.environ.get("DD_MEMORY_STAGE", f"/tmp/{user}/dd-memory"))
