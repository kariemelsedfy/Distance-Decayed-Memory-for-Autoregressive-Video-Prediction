from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_ddp_smoke():
    path = ROOT / "scripts" / "hpc" / "ddp_smoke.py"
    spec = importlib.util.spec_from_file_location("ddp_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("algorithm_bandwidth", "world_size", "expected"),
    [(100.0, 1, 0.0), (100.0, 2, 100.0), (100.0, 4, 150.0), (70.0, 7, 120.0)],
)
def test_effective_bus_bandwidth(
    algorithm_bandwidth: float, world_size: int, expected: float
) -> None:
    module = load_ddp_smoke()
    assert module.effective_bus_bandwidth_gbps(
        algorithm_bandwidth, world_size
    ) == pytest.approx(expected)
