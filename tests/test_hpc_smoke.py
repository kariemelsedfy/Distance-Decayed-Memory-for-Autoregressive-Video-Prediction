from __future__ import annotations

import ast
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


def load_checkpoint_smoke():
    path = ROOT / "scripts" / "hpc" / "checkpoint_smoke.py"
    spec = importlib.util.spec_from_file_location("checkpoint_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sigusr1_handler_is_registered_before_torch_is_imported() -> None:
    """A late handler lets the preemption signal kill the process outright.

    ``main`` must install it first; ``torch`` is imported later, inside ``run``.
    """
    source = (ROOT / "scripts" / "hpc" / "checkpoint_smoke.py").read_text()
    tree = ast.parse(source)
    top_level_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "torch" not in top_level_imports

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first = main.body[0]
    assert isinstance(first, ast.Expr)
    assert ast.unparse(first.value).startswith("signal.signal(signal.SIGUSR1")


def test_request_stop_sets_the_flag() -> None:
    module = load_checkpoint_smoke()
    assert module.STOP_REQUESTED is False
    module.request_stop(10, None)
    assert module.STOP_REQUESTED is True


def test_write_json_atomic_leaves_no_temporary_file(tmp_path: Path) -> None:
    module = load_checkpoint_smoke()
    target = tmp_path / "results.json"
    module.write_json_atomic(target, {"passed": True})
    assert target.read_text().endswith("\n")
    assert list(tmp_path.iterdir()) == [target]
