from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from distance_decayed_memory.data.revisit_detector import NOVEL, REVISIT  # noqa: E402
from distance_decayed_memory.data.shards import (  # noqa: E402
    ShardWriter,
    build_manifest,
    shard_name,
)
from distance_decayed_memory.eval import sanity  # noqa: E402
from distance_decayed_memory.eval.metrics import psnr, ssim  # noqa: E402
from distance_decayed_memory.eval.protocols import (  # noqa: E402
    ProtocolConfig,
    p1_windows,
)
from distance_decayed_memory.eval.results import (  # noqa: E402
    Run,
    bootstrap_mean,
    condition_masks,
    load_run,
    paired_difference,
    summarize,
)
from distance_decayed_memory.memory import Geometry, make_policy  # noqa: E402
from distance_decayed_memory.models.dit import DiTConfig, PixelDiT  # noqa: E402
from distance_decayed_memory.train import a1  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
TINY = {"layers": 1, "width": 48, "heads": 2, "patch": 16}
PER_FRAME = 16


def load_script(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(monkeypatch, relative: str, *argv: str) -> int:
    module = load_script(relative)
    monkeypatch.setattr(sys, "argv", [relative, *argv])
    return module.main()


def revisit_poses(frames: int) -> np.ndarray:
    """Walk out along a corridor and back twice, so revisits and novel views mix."""
    leg = frames // 4
    out = np.column_stack(
        [np.linspace(0.5, 8.5, leg), np.full(leg, 0.5), np.zeros(leg)]
    )
    back = np.column_stack(
        [np.linspace(8.5, 0.5, leg), np.full(leg, 0.5), np.full(leg, np.pi)]
    )
    return np.concatenate([out, back, out, back]).astype(np.float32)


def make_split(root: pathlib.Path, name: str, episodes: int = 2, frames: int = 64):
    split = root / name
    rng = np.random.default_rng(1)
    writer = ShardWriter(split / shard_name(0), episodes * frames)
    for index in range(episodes):
        meta = {
            "episode_id": f"{name}-{index}",
            "events": [
                {"status": "completed", "return_frame": 36, "anchor_frame": 4},
                {"status": "completed", "return_frame": 40, "anchor_frame": 8},
                {"status": "partial", "return_frame": None},
            ],
        }
        writer.add(
            rng.integers(0, 256, (frames, 64, 64, 3), dtype=np.uint8),
            rng.integers(0, 6, frames).astype(np.int8),
            revisit_poses(frames),
            meta,
        )
    writer.finish()
    build_manifest(split, {"split": name})
    return split


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    val = make_split(root, "val")
    config = a1.A1Config(
        run_dir=str(root / "a1"),
        train_split=str(val),
        val_split=str(val),
        model_overrides=TINY,
        clip_frames=8,
        batch_size=2,
        warmup_steps=1,
        max_steps=2,
        num_workers=0,
        precision="fp32",
        val_every=100,
        sample_every=100,
        log_every=1,
        checkpoint_every=2,
        keep_every=100,
    )
    assert a1.train(config) == 0
    return root, val, root / "a1" / "checkpoints" / "latest.pt"


# -- metrics ------------------------------------------------------------------


def test_psnr_and_ssim_behave() -> None:
    frames = torch.randint(0, 256, (3, 16, 16, 3), dtype=torch.uint8)
    assert torch.all(psnr(frames, frames) > 90)
    torch.testing.assert_close(ssim(frames, frames), torch.ones(3))
    shifted = (frames.int() + 10).clamp(0, 255).to(torch.uint8)
    expected = 10 * math.log10(1 / ((10 / 255) ** 2))
    assert (
        abs(
            float(psnr(torch.full_like(frames, 110), torch.full_like(frames, 100))[0])
            - expected
        )
        < 1e-3
    )
    assert torch.all(ssim(frames, shifted) < 1)
    assert torch.all(ssim(frames, 255 - frames) < ssim(frames, shifted))


# -- aggregation --------------------------------------------------------------


def fake_run(label_policy: str, budget: int | None, values: dict, seed: int = 0):
    """Records where each condition has a fixed metric value per episode."""
    rows = {
        k: []
        for k in (
            "episode",
            "frame",
            "kind",
            "gap",
            "bucket",
            "visit_age",
            "lpips",
            "psnr",
            "ssim",
            "window_start",
        )
    }
    for episode in range(6):
        frame = 0
        for condition, value in values.items():
            kind = NOVEL if condition == "novel" else REVISIT
            bucket = -1 if condition == "novel" else int(condition)
            for _ in range(5):
                rows["episode"].append(episode)
                rows["frame"].append(frame)
                rows["kind"].append(kind)
                rows["gap"].append(-1 if bucket < 0 else 16 * 2**bucket)
                rows["bucket"].append(bucket)
                rows["visit_age"].append(0)
                rows["lpips"].append(value + 0.001 * episode)
                rows["psnr"].append(30 - 10 * value)
                rows["ssim"].append(1 - value)
                rows["window_start"].append(0)
                frame += 1
    records = {k: np.asarray(v) for k, v in rows.items()}
    config = {} if budget is None else {"budget_tokens": budget}
    info = {
        "policy": label_policy,
        "policy_config": config,
        "train_seed": seed,
        "protocol_config": {},
        "checkpoint": "c",
        "protocol": "p1",
        "cache": {
            "max_policy_tokens": budget or 0,
            "mean_policy_tokens": (budget or 0) * 0.95,
        },
    }
    return Run(pathlib.Path(label_policy), records, info)


def test_bootstrap_interval_contains_the_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 0.1, 500)
    clusters = np.repeat(np.arange(50), 10)
    result = bootstrap_mean(values, clusters)
    assert result["low"] < result["mean"] < result["high"]
    assert result["clusters"] == 50 and result["frames"] == 500


def test_summary_pools_seeds_and_splits_conditions() -> None:
    runs = [
        fake_run("decay_continuous", 4096, {"novel": 0.5, "0": 0.1}, seed=s)
        for s in (0, 1)
    ]
    summary = summarize(runs, metrics=("lpips",), resamples=200)
    entry = summary["decay_continuous@4096"]["lpips"]
    assert set(entry) == {"novel", "[16,32)"}
    assert entry["[16,32)"]["clusters"] == 12
    assert abs(entry["novel"]["mean"] - 0.5025) < 1e-6


def test_paired_difference_matches_the_gap() -> None:
    a = [fake_run("decay_continuous", 4096, {"novel": 0.5, "0": 0.1})]
    b = [fake_run("window", 4096, {"novel": 0.5, "0": 0.3})]
    difference = paired_difference(a, b, "lpips", resamples=200)
    assert abs(difference["[16,32)"]["mean"] + 0.2) < 1e-9
    assert abs(difference["novel"]["mean"]) < 1e-9


# -- sanity checks 1-5 on synthetic summaries ------------------------------------


def cliff_summary(window_far: float) -> dict:
    full = fake_run("full", None, {"novel": 0.5, "0": 0.1, "4": 0.1})
    window = fake_run("window", 4096, {"novel": 0.5, "0": 0.12, "4": window_far})
    return summarize([full, window], metrics=("lpips",), resamples=200)


def test_window_cliff_passes_and_fails() -> None:
    # Bucket 0 is [16,32) (inside a 24-frame reach); bucket 4 is [256,512).
    good = sanity.window_cliff(cliff_summary(0.49), "window@4096", "full", 32)
    assert good.passed, good.details
    bad = sanity.window_cliff(cliff_summary(0.12), "window@4096", "full", 32)
    assert not bad.passed


def test_oracle_is_best_flags_a_better_policy() -> None:
    full = fake_run("full", None, {"novel": 0.5, "0": 0.3})
    good = fake_run("decay_continuous", 4096, {"novel": 0.5, "0": 0.35})
    cheat = fake_run("relic_discrete", 4096, {"novel": 0.5, "0": 0.1})
    summary = summarize([full, good, cheat], metrics=("lpips",), resamples=200)
    check = sanity.oracle_is_best(summary, "full")
    assert not check.passed
    assert list(check.details["violations"]) == ["relic_discrete@4096"]


def test_memory_is_used_and_budget_checks() -> None:
    base = summarize(
        [fake_run("decay_continuous", 4096, {"novel": 0.5, "0": 0.1})],
        metrics=("lpips",),
        resamples=200,
    )["decay_continuous@4096"]
    hurt = summarize(
        [fake_run("decay_continuous", 4096, {"novel": 0.51, "0": 0.4})],
        metrics=("lpips",),
        resamples=200,
    )["decay_continuous@4096"]
    assert sanity.memory_is_used(base, hurt).passed
    assert not sanity.memory_is_used(base, base).passed
    over = fake_run("window", 4096, {"novel": 0.5})
    over.info["cache"]["max_policy_tokens"] = 5000
    assert not sanity.budgets([over]).passed
    assert sanity.budgets([fake_run("window", 4096, {"novel": 0.5})]).passed


def test_equivalence_check() -> None:
    a = fake_run("full", None, {"novel": 0.5, "0": 0.1})
    b = fake_run("decay_continuous", 10**9, {"novel": 0.5, "0": 0.1})
    assert sanity.equivalence(a, b).passed
    c = fake_run("decay_continuous", 10**9, {"novel": 0.5, "0": 0.2})
    assert not sanity.equivalence(a, c).passed


# -- protocols end to end -------------------------------------------------------


def test_p1_windows_follow_completed_returns() -> None:
    episode = {
        "frames": np.zeros(64),
        "meta": {
            "events": [
                {"status": "completed", "return_frame": 36},
                {"status": "completed", "return_frame": 37},
                {"status": "timeout", "return_frame": 50},
                {"status": "completed", "return_frame": 60},
            ]
        },
    }
    windows = p1_windows(
        episode, 4, ProtocolConfig(context_before=16, generate_frames=32)
    )
    # 36-16=20 -> start 20; 37 duplicates it; 60 would run past frame 64.
    assert windows == [(20, 36)]


def test_evaluation_pipeline_end_to_end(setup, monkeypatch, tmp_path) -> None:
    root, val, checkpoint = setup
    common = [
        "--checkpoint",
        str(checkpoint),
        "--split-dir",
        str(val),
        "--no-lpips",
        "--precision",
        "fp32",
        "--steps",
        "2",
    ]
    runs = {}
    for name, extra in {
        "full": ["--policy", "full", "--policy-config", "{}"],
        "window": [
            "--policy",
            "window",
            "--policy-config",
            json.dumps({"budget_tokens": 2 * PER_FRAME}),
        ],
        "window_drop": [
            "--policy",
            "window",
            "--policy-config",
            json.dumps({"budget_tokens": 2 * PER_FRAME}),
            "--ablation",
            "drop",
        ],
    }.items():
        out = tmp_path / name
        assert (
            run_main(
                monkeypatch,
                "scripts/eval/evaluate.py",
                *common,
                *extra,
                "--protocol",
                "p1",
                "--output-dir",
                str(out),
            )
            == 0
        )
        runs[name] = load_run(out)
    p1 = runs["window"].records
    # Returns at 36 and 40 give windows starting at 20 and 24 (16 frames
    # earlier, chunk-aligned): 2 episodes x 2 windows x 32 frames.
    assert len(p1["frame"]) == 2 * 2 * 32
    assert set(np.unique(p1["window_start"])) == {20, 24}
    assert np.all(np.isfinite(p1["psnr"])) and np.all(np.isnan(p1["lpips"]))
    assert runs["window"].info["cache"]["max_policy_tokens"] <= 2 * PER_FRAME

    out = tmp_path / "p2"
    assert (
        run_main(
            monkeypatch,
            "scripts/eval/evaluate.py",
            *common,
            "--policy",
            "full",
            "--policy-config",
            "{}",
            "--protocol",
            "p2",
            "--output-dir",
            str(out),
        )
        == 0
    )
    p2 = load_run(out).records
    assert len(p2["frame"]) == 2 * (64 - 16) and p2["frame"].min() == 16

    summary_path = tmp_path / "summary.json"
    assert (
        run_main(
            monkeypatch,
            "scripts/eval/summarize.py",
            str(tmp_path / "full"),
            str(tmp_path / "window"),
            str(tmp_path / "window_drop"),
            "--output",
            str(summary_path),
            "--resamples",
            "100",
        )
        == 0
    )
    summary = json.loads(summary_path.read_text())
    names = {check["name"] for check in summary["sanity"]["checks"]}
    assert {"budgets", "oracle_is_best", "window_cliff", "memory_is_used"} <= names
    assert summary["metric"] == "psnr"

    figure = tmp_path / "fig" / "headline"
    assert (
        run_main(
            monkeypatch,
            "scripts/figures/headline.py",
            str(summary_path),
            "--budget",
            str(2 * PER_FRAME),
            "--output",
            str(figure),
        )
        == 0
    )
    assert pathlib.Path(f"{figure}_{2 * PER_FRAME}.png").is_file()


def test_test_split_is_refused_without_final(setup, monkeypatch, tmp_path) -> None:
    root, _, checkpoint = setup
    test = make_split(tmp_path, "test", episodes=1)
    with pytest.raises(SystemExit, match="frozen"):
        run_main(
            monkeypatch,
            "scripts/eval/evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--split-dir",
            str(test),
            "--output-dir",
            str(tmp_path / "x"),
            "--policy",
            "full",
            "--no-lpips",
        )


def test_streaming_matches_inference_exactly_with_full_cache() -> None:
    torch.manual_seed(0)
    config = DiTConfig(
        layers=2, width=48, heads=2, image_size=16, patch=4, chunk_frames=2
    )
    model = PixelDiT(config).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    frames = torch.randint(0, 256, (16, 16, 16, 3), dtype=torch.uint8)
    actions = torch.randint(0, 6, (16,))
    full = sanity.streaming_matches_inference(
        model, frames, actions, lambda: make_policy("full", geometry=Geometry(4))
    )
    assert full.passed and full.details["chunks_compared"] == 6
    window = sanity.streaming_matches_inference(
        model,
        frames,
        actions,
        lambda: make_policy("window", budget_tokens=32, geometry=Geometry(4)),
    )
    assert window.details["max_relative_key_difference"] > 1e-6


def test_condition_masks_respect_visit_age() -> None:
    records = {
        "kind": np.array([NOVEL, REVISIT, REVISIT]),
        "bucket": np.array([-1, 0, 0]),
        "visit_age": np.array([0, 0, 30]),
    }
    assert condition_masks(records)["[16,32)"].sum() == 2
    assert condition_masks(records, max_visit_age=16)["[16,32)"].sum() == 1


def test_evaluate_reads_an_arguments_file(setup, monkeypatch, tmp_path) -> None:
    _, val, checkpoint = setup
    out = tmp_path / "from-file"
    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--split-dir",
        str(val),
        "--no-lpips",
        "--precision",
        "fp32",
        "--steps",
        "1",
        "--policy",
        "decay_continuous",
        "--policy-config",
        json.dumps({"budget_tokens": 3 * PER_FRAME, "horizon": 64, "window": 1}),
        "--episodes",
        "1",
        "--output-dir",
        str(out),
    ]
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps(arguments))
    assert (
        run_main(monkeypatch, "scripts/eval/evaluate.py", "--args-file", str(args_file))
        == 0
    )
    run = load_run(out)
    assert run.info["policy_config"]["budget_tokens"] == 3 * PER_FRAME
    assert run.info["episodes"] == 1


def test_full_policy_horizon_and_forks() -> None:
    geometry = Geometry(4)
    full = make_policy("full", geometry=geometry, horizon=8)
    k = torch.randn(1, 1, 4 * PER_FRAME, 6)
    for start in range(0, 16, 4):
        full.append(k, k, None, start)
        full.compact()
    assert min(block.t0 for block in full.blocks) == 8  # frames 0-7 beyond horizon
    twin = full.fork()
    twin.append(k, k, None, 16)
    twin.compact()
    # The original still holds frames 8-15; the fork moved on to 12-19 and
    # shares the unmodified cells with it.
    assert [b.t0 for b in full.blocks] == list(range(8, 16)) and full.now == 15
    assert [b.t0 for b in twin.blocks] == list(range(12, 20))
    assert twin.blocks[0] is full.blocks[4]
    window = make_policy("window", budget_tokens=2 * PER_FRAME, geometry=geometry)
    window.append(k, k, None, 0)
    window.compact()
    copy = window.fork()
    copy.blocks[0].k.add_(1.0)
    assert not torch.equal(copy.blocks[0].k, window.blocks[0].k)
