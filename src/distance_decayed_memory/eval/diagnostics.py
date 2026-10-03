"""Rollout diagnostics for a base model: drift, collapse, and action following.

These are engineering checks on A1, not results. They answer two questions
about free-running generation from a few true frames:

* **Drift and collapse.** How fast does the rollout leave the true frames,
  and how often does it settle on a static, featureless view (for example a
  wall close-up) while the true camera keeps moving?
* **Action following.** When the same context is continued with different
  actions, does the generated view turn the way real frames turn for those
  actions?

Turning shows up as a horizontal shift of the whole view, measured here by
phase correlation between consecutive frames. Its sign convention does not
matter: the true frames calibrate it per action.
"""

from __future__ import annotations

import numpy as np

# Episode-level heuristics for "stuck": over the scored tail, the generated view
# moves less than STUCK_MOTION of the true motion and keeps less than
# STUCK_DETAIL of the true detail. Tails whose true motion is below MIN_MOTION
# (pauses) are not classified.
STUCK_MOTION = 0.25
STUCK_DETAIL = 0.5
MIN_MOTION = 2.0


def grayscale(frames: np.ndarray) -> np.ndarray:
    """``[..., H, W, 3]`` in 0-255 to float32 luminance ``[..., H, W]``."""
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    return frames.astype(np.float32) @ weights


def detail(frames: np.ndarray) -> np.ndarray:
    """Mean absolute luminance gradient per frame (0-255 units); low for flat views."""
    gray = grayscale(frames)
    dx = np.abs(np.diff(gray, axis=-1)).mean(axis=(-2, -1))
    dy = np.abs(np.diff(gray, axis=-2)).mean(axis=(-2, -1))
    return dx + dy


def motion(frames: np.ndarray) -> np.ndarray:
    """Mean absolute luminance change from the previous frame (first frame: NaN)."""
    gray = grayscale(frames)
    change = np.abs(np.diff(gray, axis=-3)).mean(axis=(-2, -1))
    pad = np.full(change.shape[:-1] + (1,), np.nan, dtype=change.dtype)
    return np.concatenate([pad, change], axis=-1)


def _hann(size: int) -> np.ndarray:
    window = np.hanning(size).astype(np.float32)
    return np.outer(window, window)


def horizontal_shift(first: np.ndarray, second: np.ndarray) -> float:
    """Horizontal shift in pixels from ``first`` to ``second`` (``[H, W]`` gray).

    Phase correlation with a Hann window and parabolic sub-pixel refinement.
    Positive means the content moved toward larger column indices.
    """
    window = _hann(first.shape[0])[:, : first.shape[1]]
    a = (first - first.mean()) * window
    b = (second - second.mean()) * window
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    cross /= np.maximum(np.abs(cross), 1e-8)
    surface = np.fft.ifft2(cross).real
    row, column = np.unravel_index(np.argmax(surface), surface.shape)
    width = surface.shape[1]
    left = surface[row, (column - 1) % width]
    centre = surface[row, column]
    right = surface[row, (column + 1) % width]
    denominator = left - 2 * centre + right
    offset = 0.5 * (left - right) / denominator if denominator != 0 else 0.0
    shift = column + offset
    return float(shift - width if shift > width / 2 else shift)


def shifts(frames: np.ndarray) -> np.ndarray:
    """Per-frame horizontal shift from the previous frame (first frame: NaN).

    ``frames`` is ``[..., T, H, W, 3]``; returns ``[..., T]``.
    """
    gray = grayscale(frames)
    flat = gray.reshape(-1, *gray.shape[-3:])
    out = np.full(flat.shape[:2], np.nan, dtype=np.float32)
    for index, clip in enumerate(flat):
        for t in range(1, len(clip)):
            out[index, t] = horizontal_shift(clip[t - 1], clip[t])
    return out.reshape(gray.shape[:-2])


def psnr_np(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """PSNR in dB over the last three axes (uint8-range inputs)."""
    error = (prediction.astype(np.float32) - target.astype(np.float32)) ** 2
    mse = error.mean(axis=(-3, -2, -1))
    return 10.0 * np.log10(255.0**2 / np.maximum(mse, 1e-8))


def stuck(
    generated: np.ndarray, truth: np.ndarray, tail: int
) -> tuple[bool | None, float, float]:
    """Whether a rollout's last ``tail`` frames are static and flat versus the truth.

    Returns ``(flag, motion_ratio, detail_ratio)``; ``flag`` is ``None`` when the
    true camera barely moves over the tail (a pause), so the check cannot tell.
    """
    true_motion = np.nanmean(motion(truth)[-tail:])
    generated_motion = np.nanmean(motion(generated)[-tail:])
    motion_ratio = float(generated_motion / max(true_motion, 1e-6))
    detail_ratio = float(
        detail(generated)[-tail:].mean() / max(detail(truth)[-tail:].mean(), 1e-6)
    )
    if true_motion < MIN_MOTION:
        return None, motion_ratio, detail_ratio
    flag = motion_ratio < STUCK_MOTION and detail_ratio < STUCK_DETAIL
    return bool(flag), motion_ratio, detail_ratio


def override_actions(prev: np.ndarray, context: int, action: int | None) -> np.ndarray:
    """Conditioning actions with every generated frame driven by ``action``.

    ``prev[t]`` is the action that led into frame ``t``; frames before
    ``context`` keep their true actions. ``None`` keeps the true actions.
    """
    out = np.array(prev, copy=True)
    if action is not None:
        out[..., context:] = action
    return out


def shift_by_action(
    frame_shifts: np.ndarray, prev: np.ndarray, actions: int
) -> dict[int, tuple[float, float, int]]:
    """Mean, standard deviation, and count of shifts grouped by the leading action."""
    out = {}
    valid = np.isfinite(frame_shifts)
    for action in range(actions):
        values = frame_shifts[valid & (prev == action)]
        if len(values):
            out[action] = (float(values.mean()), float(values.std()), int(len(values)))
    return out
