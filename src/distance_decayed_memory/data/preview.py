"""Human-viewable previews of generated episodes: a GIF, a map, revisit pairs."""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from distance_decayed_memory.data.revisit_script import RevisitEvent

SCALE = 3
GIF_SCALE = 2
PHASE_COLORS = {
    "explore": (90, 160, 255),
    "return": (255, 140, 40),
    "arrive": (255, 60, 60),
    "linger": (255, 60, 60),
    "pause": (170, 170, 170),
}


def _map_image(layout: np.ndarray, cell_pixels: int) -> Image.Image:
    height, width = layout.shape
    image = Image.new("RGB", (width * cell_pixels, height * cell_pixels), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        for x in range(width):
            if layout[y, x]:
                top = (height - 1 - y) * cell_pixels
                draw.rectangle(
                    [
                        x * cell_pixels,
                        top,
                        (x + 1) * cell_pixels - 1,
                        top + cell_pixels - 1,
                    ],
                    fill=(235, 235, 235),
                )
    return image


def _to_map(
    point: Sequence[float], height: int, cell_pixels: int
) -> tuple[float, float]:
    return point[0] * cell_pixels, (height - point[1]) * cell_pixels


def _arrow(draw, point, heading, cell_pixels, color, height) -> None:
    x, y = _to_map(point, height, cell_pixels)
    length = cell_pixels * 0.6
    tip = (x + length * math.cos(heading), y - length * math.sin(heading))
    draw.line([(x, y), tip], fill=color, width=2)
    radius = max(2, cell_pixels // 6)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def write_map(
    path: pathlib.Path,
    layout: np.ndarray,
    poses: np.ndarray,
    phases: Sequence[str],
    events: Sequence[RevisitEvent],
    cell_pixels: int = 48,
) -> None:
    """Top-down layout with the trajectory coloured by phase and anchors marked."""
    height = layout.shape[0]
    image = _map_image(layout, cell_pixels)
    draw = ImageDraw.Draw(image)
    for t in range(1, len(poses)):
        draw.line(
            [
                _to_map(poses[t - 1], height, cell_pixels),
                _to_map(poses[t], height, cell_pixels),
            ],
            fill=PHASE_COLORS.get(phases[t], (0, 0, 0)),
            width=2,
        )
    for event in events:
        if event.status == "completed":
            _arrow(
                draw,
                event.anchor_position,
                event.anchor_heading,
                cell_pixels,
                (200, 0, 0),
                height,
            )
    image.save(path)


def write_revisit_pairs(
    path: pathlib.Path,
    frames: np.ndarray,
    events: Sequence[RevisitEvent],
    limit: int = 8,
) -> None:
    """One row per completed revisit: anchor frame, return frame, and the gap."""
    completed = [e for e in events if e.status == "completed"][:limit]
    size = frames.shape[1] * SCALE
    label_width = 150
    rows = max(len(completed), 1)
    image = Image.new("RGB", (2 * size + label_width, rows * size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    if not completed:
        draw.text((8, 8), "no completed revisits", fill=(0, 0, 0))
    for row, event in enumerate(completed):
        for column, frame in enumerate((event.anchor_frame, event.return_frame)):
            tile = Image.fromarray(frames[frame]).resize((size, size), Image.NEAREST)
            image.paste(tile, (column * size, row * size))
        draw.text(
            (2 * size + 8, row * size + 8),
            f"anchor t={event.anchor_frame}\nreturn t={event.return_frame}\n"
            f"gap={event.gap} frames",
            fill=(0, 0, 0),
        )
    image.save(path)


def write_gif(
    path: pathlib.Path,
    frames: np.ndarray,
    layout: np.ndarray,
    poses: np.ndarray,
    phases: Sequence[str],
    events: Sequence[RevisitEvent],
    stride: int = 1,
    fps: int = 16,
) -> None:
    """First-person view beside a live map; the active anchor is drawn in red."""
    size = frames.shape[1] * GIF_SCALE
    height = layout.shape[0]
    cell_pixels = size // height
    base = _map_image(layout, cell_pixels)
    active = sorted(
        (e for e in events if e.depart_frame is not None),
        key=lambda e: e.depart_frame,
    )
    pictures = []
    for t in range(0, len(frames), stride):
        canvas = Image.new("RGB", (2 * size, size + 14), (255, 255, 255))
        canvas.paste(Image.fromarray(frames[t]).resize((size, size), Image.NEAREST))
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        for event in active:
            end = event.return_frame if event.return_frame is not None else t
            if event.depart_frame <= t <= end + 8:
                _arrow(
                    draw,
                    event.anchor_position,
                    event.anchor_heading,
                    cell_pixels,
                    (220, 0, 0),
                    height,
                )
        _arrow(
            draw,
            poses[t],
            poses[t][2],
            cell_pixels,
            PHASE_COLORS.get(phases[t], (0, 0, 0)),
            height,
        )
        canvas.paste(panel, (size, 0))
        ImageDraw.Draw(canvas).text((4, size + 1), f"t={t} {phases[t]}", fill=(0, 0, 0))
        pictures.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    pictures[0].save(
        path,
        save_all=True,
        append_images=pictures[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
