"""A pixel-free stand-in for Memory Maze dynamics, for tests and dry runs.

It mimics what was measured on the real environment at 4 Hz: turns approach
about 18.4° per step and forward motion about 0.25 cell per step, both with
momentum, and a ball of radius ``radius`` cannot enter wall cells.
"""

from __future__ import annotations

import math

import numpy as np

from distance_decayed_memory.data.navigation import (
    FORWARD,
    FORWARD_LEFT,
    FORWARD_RIGHT,
    LEFT,
    RIGHT,
    is_free,
)

TOY_LAYOUT_9X9 = np.asarray(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)
"""Layout of Memory Maze 9×9 seed 0, as returned by the real environment."""


class ToyMaze:
    def __init__(
        self,
        layout: np.ndarray,
        position: tuple[float, float],
        heading: float = 0.0,
        speed: float = 0.25,
        turn_rate: float = math.radians(18.4),
        response: float = 0.55,
        radius: float = 0.15,
    ) -> None:
        self.layout = np.asarray(layout)
        self.position = np.asarray(position, dtype=np.float64)
        self.heading = float(heading)
        self.speed = speed
        self.turn_rate = turn_rate
        self.response = response
        self.radius = radius
        self._velocity = 0.0
        self._angular = 0.0

    @property
    def direction(self) -> np.ndarray:
        return np.asarray([math.cos(self.heading), math.sin(self.heading)])

    def _clear(self, point: np.ndarray) -> bool:
        r = self.radius
        return all(
            is_free(self.layout, (math.floor(point[0] + dx), math.floor(point[1] + dy)))
            for dx in (-r, r)
            for dy in (-r, r)
        )

    def step(self, action: int) -> None:
        forward = 1.0 if action in (FORWARD, FORWARD_LEFT, FORWARD_RIGHT) else 0.0
        turn = {LEFT: 1.0, FORWARD_LEFT: 1.0, RIGHT: -1.0, FORWARD_RIGHT: -1.0}.get(
            action, 0.0
        )
        a = self.response
        self._velocity = (1 - a) * self._velocity + a * forward * self.speed
        self._angular = (1 - a) * self._angular + a * turn * self.turn_rate
        self.heading = (self.heading + self._angular + math.pi) % (
            2 * math.pi
        ) - math.pi
        moved = self.position + self._velocity * self.direction
        for candidate in (
            moved,
            np.asarray([moved[0], self.position[1]]),
            np.asarray([self.position[0], moved[1]]),
        ):
            if self._clear(candidate):
                self.position = candidate
                return
        self._velocity = 0.0
