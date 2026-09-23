"""Scripted exploration with deliberate returns to earlier views.

The planner implements ``TRACK_A_PLAN.md`` §3.2. It explores with a
coverage-seeking policy, schedules revisit events whose gap is sampled
log-uniformly, leaves on time to arrive near ``anchor_frame + target_gap``, and
then drives back to the anchor position and heading. Pauses, detours, partial
returns, episodes without scripted revisits, and random action noise keep the
return timing from becoming a learnable shortcut.

The planner only sees poses and the layout, never pixels, so it can be tested
on a toy simulator without Memory Maze.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from distance_decayed_memory.data.navigation import (
    NOOP,
    NUM_ACTIONS,
    Cell,
    ControllerConfig,
    PathFollower,
    bfs_distances,
    cell_center,
    cell_of,
    estimate_travel_steps,
    free_cells,
)


@dataclass(frozen=True)
class ScriptConfig:
    episode_frames: int = 2048
    min_gap: int = 16
    max_gap: int = 2048
    no_revisit_fraction: float = 0.3
    initial_explore_frames: tuple[int, int] = (32, 256)
    action_noise: float = 0.05
    pause_probability: float = 0.01
    pause_frames: tuple[int, int] = (2, 12)
    detour_probability: float = 0.25
    partial_return_probability: float = 0.1
    linger_frames: tuple[int, int] = (0, 8)
    end_margin: int = 8
    schedule_attempts: int = 16
    retry_after: int = 8
    navigation_timeout_factor: float = 4.0
    navigation_timeout_slack: int = 64
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self) -> None:
        if not 0 < self.min_gap <= self.max_gap:
            raise ValueError("Need 0 < min_gap <= max_gap")
        if self.episode_frames <= self.min_gap + self.end_margin:
            raise ValueError("Episode is too short for the minimum revisit gap")
        for name in ("no_revisit_fraction", "action_noise", "pause_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be a probability")


@dataclass
class RevisitEvent:
    """One scripted return. ``kind`` is ``revisit`` or ``partial``.

    ``status`` ends as ``completed`` (the agent reached the anchor pose at
    ``return_frame``), ``early`` (it did, but before ``min_gap`` frames had
    passed), ``partial`` (a deliberate abandoned return), ``timeout``, or
    ``unfinished`` (the episode ended first).
    """

    kind: str
    anchor_frame: int
    anchor_position: tuple[float, float]
    anchor_heading: float
    target_gap: int
    scheduled_frame: int
    detour_cell: Cell | None = None
    status: str = "pending"
    depart_frame: int | None = None
    return_frame: int | None = None
    return_position: tuple[float, float] | None = None
    return_heading: float | None = None

    @property
    def gap(self) -> int | None:
        if self.return_frame is None:
            return None
        return self.return_frame - self.anchor_frame

    def to_dict(self) -> dict:
        record = asdict(self)
        record["gap"] = self.gap
        return record


def sample_log_uniform_gap(rng: np.random.Generator, low: int, high: int) -> int:
    """Integer gap whose logarithm is uniform on ``[log low, log(high + 1))``."""
    value = math.exp(rng.uniform(math.log(low), math.log(high + 1)))
    return min(max(int(value), low), high)


class RevisitScript:
    """Choose one discrete action per observed frame."""

    def __init__(
        self,
        layout: np.ndarray,
        rng: np.random.Generator,
        config: ScriptConfig | None = None,
    ) -> None:
        self.layout = np.asarray(layout)
        self.rng = rng
        self.config = config or ScriptConfig()
        self.cells = free_cells(self.layout)
        self.revisits_enabled = bool(rng.random() >= self.config.no_revisit_fraction)
        low, high = self.config.initial_explore_frames
        self.explore_until = int(rng.integers(low, high + 1))
        self.positions: list[tuple[float, float]] = []
        self.headings: list[float] = []
        self.phases: list[str] = []
        self.events: list[RevisitEvent] = []
        self.visited: set[Cell] = set()
        self._active: RevisitEvent | None = None
        self._followers: list[PathFollower] = []
        self._abort_frame: int | None = None
        self._deadline: int | None = None
        self._explorer: PathFollower | None = None
        self._explore_deadline = 0
        self._pause = 0
        self._linger = 0
        self._next_schedule = self.explore_until

    # -- public ---------------------------------------------------------------

    def act(self, frame: int, position: Sequence[float], heading: float) -> int:
        if frame != len(self.positions):
            raise ValueError(f"Expected frame {len(self.positions)}, got {frame}")
        position = (float(position[0]), float(position[1]))
        self.positions.append(position)
        self.headings.append(float(heading))
        self.visited.add(cell_of(position))
        action, phase = self._decide(frame, position, float(heading))
        self.phases.append(phase)
        return action

    def finish(self) -> list[RevisitEvent]:
        """Close any open event and return every event of the episode."""
        if self._active is not None:
            self._active.status = "unfinished"
            self._active = None
        return self.events

    # -- decisions ------------------------------------------------------------

    def _decide(
        self, frame: int, position: tuple[float, float], heading: float
    ) -> tuple[int, str]:
        if self._pause > 0:
            self._pause -= 1
            return NOOP, "pause"
        if self._linger > 0:
            self._linger -= 1
            return NOOP, "linger"

        event = self._active
        if event is not None and event.status == "pending":
            if frame + self._travel_estimate(event, position, heading) >= (
                event.anchor_frame + event.target_gap
            ):
                self._depart(frame, event, position, heading)
        if event is not None and event.status == "navigating":
            action = self._navigate(frame, event, position, heading)
            if action is not None:
                return self._noisy(action), "return"
            if event.status == "completed":
                return NOOP, "arrive"

        if (
            self._active is None
            and self.revisits_enabled
            and frame >= self._next_schedule
        ):
            self._schedule(frame, position, heading)

        low, high = self.config.pause_frames
        if self.rng.random() < self.config.pause_probability:
            self._pause = int(self.rng.integers(low, high + 1)) - 1
            return NOOP, "pause"
        return self._noisy(self._explore(frame, position, heading)), "explore"

    def _noisy(self, action: int) -> int:
        if self.rng.random() < self.config.action_noise:
            return int(self.rng.integers(NUM_ACTIONS))
        return int(action)

    def _explore(
        self, frame: int, position: tuple[float, float], heading: float
    ) -> int:
        for _ in range(2):
            if (
                self._explorer is None
                or self._explorer.done
                or frame > self._explore_deadline
            ):
                self._explorer = self._new_explorer(frame, position, heading)
            action = self._explorer.act(position, heading)
            if action is not None:
                return action
        return NOOP

    def _new_explorer(
        self, frame: int, position: tuple[float, float], heading: float
    ) -> PathFollower:
        current = cell_of(position)
        distances = bfs_distances(self.layout, current)
        unvisited = [c for c in distances if c not in self.visited]
        if unvisited:
            nearest = min(distances[c] for c in unvisited)
            options = [c for c in unvisited if distances[c] == nearest]
        else:
            options = [c for c in distances if c != current] or [current]
        goal_cell = options[int(self.rng.integers(len(options)))]
        goal = cell_center(goal_cell) + self.rng.uniform(-0.25, 0.25, size=2)
        follower = PathFollower(
            self.layout, goal, config=self.config.controller, stop_at_goal=False
        )
        steps = estimate_travel_steps(
            self.layout, position, heading, goal, None, self.config.controller
        )
        self._explore_deadline = frame + self._timeout(steps)
        return follower

    # -- revisit events -------------------------------------------------------

    def _schedule(
        self, frame: int, position: tuple[float, float], heading: float
    ) -> None:
        config = self.config
        last_return = config.episode_frames - 1 - config.end_margin
        upper = min(config.max_gap, last_return)
        if frame < 1 or upper < config.min_gap:
            self._next_schedule = config.episode_frames
            return
        for _ in range(config.schedule_attempts):
            gap = sample_log_uniform_gap(self.rng, config.min_gap, upper)
            anchor_frame = int(self.rng.integers(max(0, frame - gap), frame))
            return_frame = anchor_frame + gap
            if return_frame > last_return:
                continue
            detour = None
            if self.rng.random() < config.detour_probability:
                detour = self.cells[int(self.rng.integers(len(self.cells)))]
            event = RevisitEvent(
                kind=(
                    "partial"
                    if self.rng.random() < config.partial_return_probability
                    else "revisit"
                ),
                anchor_frame=anchor_frame,
                anchor_position=self.positions[anchor_frame],
                anchor_heading=self.headings[anchor_frame],
                target_gap=gap,
                scheduled_frame=frame,
                detour_cell=detour,
            )
            if frame + self._travel_estimate(event, position, heading) > return_frame:
                continue
            self.events.append(event)
            self._active = event
            return
        self._next_schedule = frame + config.retry_after

    def _travel_estimate(
        self, event: RevisitEvent, position: tuple[float, float], heading: float
    ) -> int:
        return estimate_travel_steps(
            self.layout,
            position,
            heading,
            event.anchor_position,
            event.anchor_heading,
            self.config.controller,
            via=event.detour_cell,
        )

    def _depart(
        self,
        frame: int,
        event: RevisitEvent,
        position: tuple[float, float],
        heading: float,
    ) -> None:
        controller = self.config.controller
        event.status = "navigating"
        event.depart_frame = frame
        self._followers = []
        if event.detour_cell is not None:
            self._followers.append(
                PathFollower(
                    self.layout,
                    cell_center(event.detour_cell),
                    config=controller,
                    stop_at_goal=False,
                )
            )
        self._followers.append(
            PathFollower(
                self.layout,
                event.anchor_position,
                event.anchor_heading,
                config=controller,
            )
        )
        steps = self._travel_estimate(event, position, heading)
        self._deadline = frame + self._timeout(steps)
        self._abort_frame = None
        if event.kind == "partial":
            fraction = self.rng.uniform(0.2, 0.8)
            self._abort_frame = frame + max(1, int(fraction * steps))

    def _navigate(
        self,
        frame: int,
        event: RevisitEvent,
        position: tuple[float, float],
        heading: float,
    ) -> int | None:
        if self._abort_frame is not None and frame >= self._abort_frame:
            self._close(event, "partial")
            return None
        if self._deadline is not None and frame > self._deadline:
            self._close(event, "timeout")
            return None
        while self._followers:
            action = self._followers[0].act(position, heading)
            if action is not None:
                return action
            self._followers.pop(0)
        event.return_frame = frame
        event.return_position = position
        event.return_heading = heading
        if frame - event.anchor_frame < self.config.min_gap:
            self._close(event, "early")
            return None
        self._close(event, "completed")
        low, high = self.config.linger_frames
        self._linger = int(self.rng.integers(low, high + 1))
        return None

    def _close(self, event: RevisitEvent, status: str) -> None:
        event.status = status
        self._active = None
        self._followers = []
        self._explorer = None
        self._next_schedule = 0

    def _timeout(self, steps: int) -> int:
        config = self.config
        return int(steps * config.navigation_timeout_factor) + (
            config.navigation_timeout_slack
        )
