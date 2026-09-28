"""Grid search and a closed-loop controller for Memory Maze's discrete actions.

Coordinates follow the Memory Maze global observables: ``agent_pos`` is
``(x, y)`` in grid units with the bottom-left corner at the origin, a cell
``(x, y)`` covers ``[x, x + 1) × [y, y + 1)``, and ``maze_layout[y, x]`` is 1
for a traversable cell. Heading is ``atan2(dir_y, dir_x)``; the ``left``
action increases it (counter-clockwise).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NOOP, FORWARD, LEFT, RIGHT, FORWARD_LEFT, FORWARD_RIGHT = range(6)
ACTION_NAMES = ("noop", "forward", "left", "right", "forward_left", "forward_right")
NUM_ACTIONS = len(ACTION_NAMES)

Cell = tuple[int, int]
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def cell_of(position: Sequence[float]) -> Cell:
    return int(math.floor(position[0])), int(math.floor(position[1]))


def cell_center(cell: Cell) -> np.ndarray:
    return np.asarray([cell[0] + 0.5, cell[1] + 0.5])


def is_free(layout: np.ndarray, cell: Cell) -> bool:
    x, y = cell
    height, width = layout.shape
    return 0 <= x < width and 0 <= y < height and bool(layout[y, x])


def free_cells(layout: np.ndarray) -> list[Cell]:
    ys, xs = np.nonzero(layout)
    return [(int(x), int(y)) for x, y in zip(xs, ys, strict=True)]


def neighbors(layout: np.ndarray, cell: Cell) -> list[Cell]:
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in _STEPS if is_free(layout, (x + dx, y + dy))]


def astar(layout: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest 4-connected path from ``start`` to ``goal`` inclusive, or None."""
    if not (is_free(layout, start) and is_free(layout, goal)):
        return None

    def heuristic(cell: Cell) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    frontier = [(heuristic(start), 0, start)]
    came_from: dict[Cell, Cell] = {}
    cost = {start: 0}
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        if current_cost > cost[current]:
            continue
        for successor in neighbors(layout, current):
            successor_cost = current_cost + 1
            if successor_cost < cost.get(successor, math.inf):
                cost[successor] = successor_cost
                came_from[successor] = current
                heapq.heappush(
                    frontier,
                    (successor_cost + heuristic(successor), successor_cost, successor),
                )
    return None


def bfs_distances(layout: np.ndarray, start: Cell) -> dict[Cell, int]:
    """Grid distance from ``start`` to every reachable free cell."""
    if not is_free(layout, start):
        return {}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for successor in neighbors(layout, current):
            if successor not in distances:
                distances[successor] = distances[current] + 1
                queue.append(successor)
    return distances


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_of(direction: Sequence[float]) -> float:
    return math.atan2(float(direction[1]), float(direction[0]))


@dataclass(frozen=True)
class ControllerConfig:
    """Thresholds tuned to Memory Maze at 4 Hz: about 18° per turn, 0.25 cell/step."""

    turn_in_place: float = math.radians(40.0)
    steer: float = math.radians(12.0)
    # One turn command from rest rotates about 17-18 degrees once momentum
    # decays, so the tolerance must exceed half of that or alignment oscillates.
    heading_tolerance: float = math.radians(10.0)
    fallback_heading_tolerance: float = math.radians(15.0)
    fallback_after_turns: int = 6
    waypoint_radius: float = 0.3
    goal_radius: float = 0.2
    brake_distance: float = 0.35
    settled_speed: float = 0.02
    settled_turn_rate: float = math.radians(1.0)
    steps_per_cell: float = 4.5
    steps_per_radian: float = 3.3
    arrival_steps: int = 10


def turn_toward(heading: float, target_heading: float, tolerance: float) -> int | None:
    """Turn-in-place action toward ``target_heading``, or None when within tolerance."""
    error = wrap_angle(target_heading - heading)
    if abs(error) <= tolerance:
        return None
    return LEFT if error > 0 else RIGHT


def steer_toward(
    position: Sequence[float],
    heading: float,
    target: Sequence[float],
    config: ControllerConfig,
) -> int:
    """One action that moves the agent toward ``target``."""
    dx = float(target[0]) - float(position[0])
    dy = float(target[1]) - float(position[1])
    error = wrap_angle(math.atan2(dy, dx) - heading)
    if abs(error) > config.turn_in_place:
        return LEFT if error > 0 else RIGHT
    if abs(error) > config.steer:
        return FORWARD_LEFT if error > 0 else FORWARD_RIGHT
    return FORWARD


class PathFollower:
    """Drive through cell centres to an exact goal point and optionally a heading.

    With ``stop_at_goal`` the agent brakes, settles, and then turns to
    ``goal_heading``; without it the follower finishes as soon as the agent
    passes within ``waypoint_radius`` of the goal. The follower is closed-loop:
    it recomputes the action from the observed pose every step, so momentum,
    action noise, and wall contact only delay it. It replans from the current
    cell whenever the agent leaves the planned path.
    """

    def __init__(
        self,
        layout: np.ndarray,
        goal: Sequence[float],
        goal_heading: float | None = None,
        config: ControllerConfig | None = None,
        stop_at_goal: bool = True,
    ) -> None:
        self.layout = layout
        self.goal = np.asarray(goal, dtype=np.float64)
        self.goal_cell = cell_of(self.goal)
        self.goal_heading = goal_heading
        self.stop_at_goal = stop_at_goal
        self.config = config or ControllerConfig()
        self.cells: list[Cell] = []
        self.done = False
        self._previous: np.ndarray | None = None
        self._previous_heading: float | None = None
        self._alignment_turns = 0

    def _plan(self, cell: Cell) -> None:
        path = astar(self.layout, cell, self.goal_cell)
        if path is None:
            raise ValueError(f"No path from {cell} to {self.goal_cell}")
        self.cells = path

    def remaining_cells(self) -> int:
        return max(len(self.cells) - 1, 0)

    def act(self, position: Sequence[float], heading: float) -> int | None:
        """Next action, or None once the goal (and heading) is reached and settled."""
        position = np.asarray(position, dtype=np.float64)
        speed = (
            0.0
            if self._previous is None
            else float(np.linalg.norm(position - self._previous))
        )
        turn_rate = (
            0.0
            if self._previous_heading is None
            else abs(wrap_angle(heading - self._previous_heading))
        )
        self._previous = position
        self._previous_heading = heading
        if self.done:
            return None

        cell = cell_of(position)
        if cell in self.cells:
            del self.cells[: self.cells.index(cell)]
        else:
            self._plan(cell)

        config = self.config
        if len(self.cells) > 1:
            waypoint = cell_center(self.cells[1])
            if np.linalg.norm(waypoint - position) < config.waypoint_radius:
                waypoint = (
                    cell_center(self.cells[2]) if len(self.cells) > 2 else self.goal
                )
            return steer_toward(position, heading, waypoint, config)

        distance = float(np.linalg.norm(self.goal - position))
        if not self.stop_at_goal:
            if distance > config.waypoint_radius:
                return steer_toward(position, heading, self.goal, config)
            self.done = True
            return None
        if distance > config.goal_radius:
            if speed > config.settled_speed and distance < config.brake_distance:
                return NOOP
            return steer_toward(position, heading, self.goal, config)
        if speed > config.settled_speed or turn_rate > config.settled_turn_rate:
            return NOOP
        if self.goal_heading is not None:
            tolerance = (
                config.fallback_heading_tolerance
                if self._alignment_turns >= config.fallback_after_turns
                else config.heading_tolerance
            )
            action = turn_toward(heading, self.goal_heading, tolerance)
            if action is not None:
                self._alignment_turns += 1
                return action
        self.done = True
        return None


def estimate_travel_steps(
    layout: np.ndarray,
    position: Sequence[float],
    heading: float,
    goal: Sequence[float],
    goal_heading: float | None,
    config: ControllerConfig | None = None,
    via: Cell | None = None,
) -> int:
    """Rough number of steps a :class:`PathFollower` needs, for scheduling."""
    config = config or ControllerConfig()
    start = cell_of(position)
    stops = [via, cell_of(goal)] if via is not None else [cell_of(goal)]
    points = [np.asarray(position, dtype=np.float64)]
    cells = [start]
    for stop in stops:
        leg = astar(layout, cells[-1], stop)
        if leg is None:
            raise ValueError(f"No path from {cells[-1]} to {stop}")
        cells.extend(leg[1:])
        points.extend(cell_center(c) for c in leg[1:])
    points.append(np.asarray(goal, dtype=np.float64))

    distance = 0.0
    turning = 0.0
    current_heading = heading
    for origin, target in zip(points[:-1], points[1:], strict=True):
        delta = target - origin
        length = float(np.linalg.norm(delta))
        if length < config.goal_radius:
            distance += length
            continue
        direction = math.atan2(delta[1], delta[0])
        turning += abs(wrap_angle(direction - current_heading))
        current_heading = direction
        distance += length
    if goal_heading is not None:
        turning += abs(wrap_angle(goal_heading - current_heading))
    return math.ceil(
        distance * config.steps_per_cell
        + turning * config.steps_per_radian
        + config.arrival_steps
    )
