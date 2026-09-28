from __future__ import annotations

import math

import numpy as np
import pytest

from distance_decayed_memory.data import navigation as nav
from distance_decayed_memory.data.toy_maze import TOY_LAYOUT_9X9, ToyMaze

CORRIDOR = np.asarray(
    [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)


def test_cell_indexing_matches_memory_maze_coordinates() -> None:
    assert nav.cell_of((0.1, 0.1)) == (0, 0)
    assert nav.cell_of((4.99, 2.0)) == (4, 2)
    assert nav.is_free(CORRIDOR, (0, 0))
    assert not nav.is_free(CORRIDOR, (0, 1))
    assert not nav.is_free(CORRIDOR, (5, 0))


def test_astar_follows_the_only_corridor() -> None:
    path = nav.astar(CORRIDOR, (0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2)] + [
        (3, 2),
        (2, 2),
        (1, 2),
        (0, 2),
    ]


def test_astar_rejects_walls_and_unreachable_goals() -> None:
    assert nav.astar(CORRIDOR, (0, 1), (0, 0)) is None
    island = np.asarray([[1, 0, 1]], dtype=np.uint8)
    assert nav.astar(island, (0, 0), (2, 0)) is None
    assert nav.astar(island, (0, 0), (0, 0)) == [(0, 0)]


@pytest.mark.parametrize("seed", range(20))
def test_astar_is_shortest_on_random_grids(seed: int) -> None:
    rng = np.random.default_rng(seed)
    layout = (rng.random((9, 9)) < 0.7).astype(np.uint8)
    cells = nav.free_cells(layout)
    start = cells[int(rng.integers(len(cells)))]
    distances = nav.bfs_distances(layout, start)
    for goal in cells:
        path = nav.astar(layout, start, goal)
        if goal not in distances:
            assert path is None
            continue
        assert path is not None
        assert len(path) - 1 == distances[goal]
        assert path[0] == start and path[-1] == goal
        for a, b in zip(path[:-1], path[1:], strict=True):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            assert nav.is_free(layout, b)


def test_steer_toward_chooses_turn_direction() -> None:
    config = nav.ControllerConfig()
    origin = (1.5, 1.5)
    assert nav.steer_toward(origin, 0.0, (3.5, 1.5), config) == nav.FORWARD
    assert nav.steer_toward(origin, 0.0, (1.5, 3.5), config) == nav.LEFT
    assert nav.steer_toward(origin, 0.0, (1.5, -0.5), config) == nav.RIGHT
    assert nav.steer_toward(origin, 0.0, (3.5, 2.0), config) == nav.FORWARD_LEFT
    assert nav.steer_toward(origin, 0.0, (3.5, 1.0), config) == nav.FORWARD_RIGHT


def test_turn_toward_uses_the_short_way_round() -> None:
    tolerance = math.radians(5)
    assert nav.turn_toward(math.radians(170), math.radians(-170), tolerance) == nav.LEFT
    assert (
        nav.turn_toward(math.radians(-170), math.radians(170), tolerance) == nav.RIGHT
    )
    assert nav.turn_toward(0.0, math.radians(3), tolerance) is None


def drive(sim: ToyMaze, follower: nav.PathFollower, limit: int) -> int:
    for step in range(limit):
        action = follower.act(sim.position, sim.heading)
        if action is None:
            return step
        sim.step(action)
        assert nav.is_free(sim.layout, nav.cell_of(sim.position))
    raise AssertionError("follower did not arrive")


@pytest.mark.parametrize("seed", range(10))
def test_follower_reaches_pose_in_toy_maze(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cells = nav.free_cells(TOY_LAYOUT_9X9)
    start, goal_cell = (cells[int(i)] for i in rng.choice(len(cells), 2, replace=False))
    goal = nav.cell_center(goal_cell) + rng.uniform(-0.3, 0.3, size=2)
    goal_heading = float(rng.uniform(-math.pi, math.pi))
    sim = ToyMaze(TOY_LAYOUT_9X9, tuple(nav.cell_center(start)), heading=1.0)
    follower = nav.PathFollower(TOY_LAYOUT_9X9, goal, goal_heading)
    estimate = nav.estimate_travel_steps(
        TOY_LAYOUT_9X9, sim.position, sim.heading, goal, goal_heading
    )
    steps = drive(sim, follower, limit=4 * estimate + 64)
    assert np.linalg.norm(sim.position - goal) <= follower.config.goal_radius
    assert abs(nav.wrap_angle(sim.heading - goal_heading)) <= math.radians(15)
    assert steps <= 2 * estimate + 16


def test_follower_replans_after_being_pushed_off_path() -> None:
    sim = ToyMaze(CORRIDOR, (0.5, 0.5), heading=0.0)
    follower = nav.PathFollower(CORRIDOR, (0.5, 2.5))
    for _ in range(10):
        sim.step(follower.act(sim.position, sim.heading))
    sim.position = np.asarray([4.5, 2.5])
    drive(sim, follower, limit=200)
    assert nav.cell_of(sim.position) == (0, 2)


def test_travel_estimate_grows_with_path_length() -> None:
    near = nav.estimate_travel_steps(CORRIDOR, (0.5, 0.5), 0.0, (2.5, 0.5), None)
    far = nav.estimate_travel_steps(CORRIDOR, (0.5, 0.5), 0.0, (0.5, 2.5), None)
    here = nav.estimate_travel_steps(CORRIDOR, (0.5, 0.5), 0.0, (0.55, 0.5), math.pi)
    assert near < far
    assert here < 25
