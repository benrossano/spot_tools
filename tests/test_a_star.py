import numpy as np
from helpers import OCC, UNK, planner_on


def test_straight_line_in_free_grid():
    path = planner_on(np.zeros((20, 20))).a_star((0, 0), (0, 19))
    assert path[0] == (0, 0) and path[-1] == (0, 19) and len(path) == 20


def test_routes_through_the_only_gap():
    grid = np.zeros((20, 20), np.int8)
    grid[:, 10] = OCC
    grid[10, 10] = 0
    path = planner_on(grid).a_star((0, 0), (19, 19))
    assert (10, 10) in path
    assert all(grid[i, j] != OCC for i, j in path)


def test_fully_blocked_returns_none():
    grid = np.zeros((20, 20), np.int8)
    grid[:, 10] = OCC
    assert planner_on(grid).a_star((0, 0), (19, 19)) is None


def test_start_or_goal_inside_obstacle_returns_none():
    grid = np.zeros((5, 5), np.int8)
    grid[4, 4] = OCC
    p = planner_on(grid)
    assert p.a_star((0, 0), (4, 4)) is None
    assert p.a_star((4, 4), (0, 0)) is None


def test_out_of_bounds_goal_returns_none():
    assert planner_on(np.zeros((5, 5))).a_star((0, 0), (5, 5)) is None


def test_unknown_is_traversed_as_free():
    """Characterisation: only cells > 0 block; UNKNOWN (-1) is planned through."""
    grid = np.zeros((20, 20), np.int8)
    grid[:, 10] = UNK
    path = planner_on(grid).a_star((0, 0), (0, 19))
    assert path is not None and any(grid[i, j] == UNK for i, j in path)


def test_four_connected_has_no_diagonal_steps():
    path = planner_on(np.zeros((10, 10))).a_star(
        (0, 0), (9, 9), diagonal_movement=False
    )
    steps = np.abs(np.diff(np.array(path), axis=0)).sum(axis=1)
    assert (steps == 1).all() and len(path) == 19


def test_cost_map_pushes_path_away_from_wall():
    grid = np.zeros((21, 41), np.int8)
    grid[0, :] = OCC
    straight = planner_on(grid).a_star((1, 0), (1, 40))
    bowed = planner_on(
        grid, cost=True, safe_distance=0.5, nearest_obstacle_cost=5.0
    ).a_star((1, 0), (1, 40))
    assert max(i for i, _ in bowed) > max(i for i, _ in straight)


def test_open_grid_diagonal_is_optimal_despite_manhattan_heuristic():
    """Manhattan is inadmissible for unit-cost diagonals; on an open grid it still finds the optimum."""
    path = planner_on(np.zeros((30, 30))).a_star((0, 0), (29, 29))
    assert len(path) == 30
