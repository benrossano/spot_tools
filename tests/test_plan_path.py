import numpy as np
import shapely
from helpers import OCC, RES, cell_of, planner_on


def test_target_sits_one_lookahead_along_the_path():
    p = planner_on(np.zeros((100, 100)), robot_xy=(1.05, 1.05), lookahead=20)
    ok, out = p.plan_path(np.array([[1.05, 1.05], [8.05, 1.05]]))
    assert ok
    assert np.allclose(out.global_path_target_point_metric[:2, 0], [3.0, 1.0], atol=RES)
    assert np.allclose(out.target_point_metric, out.global_path_target_point_metric)
    wp = np.asarray(out.path_waypoints_metric)
    assert np.allclose(wp[0], [1.0, 1.0], atol=RES) and np.allclose(
        wp[-1], [3.0, 1.0], atol=RES
    )
    assert (
        isinstance(out.path_shapely, shapely.LineString) and out.path_shapely.length > 0
    )


def test_target_inside_obstacle_is_projected_to_free_space():
    grid = np.zeros((100, 100), np.int8)
    grid[5:16, 28:33] = OCC
    p = planner_on(grid, robot_xy=(1.05, 1.05), lookahead=20)
    ok, out = p.plan_path(np.array([[1.05, 1.05], [8.05, 1.05]]))
    assert ok
    i, j = cell_of(p.occupancy_map_obj, out.target_point_metric)
    assert grid[i, j] == 0
    assert not np.allclose(out.target_point_metric, out.global_path_target_point_metric)


def test_lookahead_beyond_path_end_clamps_to_the_end():
    p = planner_on(np.zeros((50, 50)), robot_xy=(0.55, 0.55), lookahead=200)
    ok, out = p.plan_path(np.array([[0.55, 0.55], [3.05, 0.55]]))
    assert ok and np.allclose(
        out.global_path_target_point_metric[:2, 0], [3.0, 0.5], atol=RES
    )


def test_target_outside_grid_is_projected_to_the_frontier():
    p = planner_on(np.zeros((50, 50)), robot_xy=(0.55, 0.55), lookahead=200)
    ok, out = p.plan_path(np.array([[0.55, 0.55], [10.0, 0.55]]))
    assert ok
    assert np.allclose(out.target_point_metric[:2, 0], [4.9, 0.5], atol=RES)


def test_robot_outside_grid_is_clamped_onto_it():
    p = planner_on(np.zeros((50, 50)), robot_xy=(-1.0, 2.0), lookahead=10)
    ok, out = p.plan_path(np.array([[-1.0, 2.0], [4.0, 2.0]]))
    assert ok and len(out.path_waypoints_metric) > 1


def test_unreachable_target_reports_failure_with_fallback_path():
    grid = np.zeros((50, 50), np.int8)
    grid[:, 25] = OCC
    p = planner_on(grid, robot_xy=(0.55, 2.55), lookahead=40)
    ok, out = p.plan_path(np.array([[0.55, 2.55], [4.55, 2.55]]))
    assert not ok
    assert len(out.path_waypoints_metric) == 0
    assert (
        out.path_shapely.length
        == shapely.LineString([[0.55, 2.55], [4.55, 2.55]]).length
    )


def test_planned_waypoints_avoid_inflated_obstacles():
    grid = np.zeros((80, 80), np.int8)
    grid[30:50, 40] = OCC
    p = planner_on(grid, robot_xy=(1.05, 4.05), lookahead=50, inflate=0.3)
    ok, out = p.plan_path(np.array([[1.05, 4.05], [7.05, 4.05]]))
    assert ok
    inflated = p.occupancy_map
    for wp in out.path_waypoints_metric:
        i, j = cell_of(p.occupancy_map_obj, wp)
        assert inflated[i, j] == 0


def test_commitment_cost_is_off_by_default_and_without_history():
    p = planner_on(np.zeros((50, 50)), robot_xy=(0.55, 2.05), lookahead=30)
    hl = np.array([[0.55, 2.05], [4.55, 2.05]])
    assert p._commitment_cost_map(hl) is None
    ok, _ = p.plan_path(hl)
    assert ok and p.prev_path_metric is not None and p._commitment_cost_map(hl) is None


def test_commitment_biases_a_star_toward_previous_path():
    from robot_executor_interface.mid_level_planner import MidLevelPlanner

    grid = np.zeros((50, 60))
    om = planner_on(grid).occupancy_map_obj
    p = MidLevelPlanner(
        om, om.feedback, path_commitment_weight=1.0, path_commitment_band_m=0.3
    )
    hl = np.array([[1.0, 2.0], [5.0, 2.0]])
    # previous local path bowed 0.8 m north of the straight line
    prev = np.array(
        [
            [x, 2.0 + 0.8 * np.sin(np.pi * (x - 1.0) / 4.0)]
            for x in np.arange(1.0, 5.01, RES)
        ]
    )
    p.prev_path_metric = prev
    p.prev_high_level_metric = hl.copy()
    straight = planner_on(grid).a_star((20, 10), (20, 50))
    p._commitment_cost = p._commitment_cost_map(hl)
    assert p._commitment_cost is not None and p._commitment_cost.max() == 1.0
    committed = p.a_star((20, 10), (20, 50))
    assert max(i for i, _ in straight) <= 21
    assert max(i for i, _ in committed) >= 26


def test_commitment_resets_when_the_high_level_path_changes():
    from robot_executor_interface.mid_level_planner import MidLevelPlanner

    om = planner_on(np.zeros((50, 50)), robot_xy=(0.55, 2.05)).occupancy_map_obj
    p = MidLevelPlanner(
        om, om.feedback, lookahead_distance_grid=30, path_commitment_weight=1.0
    )
    ok, _ = p.plan_path(np.array([[0.55, 2.05], [4.55, 2.05]]))
    assert ok and p.prev_path_metric is not None
    assert p._commitment_cost_map(np.array([[0.55, 2.05], [4.55, 4.05]])) is None
    assert p.prev_path_metric is None


def test_plan_path_without_a_grid_falls_back_to_the_high_level_path():
    from helpers import Feedback

    from robot_executor_interface.mid_level_planner import MidLevelPlanner, OccupancyMap

    fb = Feedback()
    p = MidLevelPlanner(OccupancyMap(fb), fb)
    hl = np.array([[0.0, 0.0], [3.0, 0.0]])
    ok, out = p.plan_path(hl)
    assert not ok and out.target_point_metric is None
    assert out.path_shapely.length == 3.0 and len(out.path_waypoints_metric) == 0
    assert any("No occupancy grid" in m for _, m in fb.messages)


def test_target_in_unknown_is_projected_unless_allowed():
    from robot_executor_interface.mid_level_planner import MidLevelPlanner

    grid = np.zeros((60, 60), np.int8)
    grid[:, 30:] = -1  # right half never observed
    hl = np.array([[0.55, 3.05], [5.55, 3.05]])
    om = planner_on(grid, robot_xy=(0.55, 3.05)).occupancy_map_obj
    default = MidLevelPlanner(om, om.feedback, lookahead_distance_grid=45)
    ok, out = default.plan_path(hl)
    assert ok and out.target_point_metric[0, 0] < 3.0  # pulled back into observed space
    allowed = MidLevelPlanner(
        om, om.feedback, lookahead_distance_grid=45, allow_unknown_target=True
    )
    ok, out = allowed.plan_path(hl)
    assert ok and np.isclose(out.target_point_metric[0, 0], 5.0, atol=RES)
    assert np.allclose(
        out.target_point_metric[:2, 0], out.global_path_target_point_metric[:2, 0]
    )
