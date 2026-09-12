import time

import numpy as np
import pytest
from helpers import RES, Feedback, homo, make_map
from spot_executor.fake_spot import FakeCommandClient, FakeSpot
from spot_skills.navigation_utils import follow_trajectory_continuous

from robot_executor_interface.mid_level_planner import MidLevelPlanner


@pytest.fixture
def fast(monkeypatch):
    """FakeSpot sleeps 0.5 s per command and the follower 0.1 s per tick; skip both."""
    monkeypatch.setattr(time, "sleep", lambda s: None)


def tracking_planner(spot, grid, feedback, lookahead=50, **map_kw):
    """Planner whose robot pose follows the fake robot, as the ROS occupancy callback would."""
    om = make_map(grid, **map_kw)
    origin = np.eye(4)

    class Tracking(MidLevelPlanner):
        def plan_path(self, high_level_path):
            x, y = spot.get_pose()[:2]
            om.set_grid(np.asarray(grid, np.int8), RES, origin, homo(x, y), 0.0)
            return super().plan_path(high_level_path)

    return Tracking(om, feedback, lookahead_distance_grid=lookahead)


def test_reaches_goal_without_planner(fast):
    fb = Feedback()
    spot = FakeSpot(init_pose=np.zeros(4))
    wps = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], float)
    assert follow_trajectory_continuous(spot, wps, 1.0, 0.5, 10.0, None, feedback=fb)
    assert np.linalg.norm(spot.get_pose()[:2] - wps[-1, :2]) < 0.5


def test_reaches_goal_with_planner_on_open_grid(fast):
    fb = Feedback()
    spot = FakeSpot(init_pose=np.array([0.5, 0.5, 0.0, 0.0]))
    planner = tracking_planner(spot, np.zeros((100, 100)), fb)
    wps = np.array([[0.5, 0.5, 0], [6.5, 0.5, 0]])
    assert follow_trajectory_continuous(spot, wps, 1.0, 0.5, 30.0, planner, feedback=fb)
    assert not any("A* failed" in m for _, m in fb.messages)


def test_planner_detours_around_wall_and_stays_in_free_space(fast):
    fb = Feedback()
    grid = np.zeros((100, 100), np.int8)
    grid[0:70, 50] = 100  # wall at x = 5 m from y = 0 to 7 m; only way round is north
    spot = FakeSpot(init_pose=np.array([0.5, 3.0, 0.0, 0.0]))
    planner = tracking_planner(spot, grid, fb, inflate=0.3)
    visited = []
    orig = FakeCommandClient.robot_command

    def record(self, *a, **k):
        orig(self, *a, **k)
        visited.append(np.array(self.fake_spot.get_pose()[:2], dtype=float))

    FakeCommandClient.robot_command = record
    try:
        ok = follow_trajectory_continuous(
            spot,
            np.array([[0.5, 3.0, 0], [9.5, 3.0, 0]]),
            1.0,
            0.5,
            60.0,
            planner,
            feedback=fb,
        )
    finally:
        FakeCommandClient.robot_command = orig
    assert ok
    assert max(y for _, y in visited) > 6.5
    for x, y in visited:
        assert grid[int(y / RES), int(x / RES)] == 0


def test_times_out_when_robot_does_not_move(fast, monkeypatch):
    monkeypatch.setattr(FakeCommandClient, "robot_command", lambda self, *a, **k: None)
    spot = FakeSpot(init_pose=np.zeros(4))
    wps = np.array([[0, 0, 0], [5, 0, 0]], float)
    assert not follow_trajectory_continuous(
        spot, wps, 1.0, 0.5, 0.2, None, feedback=Feedback()
    )


def test_returns_false_when_plan_invalidated_by_lease(fast):
    fb = Feedback()
    fb.plan_valid = False
    spot = FakeSpot(init_pose=np.zeros(4))
    wps = np.array([[0, 0, 0], [5, 0, 0]], float)
    assert not follow_trajectory_continuous(
        spot, wps, 1.0, 0.5, 10.0, None, feedback=fb
    )
    assert np.allclose(spot.get_pose()[:2], 0)


@pytest.mark.xfail(
    strict=True,
    reason="feedback=None is accepted by the signature but dereferenced unconditionally",
)
def test_accepts_no_feedback(fast):
    spot = FakeSpot(init_pose=np.zeros(4))
    wps = np.array([[0, 0, 0], [2, 0, 0]], float)
    assert follow_trajectory_continuous(spot, wps, 1.0, 0.5, 10.0, None, feedback=None)


@pytest.mark.xfail(
    strict=True,
    reason="FakeSpot.get_pose is (x, y, z, yaw); Spot.get_pose is (x, y, yaw)",
)
def test_fake_spot_pose_matches_real_spot_pose_shape():
    assert len(FakeSpot(init_pose=np.zeros(4)).get_pose()) == 3


def test_progress_timeout_aborts_a_stalled_follow_before_the_global_timeout(
    fast, monkeypatch
):
    monkeypatch.setattr(FakeCommandClient, "robot_command", lambda self, *a, **k: None)
    spot = FakeSpot(init_pose=np.zeros(4))
    wps = np.array([[0, 0, 0], [5, 0, 0]], float)
    t0 = time.time()
    ok = follow_trajectory_continuous(
        spot, wps, 1.0, 0.5, 30.0, None, feedback=Feedback(), progress_timeout=0.3
    )
    assert not ok and time.time() - t0 < 5.0


def test_progress_timeout_does_not_fire_while_progressing(fast):
    fb = Feedback()
    spot = FakeSpot(init_pose=np.zeros(4))
    wps = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], float)
    assert follow_trajectory_continuous(
        spot, wps, 1.0, 0.5, 10.0, None, feedback=fb, progress_timeout=5.0
    )
    assert not any("No progress" in m for _, m in fb.messages)
