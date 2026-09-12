"""End-to-end ROS 2 check: executor + A* planner drive the fake Spot through the L hallway."""

import os
import shutil

import pytest

pytest.importorskip("rclpy")
from fake_nav_harness import run  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.ros]
ROS_SOURCED = shutil.which("ros2") is not None and bool(
    os.environ.get("AMENT_PREFIX_PATH")
)


@pytest.mark.skipif(not ROS_SOURCED, reason="ROS 2 environment not sourced")
@pytest.mark.parametrize("crop", [-1.0, 8.0])
def test_fake_spot_reaches_goal_through_the_hallway(tmp_path, crop):
    r = run(tmp_path, crop=crop, timeout_s=170.0, verbose=False, domain_id=79)
    assert r["traceback"] == "", r["traceback"]
    assert r["reached"] and r["executor_done"], {
        k: v for k, v in r.items() if k != "trajectory"
    }
    assert not r["violations"], r["violations"][:5]
    assert r["mlp_paths"] > 10
