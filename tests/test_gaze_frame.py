"""Gaze/Pick/Place points arrive in the planner's frame; Spot is commanded in its vision frame."""

from types import SimpleNamespace

import numpy as np
import pytest
import spot_executor.spot_executor as se_mod
from helpers import Feedback
from scipy.spatial.transform import Rotation
from spot_executor.fake_spot import FakeSpot
from spot_executor.spot_executor import SpotExecutor, transform_point_frame

from robot_executor_interface.action_descriptions import Gaze


def quat(yaw):
    q = Rotation.from_euler("z", yaw).as_quat()
    return SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])


def test_transform_point_frame_matches_se3():
    q = Rotation.from_euler("xyz", [0.1, -0.2, 0.9])
    tf_q = SimpleNamespace(**dict(zip("xyzw", q.as_quat())))
    t = np.array([1.0, -2.0, 0.5])
    p = np.array([3.0, 4.0, 0.7])
    assert np.allclose(transform_point_frame(t, tf_q, p), q.as_matrix() @ p + t)
    assert np.allclose(
        transform_point_frame(t, tf_q, p[:2]), q.as_matrix() @ [3.0, 4.0, 0.0] + t
    )


@pytest.fixture
def captured(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        se_mod,
        "turn_to_point",
        lambda spot, pose, pt: calls.setdefault("turn", np.array(pt)),
    )
    monkeypatch.setattr(
        se_mod,
        "gaze_at_vision_pose",
        lambda spot, pt, stow_after=False: calls.setdefault("gaze", np.array(pt))
        is not None,
    )
    return calls


def test_gaze_point_is_transformed_from_its_frame_into_vision(captured):
    yaw, t = 0.5, np.array([10.0, -3.0, 0.0])

    def tf_lookup(parent, child):
        assert (parent, child) == ("<spot_vision_frame>", "hamilton/map")
        return t, quat(yaw)

    executor = SpotExecutor(FakeSpot(init_pose=np.zeros(4)), None, tf_lookup, None)
    gaze = Gaze(
        "hamilton/map", np.array([1.0, 1.0, 0.0]), np.array([2.0, 0.0, 0.8]), "obj"
    )
    assert executor.execute_gaze(gaze, Feedback())
    expected = Rotation.from_euler("z", yaw).as_matrix() @ [2.0, 0.0, 0.8] + t
    assert np.allclose(captured["gaze"], expected)
    assert np.allclose(captured["turn"], expected)


def test_gaze_without_frame_is_taken_as_vision_and_warns(captured):
    def tf_lookup(parent, child):
        raise AssertionError("no lookup expected for an empty frame")

    fb = Feedback()
    executor = SpotExecutor(FakeSpot(init_pose=np.zeros(4)), None, tf_lookup, None)
    assert executor.execute_gaze(
        Gaze("", np.zeros(3), np.array([2.0, 0.0, 0.8]), "obj"), fb
    )
    assert np.allclose(captured["gaze"], [2.0, 0.0, 0.8])
    assert any(level == "WARNING" and "no frame" in msg for level, msg in fb.messages)
