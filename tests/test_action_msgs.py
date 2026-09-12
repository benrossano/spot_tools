"""ActionSequence <-> robot_executor_msgs round trips (needs the built messages on PYTHONPATH)."""

import numpy as np
import pytest

pytest.importorskip("robot_executor_msgs.msg")
from robot_executor_interface_ros.action_descriptions_ros import (  # noqa: E402
    from_msg,
    to_msg,
)

from robot_executor_interface.action_descriptions import (  # noqa: E402
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)


def roundtrip(action):
    seq = ActionSequence("plan-1", "hamilton", [action])
    back = from_msg(to_msg(seq))
    assert (back.plan_id, back.robot_name, len(back.actions)) == (
        "plan-1",
        "hamilton",
        1,
    )
    return back.actions[0]


def test_follow_roundtrip_keeps_frame_and_xy():
    path = np.array([[0.0, 0.0, 0.1], [1.0, 2.0, -1.0], [3.0, 3.0, 2.5]])
    f = roundtrip(Follow("hamilton/map", path))
    assert isinstance(f, Follow) and f.frame == "hamilton/map"
    assert f.path2d.shape == (3, 3) and np.allclose(f.path2d[:, :2], path[:, :2])


@pytest.mark.xfail(
    strict=True,
    reason="waypoints_to_path replaces the heading column with the segment direction",
)
def test_follow_roundtrip_keeps_heading():
    path = np.array([[0.0, 0.0, 0.1], [1.0, 2.0, -1.0], [3.0, 3.0, 2.5]])
    assert np.allclose(roundtrip(Follow("hamilton/map", path)).path2d[:, 2], path[:, 2])


def test_gaze_roundtrip_keeps_points_and_stow_flag():
    g = roundtrip(
        Gaze(
            "hamilton/map",
            np.array([1.0, 2.0, 0.0]),
            np.array([3.0, 4.0, 0.5]),
            "obj7",
            stow_after=True,
        )
    )
    assert isinstance(g, Gaze) and g.object_id == "obj7" and g.stow_after
    assert np.allclose(g.robot_point, [1, 2, 0]) and np.allclose(
        g.gaze_point, [3, 4, 0.5]
    )


def test_pick_roundtrip_keeps_class_id_and_points():
    p = roundtrip(
        Pick(
            "hamilton/map",
            "cone",
            np.array([1.0, 2.0, 0.0]),
            np.array([3.0, 4.0, 0.5]),
            "obj7",
        )
    )
    assert isinstance(p, Pick) and (p.object_class, p.object_id) == ("cone", "obj7")
    assert np.allclose(p.robot_point, [1, 2, 0]) and np.allclose(
        p.object_point, [3, 4, 0.5]
    )


def test_place_roundtrip_keeps_id_and_points():
    p = roundtrip(
        Place(
            "hamilton/map",
            "cone",
            np.array([1.0, 2.0, 0.0]),
            np.array([3.0, 4.0, 0.5]),
            "obj7",
        )
    )
    assert isinstance(p, Place) and p.object_id == "obj7"
    assert np.allclose(p.robot_point, [1, 2, 0]) and np.allclose(
        p.object_point, [3, 4, 0.5]
    )


def test_gaze_roundtrip_keeps_frame():
    assert (
        roundtrip(Gaze("hamilton/map", np.zeros(3), np.ones(3), "o")).frame
        == "hamilton/map"
    )


def test_pick_roundtrip_keeps_frame():
    assert (
        roundtrip(Pick("hamilton/map", "cone", np.zeros(3), np.ones(3), "o")).frame
        == "hamilton/map"
    )


def test_place_roundtrip_keeps_frame_and_class():
    p = roundtrip(Place("hamilton/map", "cone", np.zeros(3), np.ones(3), "o"))
    assert (p.frame, p.object_class) == ("hamilton/map", "cone")
