"""Frame conventions and averaging for the AprilTag map anchor (no ROS, no robot)."""

import math
import pathlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "spot_tools_ros" / "src"))

from spot_tools_ros.fiducial_localization import (  # noqa: E402
    BD_FIDUCIAL_CONVENTION,
    Fiducial,
    cv_tag_to_bd_tag,
    invert,
    load_fiducials,
    map_T_odom_from_tag,
    planar,
    reject_inconsistent_axes,
    robust_average,
    save_fiducials,
    tag_up_agreement,
    yaw_of,
)


def se3(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def test_cv_to_bd_axes():
    # OpenCV's canonical 36h11 marker is upside down relative to the AprilTag library
    # (what Spot runs), so BD's "up the tag" is OpenCV's -y and BD's y is OpenCV's +x.
    cam_T_tag_cv = np.eye(4)
    cam_T_tag_cv[:3, :3] = np.diag([1.0, -1.0, -1.0])  # tag facing an OpenCV camera
    cam_T_tag_cv[:3, 3] = [0, 0, 2.0]
    bd = cv_tag_to_bd_tag(cam_T_tag_cv)
    assert np.allclose(bd[:3, 0], -cam_T_tag_cv[:3, 1])
    assert np.allclose(bd[:3, 1], cam_T_tag_cv[:3, 0])
    assert np.allclose(bd[:3, 2], cam_T_tag_cv[:3, 2])  # same normal, toward the viewer
    assert np.isclose(np.linalg.det(bd[:3, :3]), 1.0)


def upright_tag(x, y, z, heading):
    """BD-convention pose of a wall-mounted tag whose normal points along `heading`."""
    T = np.eye(4)
    zt = np.array([math.cos(heading), math.sin(heading), 0.0])
    xt = np.array([0.0, 0.0, 1.0])
    T[:3, 0], T[:3, 1], T[:3, 2] = xt, np.cross(zt, xt), zt
    T[:3, 3] = [x, y, z]
    return T


def test_reject_inconsistent_axes_drops_the_flipped_solution():
    upright = upright_tag(5.0, 1.0, 0.0, 0.3)
    assert np.isclose(np.linalg.det(upright[:3, :3]), 1.0)
    flipped = upright.copy()
    flipped[:3, :3] = upright[:3, :3] @ Rotation.from_euler("y", math.pi * 0.6).as_matrix()
    Ts = [upright] * 6 + [flipped]
    keep = reject_inconsistent_axes(Ts, 20.0)
    assert keep.tolist() == [True] * 6 + [False]
    assert tag_up_agreement(upright) > 0.99 and tag_up_agreement(flipped) < 0.9


def test_anchor_recovers_the_true_map_T_odom():
    map_T_odom = se3(3.0, -1.5, 0.2, yaw=0.7)
    map_T_tag = se3(5.0, 2.0, 0.4, roll=0.5 * math.pi, yaw=1.2)  # some upright tag
    odom_T_tag = invert(map_T_odom) @ map_T_tag
    est = map_T_odom_from_tag(map_T_tag, odom_T_tag, planar_result=False)
    assert np.allclose(est, map_T_odom, atol=1e-9)
    est2 = map_T_odom_from_tag(map_T_tag, odom_T_tag, planar_result=True)
    assert np.allclose(est2, map_T_odom, atol=1e-9)  # already planar


def test_planar_zeroes_tilt_and_keeps_yaw_and_translation():
    T = se3(1.0, 2.0, 0.3, roll=0.05, pitch=-0.04, yaw=2.0)
    P = planar(T)
    assert np.allclose(P[:3, 3], T[:3, 3])
    assert np.isclose(yaw_of(P), 2.0)
    assert np.allclose(P[2, :3], [0, 0, 1]) and np.allclose(P[:3, 2], [0, 0, 1])


def test_robust_average_drops_the_planar_ambiguity_outlier():
    truth = se3(4.0, 1.0, 0.0, yaw=0.3)
    rng = np.random.default_rng(0)
    samples = [truth @ se3(*rng.normal(0, 0.01, 3), yaw=rng.normal(0, 0.005)) for _ in range(12)]
    samples.append(truth @ se3(0.6, -0.4, 0.0, yaw=0.5))  # a wrong PnP solution
    res = robust_average(samples, max_pos_dev_m=0.15, max_yaw_dev_deg=5.0)
    assert res.num_kept == 12
    assert not res.kept[-1]
    assert np.linalg.norm(res.T[:3, 3] - truth[:3, 3]) < 0.02
    assert abs(yaw_of(res.T) - 0.3) < 0.01
    assert res.position_std_m < 0.03


def test_robust_average_handles_yaw_wraparound():
    Ts = [se3(yaw=math.pi - 0.01), se3(yaw=-math.pi + 0.01), se3(yaw=math.pi)]
    res = robust_average(Ts)
    assert res.num_kept == 3
    assert abs(abs(yaw_of(res.T)) - math.pi) < 0.02
    assert res.yaw_std_deg < 1.0


def test_fiducials_round_trip(tmp_path):
    T = se3(5.4, 1.6, 0.36, roll=0.5 * math.pi, yaw=0.5 * math.pi)
    f = Fiducial(1, 0.146, T, "hamilton/map", 42, 0.02, 0.4)
    path = tmp_path / "fiducials.yaml"
    save_fiducials(path, "hamilton/map", [f], {"bag": "x"})
    loaded = load_fiducials(path)
    assert set(loaded) == {1}
    assert np.allclose(loaded[1].map_T_tag, T)
    assert loaded[1].frame_id == "hamilton/map" and loaded[1].size_m == 0.146
    assert loaded[1].num_detections == 42
    import yaml

    assert yaml.safe_load(path.read_text())["convention"] == BD_FIDUCIAL_CONVENTION


def test_load_rejects_unknown_convention(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("frame_id: map\nconvention: apriltag-cv\nfiducials: []\n")
    with pytest.raises(ValueError):
        load_fiducials(path)
