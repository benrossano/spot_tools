from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from spot_executor.spot_executor import transform_command_frame


def quat(yaw):
    q = Rotation.from_euler("z", yaw).as_quat()
    return SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])


def test_matches_se2_composition_and_leaves_input_untouched():
    yaw, t = 0.9, np.array([1.5, -2.0, 0.3])
    cmd = np.array([[0, 0, 0], [1, 0, 0.2], [2, 3, -1.0]], float)
    out = transform_command_frame(t, quat(yaw), cmd)
    R = Rotation.from_euler("z", yaw).as_matrix()[:2, :2]
    assert np.allclose(out[:, :2], cmd[:, :2] @ R.T + t[:2])
    assert np.allclose(out[:, 2], cmd[:, 2] + yaw)
    assert np.allclose(cmd[1], [1, 0, 0.2])


def test_identity_transform_is_a_noop():
    cmd = np.random.default_rng(1).normal(size=(5, 3))
    assert np.allclose(transform_command_frame(np.zeros(3), quat(0.0), cmd), cmd)


def test_nx2_path_gets_a_heading_column():
    out = transform_command_frame(
        np.array([1.0, 0, 0]), quat(0.5), np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 2], 0.5)
    assert np.allclose(out[0, :2], [1.0, 0.0])
