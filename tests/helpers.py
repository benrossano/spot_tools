"""Shared stand-ins for the ROS-free planner and follower tests."""

import importlib.util
import pathlib

import numpy as np

from robot_executor_interface.mid_level_planner import MidLevelPlanner, OccupancyMap

REPO = pathlib.Path(__file__).resolve().parents[1]
FREE, OCC, UNK = 0, 100, -1
RES = 0.1  # m per cell, Hydra's default voxel size


class Feedback:
    """Minimal FeedbackCollector: records prints, no plotting, no ROS."""

    def __init__(self):
        self.messages = []
        self.plan_valid = True
        self.break_out_of_waiting_loop = False

    def print(self, level, s):
        self.messages.append((level, str(s)))

    def follow_path_feedback(self, path):
        pass

    def path_following_progress_feedback(self, progress_point, target_point):
        pass

    def path_follow_MLP_feedback(self, *args):
        pass

    def gaze_feedback(self, *args):
        pass

    def log_lease_takeover(self, event):
        pass


def homo(x=0.0, y=0.0, yaw=0.0):
    """4x4 pose with planar translation (m) and rotation about z (rad)."""
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[0, 3], m[1, 3] = x, y
    return m


def make_map(grid, origin=None, robot=None, inflate=0.0, cost=False, **kw):
    om = OccupancyMap(
        Feedback(), inflate_radius_meters=inflate, use_cost_map=cost, **kw
    )
    om.set_grid(
        np.asarray(grid, dtype=np.int8),
        RES,
        np.eye(4) if origin is None else origin,
        np.eye(4) if robot is None else robot,
        0.0,
    )
    return om


def planner_on(grid, robot_xy=(0.05, 0.05), lookahead=50, **map_kw):
    om = make_map(grid, robot=homo(*robot_xy), **map_kw)
    return MidLevelPlanner(om, Feedback(), lookahead_distance_grid=lookahead)


def cell_of(om, pose):
    """Grid cell containing a metric point; offsets half a cell so exact corners are unambiguous."""
    p = np.array(pose, dtype=float).reshape(-1)[:2]
    return om.global_position_to_grid_cell(
        om.map_origin
        @ (
            np.linalg.inv(om.map_origin) @ np.array([p[0], p[1], 0, 1])
            + np.array([RES / 2, RES / 2, 0, 0])
        ).reshape(4, 1)
    )


def load_example_harness():
    path = REPO / "spot_tools" / "examples" / "test_mid_level_planner.py"
    spec = importlib.util.spec_from_file_location("mlp_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
