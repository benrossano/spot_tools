"""Offline planner evaluation on a derived bag (occupancy + TF), no ROS runtime.

Feeds every /<robot>/hydra/tsdf/occupancy message through OccupancyMap.set_grid exactly as
occupancy_grid_ros_updater does, plans from the recorded robot pose toward the pose the robot
actually reached `--goal-ahead` metres later along its trajectory, and reports whether A*
succeeded, whether the returned waypoints stay out of occupied cells, and how much of the
lookahead was unknown. Produce the bag with tests/ros/record_bag_occupancy.sh.

  .venv/bin/python tests/ros/replay_planner_on_bag.py <derived_bag_dir> [--goal-ahead 8] [--commit 0]
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import rosbag2_py
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

from robot_executor_interface.mid_level_planner import MidLevelPlanner, OccupancyMap
from spot_tools_ros.utils import pose_to_homo


class Feedback:
    def __init__(self):
        self.messages = []

    def print(self, level, s):
        if level in ("WARNING", "ERROR"):
            self.messages.append((level, str(s)))


def read_bag(path, robot):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    buf = tf2_ros.Buffer(cache_time=rclpy_duration(10**6))
    grids, body_track = [], []
    occ_topic = f"/{robot}/hydra/tsdf/occupancy"
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == "/tf_static":
            for tr in deserialize_message(data, TFMessage).transforms:
                buf.set_transform_static(tr, "bag")
        elif topic == "/tf":
            for tr in deserialize_message(data, TFMessage).transforms:
                buf.set_transform(tr, "bag")
                if (
                    tr.header.frame_id == f"{robot}/odom"
                    and tr.child_frame_id == f"{robot}/body"
                ):
                    body_track.append(
                        (
                            stamp_ns(tr.header.stamp),
                            tr.transform.translation.x,
                            tr.transform.translation.y,
                        )
                    )
        elif topic == occ_topic:
            grids.append(deserialize_message(data, OccupancyGrid))
    return buf, grids, np.array(body_track)


def rclpy_duration(seconds):
    import rclpy.duration

    return rclpy.duration.Duration(seconds=seconds)


def stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def lookup(buf, parent, child, stamp):
    import rclpy.time

    tf = buf.lookup_transform(parent, child, rclpy.time.Time(nanoseconds=stamp))
    t, r = tf.transform.translation, tf.transform.rotation
    return pose_to_homo([t.x, t.y, t.z], r)


def goal_ahead(track, t_ns, ahead_m):
    """Recorded body position first reached `ahead_m` metres of path length after time t_ns."""
    idx = np.searchsorted(track[:, 0], t_ns)
    if idx >= len(track) - 1:
        return None
    seg = np.linalg.norm(np.diff(track[idx:, 1:3], axis=0), axis=1).cumsum()
    hit = np.searchsorted(seg, ahead_m)
    if hit >= len(seg):
        return None
    return track[idx + 1 + hit, 1:3]


def known_fraction(grid, ci, cj, radius_cells):
    ii, jj = np.ogrid[: grid.shape[0], : grid.shape[1]]
    disk = (ii - ci) ** 2 + (jj - cj) ** 2 <= radius_cells**2
    return float((grid[disk] != -1).mean()) if disk.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--robot", default="hamilton")
    ap.add_argument(
        "--goal-ahead",
        type=float,
        default=8.0,
        help="metres along the recorded trajectory",
    )
    ap.add_argument(
        "--lookahead", type=int, default=50, help="planner lookahead, grid cells"
    )
    ap.add_argument("--inflate", type=float, default=0.2)
    ap.add_argument("--commit", type=float, default=0.0, help="path_commitment_weight")
    ap.add_argument(
        "--stride", type=int, default=1, help="use every Nth occupancy message"
    )
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    buf, grids, track = read_bag(a.bag, a.robot)
    print(f"{len(grids)} occupancy grids, {len(track)} body poses")
    if not grids or len(track) < 2:
        sys.exit("bag has no occupancy grids or no odom->body TF")

    fb = Feedback()
    om = OccupancyMap(
        fb,
        inflate_radius_meters=a.inflate,
        use_cost_map=True,
        safe_distance=1.0,
        nearest_obstacle_cost=5.0,
    )
    planner = MidLevelPlanner(
        om, fb, lookahead_distance_grid=a.lookahead, path_commitment_weight=a.commit
    )
    rows = []
    for k, msg in enumerate(grids[:: a.stride]):
        t = stamp_ns(msg.header.stamp)
        try:
            robot_pose = lookup(buf, f"{a.robot}/odom", f"{a.robot}/body", t)
            odom_T_map = lookup(buf, f"{a.robot}/odom", msg.header.frame_id, t)
        except tf2_ros.TransformException as e:
            rows.append({"k": k, "status": f"tf: {type(e).__name__}"})
            continue
        h, w = msg.info.height, msg.info.width
        raw = np.array(msg.data, dtype=np.int8).reshape((h, w))
        o = msg.info.origin
        origin_odom = odom_T_map @ pose_to_homo(
            [o.position.x, o.position.y, o.position.z], o.orientation
        )
        om.set_grid(raw, msg.info.resolution, origin_odom, robot_pose, t)

        goal = goal_ahead(track, t, a.goal_ahead)
        if goal is None:
            rows.append({"k": k, "status": "end of trajectory"})
            continue
        start = robot_pose[:2, 3]
        t0 = time.perf_counter()
        ok, out = planner.plan_path(np.array([start, goal]))
        dt = time.perf_counter() - t0

        ci, cj = om.global_position_to_grid_cell(robot_pose[:, 3].reshape(4, 1))
        res = msg.info.resolution
        row = {
            "k": k,
            "t_s": (t - stamp_ns(grids[0].header.stamp)) / 1e9,
            "status": "ok" if ok else "a_star_failed",
            "plan_ms": dt * 1e3,
            "grid": f"{h}x{w}@{res:.2f}",
            "known_2m": known_fraction(raw, ci, cj, 2 / res),
            "known_5m": known_fraction(raw, ci, cj, 5 / res),
            "known_8m": known_fraction(raw, ci, cj, 8 / res),
        }
        if ok:
            wp = np.asarray(out.path_waypoints_metric)
            cells = [
                om.global_position_to_grid_cell(np.array([x, y, 0, 1]).reshape(4, 1))
                for x, y in wp
            ]
            vals = np.array(
                [raw[i, j] if 0 <= i < h and 0 <= j < w else 100 for i, j in cells]
            )
            row.update(
                waypoints=len(wp),
                length_m=float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
                if len(wp) > 1
                else 0.0,
                occupied_hits=int((vals > 0).sum()),
                unknown_frac=float((vals == -1).mean()),
                target_known=bool(
                    raw[om.global_position_to_grid_cell(out.target_point_metric)] != -1
                )
                if out.target_point_metric is not None
                else None,
            )
        rows.append(row)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] == "a_star_failed"]
    print(
        f"planned: {len(ok_rows)} ok, {len(failed)} A* failures, {len(rows) - len(ok_rows) - len(failed)} skipped"
    )
    if ok_rows:
        arr = lambda key: np.array([r[key] for r in ok_rows], dtype=float)  # noqa: E731
        print(
            f"plan time     mean {arr('plan_ms').mean():6.1f} ms  max {arr('plan_ms').max():6.1f} ms"
        )
        print(
            f"path length   mean {arr('length_m').mean():6.2f} m   max {arr('length_m').max():6.2f} m (goal ahead {a.goal_ahead} m)"
        )
        print(
            f"waypoints in OCCUPIED raw cells: {int(arr('occupied_hits').sum())} over {len(ok_rows)} plans"
        )
        print(
            f"unknown fraction along path: mean {arr('unknown_frac').mean():.2f}  max {arr('unknown_frac').max():.2f}"
        )
        for r_m in (2, 5, 8):
            v = arr(f"known_{r_m}m")
            print(
                f"known fraction within {r_m} m: mean {np.nanmean(v):.2f}  min {np.nanmin(v):.2f}"
            )
    if fb.messages:
        print(f"{len(fb.messages)} planner warnings, first: {fb.messages[0]}")
    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=keys)
            wri.writeheader()
            wri.writerows(rows)
        print("wrote", a.csv)
    sys.exit(1 if any(r.get("occupied_hits") for r in ok_rows) else 0)


if __name__ == "__main__":
    main()
