#!/usr/bin/env python3
"""Record the robot's pose in a fixed frame to CSV, from TF, until interrupted.

    log_robot_trajectory.py <out.csv> [--parent hamilton/map] [--child hamilton/base_link] [--rate 5]

Columns: t_unix_s, x, y, z, yaw_deg. One row per sample in which the transform was
available; gaps mean TF was missing (for example before the fiducial localizer anchored).
Runs on the system python with ROS sourced; no venv needed.
"""

import argparse
import csv
import math
import sys
import time

import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--parent", default="hamilton/map")
    ap.add_argument("--child", default="hamilton/base_link")
    ap.add_argument("--rate", type=float, default=5.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("robot_trajectory_logger")
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer, node, spin_thread=False)
    period = 1.0 / args.rate
    rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_unix_s", "x", "y", "z", "yaw_deg"])
        f.flush()
        next_t = time.time()
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                now = time.time()
                if now < next_t:
                    continue
                next_t += period
                try:
                    tf = buffer.lookup_transform(args.parent, args.child, rclpy.time.Time(), Duration(seconds=0.0))
                except tf2_ros.TransformException:
                    continue
                t, q = tf.transform.translation, tf.transform.rotation
                yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
                writer.writerow([f"{now:.3f}", f"{t.x:.3f}", f"{t.y:.3f}", f"{t.z:.3f}", f"{yaw:.1f}"])
                rows += 1
                if rows % 50 == 0:
                    f.flush()
        except KeyboardInterrupt:
            pass
        finally:
            f.flush()
    print(f"{rows} samples -> {args.out}", file=sys.stderr)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
