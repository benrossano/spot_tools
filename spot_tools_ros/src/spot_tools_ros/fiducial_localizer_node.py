#!/usr/bin/env python3
"""Publish <robot>/map -> <robot>/odom from an AprilTag, for planning on a recorded map.

Stands in for Hydra's backend TF when Hydra is *not* running: it watches Spot's
world-object service for a fiducial listed in a ``fiducials.yaml`` (written by
scripts/solve_map_fiducial.py from the run that produced the map), and once the
tag has been seen well enough it broadcasts the static transform

    map_T_odom = map_T_tag @ inv(vision_T_tag)

so the executor's occupancy grid, the planner's Follow paths (both in the map
frame) and the robot's live pose (odom frame) line up. Nothing is published until
the tag is seen, which gates a run exactly like GraphNav's initial localization.

Note that spot_sensor_node publishes Spot's ``vision`` frame as ROS ``<robot>/odom``
(its ``frame_remaps`` default), so the tag pose is read in ``vision`` here. Keep
``spot_parent_frame`` in step with that node if you change the remap.

Read only: no lease, no E-Stop, safe to run next to the tablet.
"""

from __future__ import annotations

import math
import time

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rclpy
import tf2_ros
from bosdyn.api import world_object_pb2
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.world_object import WorldObjectClient
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool

from spot_tools_ros.fiducial_localization import (
    load_fiducials,
    map_T_odom_from_tag,
    robust_average,
    tag_up_agreement,
    yaw_of,
)

STATUS_NAMES = {
    v: k for k, v in world_object_pb2.AprilTagProperties.AprilTagPoseStatus.items()
}


def _matrix_to_transform(T, parent, child, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    msg.transform.translation.x = float(T[0, 3])
    msg.transform.translation.y = float(T[1, 3])
    msg.transform.translation.z = float(T[2, 3])
    msg.transform.rotation.x = float(q[0])
    msg.transform.rotation.y = float(q[1])
    msg.transform.rotation.z = float(q[2])
    msg.transform.rotation.w = float(q[3])
    return msg


def _matrix_to_pose(T, frame, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


class FiducialLocalizerNode(Node):
    def __init__(self):
        super().__init__("fiducial_localizer_node")
        ns = self.get_namespace().strip("/")
        prefix = f"{ns}/" if ns else ""

        p = self._param
        self.map_frame = p("map_frame", f"{prefix}map").string_value
        self.odom_frame = p("odom_frame", f"{prefix}odom").string_value
        self.spot_parent_frame = p("spot_parent_frame", "vision").string_value
        fiducials_file = p("fiducials_file", "").string_value
        self.tag_id = p("tag_id", -1).integer_value
        self.poll_rate_hz = p("poll_rate_hz", 5.0).double_value
        self.min_observations = p("min_observations", 5).integer_value
        self.max_range_m = p("max_range_m", 3.5).double_value
        self.max_pos_dev_m = p("max_pos_dev_m", 0.10).double_value
        self.max_yaw_dev_deg = p("max_yaw_dev_deg", 3.0).double_value
        self.use_filtered_frame = p("use_filtered_frame", True).bool_value
        self.planar = p("planar", True).bool_value
        # freeze: publish once and stop sampling (GraphNav-style initialization)
        # continuous: keep sampling, re-publish when the anchor moves > republish_jump_m
        self.mode = p("mode", "freeze").string_value
        self.republish_jump_m = p("republish_jump_m", 0.05).double_value
        self.window = p("window", 20).integer_value
        # The recorded map_T_tag and the live vision_T_tag are both in gravity-aligned
        # frames, so the tag's up axis must agree between them. A sign flip means the
        # offline solve and Spot disagree on the fiducial convention, and anchoring on it
        # would mirror the map; refuse rather than drive.
        self.require_consistent_up = p("require_consistent_up", True).bool_value
        self.min_up_agreement = p("min_up_agreement", 0.8).double_value

        if not fiducials_file:
            raise ValueError("fiducials_file parameter is required")
        fiducials = load_fiducials(fiducials_file)
        if self.tag_id < 0:
            self.tag_id = sorted(fiducials)[0]
        if self.tag_id not in fiducials:
            raise ValueError(f"tag {self.tag_id} not in {fiducials_file} (has {sorted(fiducials)})")
        self.fiducial = fiducials[self.tag_id]
        if self.fiducial.frame_id != self.map_frame:
            self.get_logger().warn(
                f"{fiducials_file} is in frame {self.fiducial.frame_id!r} but publishing as "
                f"{self.map_frame!r}; make sure the occupancy grid and plan use the same frame"
            )
        T = self.fiducial.map_T_tag
        self.get_logger().info(
            f"anchoring on tag {self.tag_id} at ({T[0, 3]:.2f}, {T[1, 3]:.2f}, {T[2, 3]:.2f}) in {self.map_frame} "
            f"(solved from {self.fiducial.num_detections} detections, "
            f"{self.fiducial.position_std_m * 100:.0f} cm / {self.fiducial.yaw_std_deg:.1f} deg scatter)"
        )

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.static_pub = tf2_ros.StaticTransformBroadcaster(self)
        self.localized_pub = self.create_publisher(Bool, "~/localized", latched)
        self.tag_pose_pub = self.create_publisher(PoseStamped, "~/tag_in_map", latched)
        self.anchor_pub = self.create_publisher(PoseStamped, "~/map_T_odom", latched)
        self.localized_pub.publish(Bool(data=False))
        self.tag_pose_pub.publish(_matrix_to_pose(T, self.map_frame, self.get_clock().now().to_msg()))

        self.robot = self._connect()
        self.world_objects = self.robot.ensure_client(WorldObjectClient.default_service_name)

        self.samples: list[np.ndarray] = []
        self.weights: list[float] = []
        self.published: np.ndarray | None = None
        self.frozen = False
        self._last_status_log = 0.0
        self._last_seen_log = 0.0
        self.timer = self.create_timer(1.0 / self.poll_rate_hz, self.poll)
        self.get_logger().info(
            f"waiting for tag {self.tag_id} within {self.max_range_m:.1f} m "
            f"({self.min_observations} good observations needed, mode={self.mode})"
        )

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value()

    def _connect(self):
        setup_logging = self._param("robot.setup_logging", False).bool_value
        should_retry = self._param("robot.should_retry", True).bool_value
        ip = self._param("robot.ip", "").string_value
        username = self._param("robot.username", "").string_value
        password = self._param("robot.password", "").string_value
        for key, value in (("robot.ip", ip), ("robot.username", username), ("robot.password", password)):
            if not value:
                raise ValueError(f"{key} parameter is required")
        name = self.get_name().replace("/", "_").lstrip("_")
        sdk = bosdyn.client.create_standard_sdk(name)
        robot = sdk.create_robot(ip)
        if setup_logging:
            bosdyn.client.util.setup_logging()
        while True:
            try:
                robot.authenticate(username, password)
                break
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"authentication failed: {e}")
                if not should_retry:
                    raise
                time.sleep(2.0)
        robot.time_sync.wait_for_sync(10)
        return robot

    # ------------------------------------------------------------------ sampling
    def _observe(self) -> tuple[np.ndarray, float] | None:
        """One vision_T_tag sample (plus a weight) or None when the tag is not usable."""
        resp = self.world_objects.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]
        )
        for obj in resp.world_objects:
            if not obj.HasField("apriltag_properties"):
                continue
            props = obj.apriltag_properties
            if props.tag_id != self.tag_id:
                continue
            if self.use_filtered_frame and props.frame_name_fiducial_filtered:
                frame, status = props.frame_name_fiducial_filtered, props.fiducial_filtered_pose_status
            else:
                frame, status = props.frame_name_fiducial, props.fiducial_pose_status
            if status != world_object_pb2.AprilTagProperties.STATUS_OK:
                self._log_throttled(f"tag {self.tag_id} seen but pose status {STATUS_NAMES.get(status, status)}")
                return None
            try:
                se3 = get_a_tform_b(obj.transforms_snapshot, self.spot_parent_frame, frame)
            except Exception as e:  # noqa: BLE001
                self._log_throttled(f"no {self.spot_parent_frame}->{frame} in the detection snapshot: {e}")
                return None
            if se3 is None:
                return None
            vision_T_tag = se3.to_matrix()
            try:
                body_T_tag = get_a_tform_b(obj.transforms_snapshot, "body", frame)
                rng = float(np.linalg.norm(body_T_tag.to_matrix()[:3, 3]))
            except Exception:  # noqa: BLE001
                rng = float("nan")
            if math.isfinite(rng) and rng > self.max_range_m:
                self._log_throttled(f"tag {self.tag_id} at {rng:.1f} m, beyond max_range_m={self.max_range_m:.1f}")
                return None
            live_up = tag_up_agreement(vision_T_tag)
            map_up = tag_up_agreement(self.fiducial.map_T_tag)
            if self.require_consistent_up and live_up * map_up < 0 and abs(live_up) > self.min_up_agreement:
                self._log_throttled(
                    f"REFUSING to anchor: tag x axis . up is {live_up:+.2f} live but {map_up:+.2f} in the "
                    f"recorded map. The fiducial convention or the tag mounting changed; re-run "
                    f"scripts/solve_map_fiducial.py (or set require_consistent_up:=false to override).",
                    period_s=2.0,
                )
                return None
            # closer detections are better conditioned; unknown range gets a neutral weight
            weight = 1.0 / (rng * rng) if math.isfinite(rng) and rng > 0.1 else 1.0
            return vision_T_tag, weight
        return None

    def poll(self):
        if self.frozen:
            return
        try:
            obs = self._observe()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"world object query failed: {e}")
            return
        now = time.monotonic()
        if obs is None:
            if now - self._last_status_log > 5.0 and not self.samples:
                self.get_logger().info(f"tag {self.tag_id} not visible yet")
                self._last_status_log = now
            return
        vision_T_tag, weight = obs
        self.samples.append(map_T_odom_from_tag(self.fiducial.map_T_tag, vision_T_tag, self.planar))
        self.weights.append(weight)
        if len(self.samples) > self.window:
            del self.samples[0]
            del self.weights[0]
        if now - self._last_seen_log > 2.0:
            self.get_logger().info(f"tag {self.tag_id} visible, {len(self.samples)}/{self.min_observations} observations")
            self._last_seen_log = now
        if len(self.samples) < self.min_observations:
            return

        res = robust_average(self.samples, self.weights, self.max_pos_dev_m, self.max_yaw_dev_deg)
        if res.num_kept < self.min_observations:
            self._log_throttled(
                f"only {res.num_kept}/{len(self.samples)} observations agree "
                f"(scatter {res.position_std_m * 100:.0f} cm / {res.yaw_std_deg:.1f} deg); waiting"
            )
            return
        if self.published is not None:
            jump = float(np.linalg.norm(res.T[:2, 3] - self.published[:2, 3]))
            if jump < self.republish_jump_m:
                return
        self._publish(res)
        if self.mode == "freeze":
            self.frozen = True
            self.get_logger().info("anchor frozen; restart the node to re-localize")

    def _publish(self, res):
        T = res.T
        stamp = self.get_clock().now().to_msg()
        self.static_pub.sendTransform(_matrix_to_transform(T, self.map_frame, self.odom_frame, stamp))
        self.anchor_pub.publish(_matrix_to_pose(T, self.map_frame, stamp))
        self.localized_pub.publish(Bool(data=True))
        self.published = T
        self.get_logger().info(
            f"LOCALIZED: {self.map_frame} -> {self.odom_frame} = "
            f"({T[0, 3]:.3f}, {T[1, 3]:.3f}, {T[2, 3]:.3f}) m, yaw {math.degrees(yaw_of(T)):.2f} deg "
            f"from {res.num_kept} observations (scatter {res.position_std_m * 100:.1f} cm / {res.yaw_std_deg:.2f} deg)"
        )

    def _log_throttled(self, text, period_s=5.0):
        now = time.monotonic()
        if now - self._last_status_log > period_s:
            self.get_logger().warn(text)
            self._last_status_log = now


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FiducialLocalizerNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
