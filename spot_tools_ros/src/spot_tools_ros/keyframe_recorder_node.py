#!/usr/bin/env python3
"""Record RGB-D keyframes in the open-set archive layout, without Hydra.

Writes exactly what Hydra's ``AgentImageExtractor`` writes into ``<run>/agents``
(``agent_<stamp>_rgb.png``, ``agent_<stamp>_depth.png`` as 16UC1 millimetres,
``agent_<stamp>_meta.json`` with ``world_T_body``, and one ``camera_calib.json``),
so the perception service can search the directory as a growing *live* stream
(``open-set-perception-service --stream robot=<dir>``) while the robot explores,
and grounding can admit objects that were not in the prior archive.

Inputs are only RGB, registered depth, camera_info and TF: ``map_frame -> body_frame``
at the image stamp gives ``world_T_body`` (the fiducial localizer anchors
``<robot>/map -> <robot>/odom`` when Hydra is not running), ``body_frame -> <depth
optical frame>`` gives ``body_T_sensor`` for ``camera_calib.json``.

A keyframe is written when the body has moved ``min_translation_m`` or turned
``min_rotation_deg`` since the last written one (the first frame is always
written), at most every ``min_interval_s``. Files are written rgb, depth, then
meta (atomically renamed), so a reader that globs ``agent_*_meta.json`` never sees
a half-written keyframe.
"""

from __future__ import annotations

import collections
import json
import pathlib
import signal
import threading

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener


def transform_to_matrix(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def image_to_bgr(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)
        return np.ascontiguousarray(img[:, :, :3] if enc == "bgra8" else img[:, :, [2, 1, 0]])
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(img if enc == "bgr8" else img[:, :, ::-1])
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported rgb encoding {msg.encoding}")


def depth_to_mm(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == "32fc1":
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        return np.clip(d, 0, 65535).astype(np.uint16)
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


class KeyframeRecorder(Node):
    def __init__(self):
        super().__init__("keyframe_recorder")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("rgb_topic", "")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("camera_info_topic", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("body_frame", "body")
        self.declare_parameter("min_translation_m", 0.5)
        self.declare_parameter("min_rotation_deg", 20.0)
        self.declare_parameter("min_interval_s", 0.5)
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("process_rate_hz", 10.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.out = pathlib.Path(p("output_dir")).expanduser()
        if not str(self.out):
            raise ValueError("output_dir is required")
        self.out.mkdir(parents=True, exist_ok=True)
        self.map_frame = p("map_frame")
        self.body_frame = p("body_frame")
        self.min_t = float(p("min_translation_m"))
        self.min_r = np.deg2rad(float(p("min_rotation_deg")))
        self.min_dt = float(p("min_interval_s"))

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.camera_info: CameraInfo | None = None
        self.calib_written = False
        self.body_T_sensor: np.ndarray | None = None
        self.lock = threading.Lock()
        self.recent: collections.deque[tuple[Image, Image]] = collections.deque(maxlen=8)  # TF lags the images
        self.last_pose: np.ndarray | None = None
        self.last_time_ns: int | None = None
        self.count = 0
        self.skipped_no_tf = 0

        self.create_subscription(CameraInfo, p("camera_info_topic"), self._on_info, 10)
        self.sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, p("rgb_topic")), Subscriber(self, Image, p("depth_topic"))],
            queue_size=10,
            slop=float(p("sync_slop_s")),
        )
        self.sync.registerCallback(self._on_pair)
        self.count_pub = self.create_publisher(Int32, "~/keyframe_count", 10)
        self.timer = self.create_timer(1.0 / float(p("process_rate_hz")), self._process)
        self.get_logger().info(
            f"recording keyframes to {self.out} (gate {self.min_t:.2f} m / "
            f"{np.rad2deg(self.min_r):.0f} deg / {self.min_dt:.1f} s; world={self.map_frame}, body={self.body_frame})"
        )

    # ------------------------------------------------------------------ inputs
    def _on_info(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg

    def _on_pair(self, rgb: Image, depth: Image):
        with self.lock:
            self.recent.append((rgb, depth))

    # ----------------------------------------------------------------- process
    def _process(self):
        with self.lock:
            pairs = list(self.recent)
        # camera_calib.json first: it needs only camera_info and the static body -> sensor TF, so it
        # exists before localization; sessions pin the live stream's calibration hash from it.
        if pairs and not self.calib_written and self.camera_info is not None:
            depth0 = pairs[-1][1]
            sensor_frame = depth0.header.frame_id or pairs[-1][0].header.frame_id
            try:
                tf = self.tf_buffer.lookup_transform(self.body_frame, sensor_frame, Time())
            except Exception:  # noqa: BLE001 - static TF not in yet
                tf = None
            if tf is not None:
                self.body_T_sensor = transform_to_matrix(tf.transform)
                self._write_calib(self.camera_info, depth0)
                self.calib_written = True
        # newest buffered pair whose map -> body TF is already in; drop everything older
        pair = None
        for i in range(len(pairs) - 1, -1, -1):
            st = Time.from_msg(pairs[i][0].header.stamp)
            if self.tf_buffer.can_transform(self.map_frame, self.body_frame, st):
                pair = pairs[i]
                with self.lock:
                    while self.recent and Time.from_msg(self.recent[0][0].header.stamp) <= st:
                        self.recent.popleft()
                break
        if pair is None:
            if pairs:
                self.skipped_no_tf += 1
                if self.skipped_no_tf in (1, 200) or self.skipped_no_tf % 2000 == 0:
                    self.get_logger().info(
                        f"no {self.map_frame} -> {self.body_frame} TF at the buffered image stamps yet "
                        f"({self.skipped_no_tf} ticks); waiting for localization"
                    )
            return
        rgb, depth = pair
        stamp = Time.from_msg(rgb.header.stamp)
        if not self.calib_written:
            return
        tf = self.tf_buffer.lookup_transform(self.map_frame, self.body_frame, stamp)
        world_T_body = transform_to_matrix(tf.transform)
        t_ns = stamp.nanoseconds
        if not self._gate(world_T_body, t_ns):
            return
        self._write_keyframe(rgb, depth, world_T_body, t_ns)

    def _gate(self, world_T_body: np.ndarray, t_ns: int) -> bool:
        if self.last_pose is None:
            return True
        if self.last_time_ns is not None and (t_ns - self.last_time_ns) < self.min_dt * 1e9:
            return False
        dt = np.linalg.norm(world_T_body[:3, 3] - self.last_pose[:3, 3])
        dR = self.last_pose[:3, :3].T @ world_T_body[:3, :3]
        angle = np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())
        return dt >= self.min_t or angle >= self.min_r

    # ------------------------------------------------------------------ output
    def _write_calib(self, info: CameraInfo, depth: Image):
        calib = {
            "fx": float(info.k[0]),
            "fy": float(info.k[4]),
            "cx": float(info.k[2]),
            "cy": float(info.k[5]),
            "width": int(info.width),
            "height": int(info.height),
            "depth_scale": 0.001,
            "depth_encoding": "16UC1_mm",
            "body_T_sensor": [float(v) for v in self.body_T_sensor.reshape(-1)],
        }
        self._atomic_json(self.out / "camera_calib.json", calib)
        self.get_logger().info(
            f"camera_calib.json written ({info.width}x{info.height}, fx {calib['fx']:.1f}, "
            f"sensor frame {depth.header.frame_id}; depth {depth.encoding} -> 16UC1 mm)"
        )

    def _write_keyframe(self, rgb: Image, depth: Image, world_T_body: np.ndarray, t_ns: int):
        stem = f"agent_{t_ns}"
        try:
            bgr = image_to_bgr(rgb)
            mm = depth_to_mm(depth)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        cv2.imwrite(str(self.out / f"{stem}_rgb.png"), bgr)
        cv2.imwrite(str(self.out / f"{stem}_depth.png"), mm)
        meta = {
            "timestamp_ns": int(t_ns),
            "frame_timestamp_ns": int(t_ns),
            "world_T_body": [float(v) for v in world_T_body.reshape(-1)],
            "rgb_file": f"{stem}_rgb.png",
            "depth_file": f"{stem}_depth.png",
            "calib": "camera_calib.json",
        }
        self._atomic_json(self.out / f"{stem}_meta.json", meta)
        self.last_pose = world_T_body
        self.last_time_ns = t_ns
        self.count += 1
        self.count_pub.publish(Int32(data=self.count))
        x, y, z = world_T_body[:3, 3]
        yaw = np.degrees(np.arctan2(world_T_body[1, 0], world_T_body[0, 0]))
        self.get_logger().info(
            f"keyframe {self.count}: {stem} at ({x:.2f}, {y:.2f}, {z:.2f}) yaw {yaw:.0f} deg"
        )

    @staticmethod
    def _atomic_json(path: pathlib.Path, payload: dict):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)


def _raise_keyboard_interrupt(*_):
    raise KeyboardInterrupt


def main(args=None):
    # ros2 launch stops children with SIGTERM; treat it like Ctrl-C so the finally block runs.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    rclpy.init(args=args)
    try:
        node = KeyframeRecorder()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.get_logger().info(f"{node.count} keyframes in {node.out}")
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
