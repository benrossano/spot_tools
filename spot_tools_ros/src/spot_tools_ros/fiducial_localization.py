"""Anchor a recorded Hydra map to Spot's live odometry through an AprilTag.

The live stack never needs this: Hydra runs alongside the robot, so the DSG, the
TSDF occupancy and the robot's TF all live in one frame that Hydra itself ties
together with its map->odom TF. Planning against a *recorded* map breaks that,
because Spot's ``vision`` frame (published in ROS as ``<robot>/odom``) starts
fresh every boot. A fiducial that is in both the recording and the live scene
restores the link:

    map_T_odom = map_T_tag (solved offline from the recording)
               @ inv(odom_T_tag) (Spot's live world-object detection)

Everything here is plain numpy so the offline solver, the ROS node and the tests
share one set of conventions.

Frame conventions
-----------------
Boston Dynamics' ``fiducial_N`` frame, checked against the GraphNav snapshots of
the 2026-09-13 tour (``x_tag . z_up = +1.00`` in Spot's gravity-aligned vision
frame, z pointing at the robot): **x up the tag, y to the viewer's left, z out
of the tag toward the viewer**.

OpenCV ``solvePnP(..., SOLVEPNP_IPPE_SQUARE)`` with aruco's corner order
(top-left, top-right, bottom-right, bottom-left *of OpenCV's canonical marker*)
and object points ``(-h,+h) (+h,+h) (+h,-h) (-h,-h)`` gives x right and y up
along that canonical marker, z toward the camera. OpenCV's canonical
``DICT_APRILTAG_36h11`` orientation is rotated 180 deg from the AprilTag
library's, which is what Spot runs: solving the 2026-09-13 tour with the naive
+90 deg mapping put the tag's x axis at ``x . up = -0.999``. The mapping that
agrees with the robot is therefore

    x_BD = -y_cv,   y_BD = +x_cv,   z_BD = z_cv

which :func:`cv_tag_to_bd_tag` applies. Every ``map_T_tag`` this module stores
is in the BD convention, so it composes directly with the SDK's
``vision_tform_fiducial``. The solver and the live node both check the tag's
up-vector, so a convention slip shows up as a loud error rather than a robot
driving off in the wrong direction.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Iterable, Sequence

import numpy as np
import yaml

BD_FIDUCIAL_CONVENTION = (
    "bosdyn: x up the tag, y to the viewer's left, z out of the tag toward the viewer"
)
BD_STANDARD_TAG_SIZE_M = 0.146  # edge length of the fiducial printed from the BD site

# Columns are Boston Dynamics' tag axes expressed in the OpenCV tag frame:
# x_BD = -y_cv, y_BD = +x_cv, z_BD = z_cv (see the module docstring for the evidence).
CV_R_BD = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
UP = np.array([0.0, 0.0, 1.0])


def tag_up_agreement(T: np.ndarray) -> float:
    """Dot product of the tag's x axis (BD: up the tag) with world up, for a pose in a
    gravity-aligned frame. ~+1 for an upright tag, ~-1 if the convention is flipped."""
    return float(np.asarray(T, float)[:3, 0] @ UP)


def reject_inconsistent_axes(
    Ts: Sequence[np.ndarray], max_angle_deg: float = 20.0
) -> np.ndarray:
    """Mask of poses whose tag x AND z axes lie within max_angle_deg of their medians.

    The planar PnP ambiguity mirrors the tag about the viewing direction, which
    swings the normal (z) by tens of degrees and often tilts x too, far more than
    real scatter does. Checking both axes catches the wrong solutions without any
    assumption about how the tag is mounted.
    """
    keep = np.ones(len(Ts), dtype=bool)
    for col in (0, 2):
        axes = np.array([np.asarray(T, float)[:3, col] for T in Ts])
        ref = np.median(axes, axis=0)
        ref /= max(np.linalg.norm(ref), 1e-9)
        cos = np.clip(axes @ ref, -1.0, 1.0)
        keep &= np.degrees(np.arccos(cos)) <= max_angle_deg
    return keep


def cv_tag_to_bd_tag(cam_T_tag_cv: np.ndarray) -> np.ndarray:
    """Re-express an OpenCV tag pose in Boston Dynamics' fiducial frame."""
    out = np.array(cam_T_tag_cv, dtype=float, copy=True)
    out[:3, :3] = out[:3, :3] @ CV_R_BD
    return out


def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def yaw_of(T: np.ndarray) -> float:
    """Heading of the x axis in the xy plane, radians."""
    return math.atan2(T[1, 0], T[0, 0])


def planar(T: np.ndarray) -> np.ndarray:
    """Zero roll and pitch, keep x, y, z and yaw.

    Both Hydra's map frame and Spot's vision frame are gravity aligned, so any
    tilt in a solved map_T_odom is solve error. Flattening keeps it from
    leaking into the 2D planner as a sloped map.
    """
    yaw = yaw_of(T)
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = T[:3, 3]
    return out


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def map_T_odom_from_tag(
    map_T_tag: np.ndarray, odom_T_tag: np.ndarray, planar_result: bool = True
) -> np.ndarray:
    """The anchor: where the live odometry origin sits in the recorded map."""
    T = np.asarray(map_T_tag, float) @ invert(np.asarray(odom_T_tag, float))
    return planar(T) if planar_result else T


def chordal_mean_rotation(Rs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted rotation average (projection of the weighted sum onto SO(3))."""
    M = sum(float(w) * np.asarray(R, float) for R, w in zip(Rs, weights))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def average_transforms(
    Ts: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> tuple[np.ndarray, float, float]:
    """Weighted mean pose plus position (m) and yaw (deg) scatter about it."""
    Ts = [np.asarray(T, float) for T in Ts]
    if not Ts:
        raise ValueError("no transforms to average")
    w = np.ones(len(Ts)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    t = sum(wi * T[:3, 3] for wi, T in zip(w, Ts))
    R = chordal_mean_rotation([T[:3, :3] for T in Ts], w)
    mean = np.eye(4)
    mean[:3, :3] = R
    mean[:3, 3] = t
    pos_dev = np.array([np.linalg.norm(T[:3, 3] - t) for T in Ts])
    yaw0 = yaw_of(mean)
    yaw_dev = np.array([abs(wrap_angle(yaw_of(T) - yaw0)) for T in Ts])
    pos_std = float(math.sqrt(np.sum(w * pos_dev**2)))
    yaw_std = float(math.degrees(math.sqrt(np.sum(w * yaw_dev**2))))
    return mean, pos_std, yaw_std


@dataclasses.dataclass
class RobustResult:
    T: np.ndarray
    kept: np.ndarray  # bool mask over the input
    position_std_m: float
    yaw_std_deg: float

    @property
    def num_kept(self) -> int:
        return int(self.kept.sum())


def robust_average(
    Ts: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    max_pos_dev_m: float = 0.15,
    max_yaw_dev_deg: float = 5.0,
    xy_only: bool = True,
) -> RobustResult:
    """Median-anchored outlier rejection, then a weighted average of the survivors.

    A single bad PnP solution (the planar ambiguity, a blurred frame) can be far
    off; the median position and yaw are immune to a few of those, so anything
    farther than the thresholds from the medians is dropped before averaging.
    xy_only judges position in the plane: Spot's vision odometry drifts vertically
    far more than horizontally, and a 2D anchor does not care about z.
    """
    Ts = [np.asarray(T, float) for T in Ts]
    n = len(Ts)
    if n == 0:
        raise ValueError("no transforms")
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    pos = np.array([T[:2, 3] if xy_only else T[:3, 3] for T in Ts])
    yaws = np.array([yaw_of(T) for T in Ts])
    med_pos = np.median(pos, axis=0)
    # circular median: the sample whose summed wrapped distance to the others is least
    yaw_cost = [sum(abs(wrap_angle(y - o)) for o in yaws) for y in yaws]
    med_yaw = yaws[int(np.argmin(yaw_cost))]
    keep = (np.linalg.norm(pos - med_pos, axis=1) <= max_pos_dev_m) & (
        np.abs([wrap_angle(y - med_yaw) for y in yaws]) <= math.radians(max_yaw_dev_deg)
    )
    if not keep.any():
        keep = np.ones(n, dtype=bool)
    mean, pos_std, yaw_std = average_transforms(
        [T for T, k in zip(Ts, keep) if k], w[keep]
    )
    return RobustResult(mean, keep, pos_std, yaw_std)


# ----------------------------------------------------------------------------- file format


@dataclasses.dataclass
class Fiducial:
    tag_id: int
    size_m: float
    map_T_tag: np.ndarray
    frame_id: str
    num_detections: int = 0
    position_std_m: float = float("nan")
    yaw_std_deg: float = float("nan")


def matrix_to_yaml(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(T, float)]


def save_fiducials(
    path: pathlib.Path | str,
    frame_id: str,
    fiducials: Iterable[Fiducial],
    source: dict | None = None,
    extra: dict | None = None,
) -> None:
    doc = {
        "frame_id": frame_id,
        "convention": BD_FIDUCIAL_CONVENTION,
        "source": source or {},
        "fiducials": [
            {
                "tag_id": int(f.tag_id),
                "size_m": float(f.size_m),
                "num_detections": int(f.num_detections),
                "position_std_m": float(f.position_std_m),
                "yaw_std_deg": float(f.yaw_std_deg),
                "map_T_tag": matrix_to_yaml(f.map_T_tag),
            }
            for f in fiducials
        ],
    }
    if extra:
        doc.update(extra)
    pathlib.Path(path).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def load_fiducials(path: pathlib.Path | str) -> dict[int, Fiducial]:
    doc = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    if doc.get("convention") != BD_FIDUCIAL_CONVENTION:
        raise ValueError(
            f"{path}: unknown fiducial convention {doc.get('convention')!r}; "
            f"expected {BD_FIDUCIAL_CONVENTION!r}"
        )
    frame_id = doc["frame_id"]
    out = {}
    for entry in doc["fiducials"]:
        T = np.asarray(entry["map_T_tag"], float)
        if T.shape != (4, 4):
            raise ValueError(f"{path}: map_T_tag for tag {entry['tag_id']} is not 4x4")
        out[int(entry["tag_id"])] = Fiducial(
            tag_id=int(entry["tag_id"]),
            size_m=float(entry.get("size_m", BD_STANDARD_TAG_SIZE_M)),
            map_T_tag=T,
            frame_id=frame_id,
            num_detections=int(entry.get("num_detections", 0)),
            position_std_m=float(entry.get("position_std_m", float("nan"))),
            yaw_std_deg=float(entry.get("yaw_std_deg", float("nan"))),
        )
    return out
