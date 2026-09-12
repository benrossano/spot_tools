"""Interface for moving the spot frame."""

import time
from typing import Tuple

import numpy as np
import shapely
from bosdyn.api import geometry_pb2
from bosdyn.api.spot import robot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from numpy.typing import ArrayLike

# Global constants to define spot motion speed
MAX_LINEAR_VEL = 0.75
MAX_ROTATION_VEL = 0.65


def navigate_to_relative_pose(
    spot,
    body_tform_goal: math_helpers.SE2Pose,
    max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
    timeout: float = 20.0,
):
    # Get an SE2 pose for vision_tform_body to convert the body-based command to a non-moving frame
    # that can be issued to the robot.
    robot_state = spot.get_state()
    transforms = robot_state.kinematic_state.transforms_snapshot

    assert str(transforms) != ""

    vision_tform_body = get_se2_a_tform_b(
        transforms, VISION_FRAME_NAME, BODY_FRAME_NAME
    )
    vision_tform_goal = vision_tform_body * body_tform_goal

    navigate_to_absolute_pose(spot, waypoint=vision_tform_goal)


def navigate_to_absolute_pose(
    spot, waypoint, frame_name=VISION_FRAME_NAME, stairs=False
):
    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )

    params = RobotCommandBuilder.mobility_params(
        stair_hint=stairs, locomotion_hint=robot_command_pb2.LocomotionHint.HINT_AUTO
    )

    max_vel_linear = geometry_pb2.Vec2(x=MAX_LINEAR_VEL, y=MAX_LINEAR_VEL)
    max_vel_se2 = geometry_pb2.SE2Velocity(
        linear=max_vel_linear, angular=MAX_ROTATION_VEL
    )
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params.vel_limit.CopyFrom(vel_limit)

    state = spot.get_state()
    manipulator_state = state.manipulator_state

    gripper_holding = manipulator_state.is_gripper_holding_item

    if gripper_holding:
        arm_joint_freeze_command = RobotCommandBuilder.arm_joint_freeze_command()
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=waypoint.x,
            goal_y=waypoint.y,
            goal_heading=waypoint.angle,
            frame_name=frame_name,
            params=params,
            build_on_command=arm_joint_freeze_command,
        )
    else:
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=waypoint.x,
            goal_y=waypoint.y,
            goal_heading=waypoint.angle,
            frame_name=frame_name,
            params=params,
        )
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + end_time
    )
    return cmd_id


def follow_trajectory_continuous(
    spot,
    waypoints_list: ArrayLike,
    lookahead_distance: float,
    goal_tolerance: float,
    timeout: float,
    mid_level_planner,
    frame_name=VISION_FRAME_NAME,
    stairs=False,
    feedback=None,
    progress_timeout=None,
    progress_epsilon=0.1,
) -> bool:
    """
    Follows a trajectory by commanding the robot to move to each waypoint in the specified frame.
    Args:
        waypoints_list (ArrayLike): List of list of positions in the format [x, y].
        frame_name (str): Desired frame for the trajectory.
        robot_command_client (RobotCommandClient): Client for sending robot commands.
        robot_state_client (RobotStateClient): Client for receiving robot state.
        occupancy_grid_subscriber (Union[spot_ros_utils.OccupancyGridSubscriber, None], optional): Subscriber for occupancy grid updates. Defaults to None.
        stairs (bool, optional): Flag indicating whether the robot is navigating stairs. Defaults to False.
        progress_timeout (float, optional): Abort if the distance to the goal has not shrunk by
            `progress_epsilon` (m) for this many seconds. None keeps only the global timeout.
    Returns:
        bool: True if the trajectory is successfully followed, False otherwise.
    """

    spot.robot.ensure_client(RobotCommandClient.default_service_name)

    end_pt = waypoints_list[-1, :2]
    t0 = time.time()
    best_distance_from_end = np.inf
    last_progress_time = t0
    rate = 10
    # TODO: reactive loop, yeild out the loop to get info
    while 1:
        curr_time = time.time()
        feedback.print("INFO", f"Timeout progress {curr_time - t0} / {timeout}")
        if curr_time - t0 > timeout:
            # TODO: I think we need to tell Spot to stop?
            # TODO: Also, we should probably have a finer-grained
            # check about making progress
            feedback.print("INFO", "follow_trajectory_continuous timeout")
            return False

        # update path every (couple?) loop
        if mid_level_planner is not None:
            mlp_success, planning_output = mid_level_planner.plan_path(
                waypoints_list[:, :2]
            )
            path = planning_output.path_shapely
            path_wp = planning_output.path_waypoints_metric
            target_point_metric = planning_output.target_point_metric
            target_point_global_path_metric = (
                planning_output.global_path_target_point_metric
            )

            if feedback is not None and target_point_metric is not None:
                feedback.print("INFO", f"target_point_metric: {target_point_metric}")
                feedback.path_follow_MLP_feedback(
                    path_wp,
                    target_point_metric,
                    target_point_global_path_metric,
                    waypoints_list[-1],
                )
        else:
            # No mid-level planner (the signature allows None, and SpotExecutor passes
            # its planner straight through). Fall through to the existing failed-plan
            # branch, which follows the high-level path directly.
            mlp_success = False

        if not mlp_success:
            feedback.print(
                "INFO", "Mid-level planner failed, following high-level path directly"
            )
            path = shapely.LineString(waypoints_list[:, :2])

        tform_body_in_vision = spot.get_pose()
        distance_from_end = np.linalg.norm(
            end_pt - np.array([tform_body_in_vision[0], tform_body_in_vision[1]])
        )
        if distance_from_end < best_distance_from_end - progress_epsilon:
            best_distance_from_end = distance_from_end
            last_progress_time = curr_time
        elif (
            progress_timeout is not None
            and curr_time - last_progress_time > progress_timeout
        ):
            feedback.print(
                "INFO",
                f"No progress toward goal for {progress_timeout}s (best {best_distance_from_end:.2f}m), aborting follow",
            )
            return False

        if distance_from_end < goal_tolerance:
            feedback.print("INFO", "Spot reached end of path")
            endpoint = math_helpers.SE2Pose(
                x=tform_body_in_vision[0],
                y=tform_body_in_vision[1],
                angle=tform_body_in_vision[2],
            )
            navigate_to_absolute_pose(spot, endpoint, frame_name, stairs=stairs)
            break

        # 1. project to current path distance
        current_point = shapely.Point(tform_body_in_vision[0], tform_body_in_vision[1])
        progress_distance = shapely.line_locate_point(path, current_point)
        progress_point = shapely.line_interpolate_point(path, progress_distance)

        # 2. get line point at lookahead
        target_distance = progress_distance + lookahead_distance
        target_point = shapely.line_interpolate_point(path, target_distance)

        yaw_angle = np.arctan2(
            target_point.y - current_point.y, target_point.x - current_point.x
        )

        if feedback is not None:
            # get data back out
            # TODO: new function for MLP
            feedback.path_following_progress_feedback(progress_point, target_point)

        # 3. send command
        current_waypoint = math_helpers.SE2Pose(
            x=target_point.x, y=target_point.y, angle=yaw_angle
        )
        feedback.print("INFO", f"Navigating to waypoint {current_waypoint}")

        if feedback is not None and not feedback.plan_valid:
            feedback.print("INFO", "Plan invalidated by lease change, exiting follow")
            return False

        navigate_to_absolute_pose(spot, current_waypoint, frame_name, stairs=stairs)
        time.sleep(1 / rate)
    return True


def turn_to_point(spot, current_position, target_position):
    d = [
        target_position[0] - current_position[0],
        target_position[1] - current_position[1],
    ]
    angle = np.arctan2(d[1], d[0])
    waypoint = math_helpers.SE2Pose(
        x=current_position[0], y=current_position[1], angle=angle
    )
    navigate_to_absolute_pose(spot, waypoint, "vision", stairs=False)
