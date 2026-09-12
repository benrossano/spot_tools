import math
import pickle as pkl
import time

import cv2
import numpy as np
from bosdyn.api import basic_command_pb2, geometry_pb2, manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import (
    ManipulationApiFeedbackRequest,
    ManipulationApiRequest,
    WalkToObjectInImage,
)
from bosdyn.api.spot import door_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.door import DoorClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from spot_skills.primitives import execute_recovery_action
from spot_skills.skills_definitions import OpenDoorFeedback, OpenDoorParams

DOOR_ID = 0
HANDLE_ID = 1


class RequestManager:
    """Helper object for displaying side by side images to the user and requesting user selected
    touchpoints. This class handles the bookkeeping for converting between a touchpoints of side by
    side display image of the frontleft and frontright fisheye images and the individual images.

    Args:
        image_dict: (dict) Dictionary from image source name to (image proto, CV2 image) pairs.
        window_name: (str) Name of display window.
    """

    def __init__(self, image_dict, model):
        self.image_dict = image_dict
        self.model = model
        self.handle_position_side_by_side = None
        self.side_by_side = None
        self.annotated_side_by_side = None
        self.clicked_source = None

    def attributes_set(self):
        """bool: True if handle and hinge position set."""
        return (
            self.handle_position_side_by_side is not None
            and self.clicked_source is not None
            and self.side_by_side.all() is not None
        )

    def detect_door_handle_in_image(self, img, class_id=HANDLE_ID):
        results = self.model.predict(img, conf=0.1)
        # results[0].show()
        detected_handles = []

        for detection in results[0].boxes:
            # Get normalized xywh (YOLO format) and class confidence
            cls = detection.cls.item()

            if cls == class_id:
                detected_handles.append(detection)

        return detected_handles

    def detect_door_handle_in_side_by_side(
        self, rotated_fl_img, rotated_fr_img, fl_detected_handles, fr_detected_handles
    ):
        # If the handle was not detected in either image.
        self.side_by_side = np.hstack([rotated_fr_img, rotated_fl_img])

        if (len(fl_detected_handles) + len(fr_detected_handles)) == 0:
            cv2.imshow("Debug", self.side_by_side)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            raise Exception("No door handles detected.")

        detected_side = None
        max_conf = 0
        max_detection = None

        # If the handle was detected only in the left image
        if len(fl_detected_handles) > 0 and len(fr_detected_handles) == 0:
            detected_side = "LEFT"
            # Find the handle with the highest confidence.
            for detection in fl_detected_handles:
                if detection.conf > max_conf:
                    max_conf = detection.conf
                    max_detection = detection

        # If the handle was detected
        elif len(fl_detected_handles) == 0 and len(fr_detected_handles) > 0:
            detected_side = "RIGHT"
            for detection in fr_detected_handles:
                if detection.conf > max_conf:
                    max_conf = detection.conf
                    max_detection = detection

        # If the handle was detected in both images

        else:
            detected_side = "LEFT"
            for detection in fl_detected_handles:
                if detection.conf > max_conf:
                    max_conf = detection.conf
                    max_detection = detection

            for detection in fr_detected_handles:
                if detection.conf > max_conf:
                    detected_side = "RIGHT"
                    max_conf = detection.conf
                    max_detection = detection

        # Set the x and y value
        if detected_side == "LEFT":
            detected_x = max_detection.xywh[0][0] + np.shape(rotated_fl_img)[1]
            detected_y = max_detection.xywh[0][1]
            self.clicked_source = "frontleft_fisheye_image"

        elif detected_side == "RIGHT":
            detected_x = max_detection.xywh[0][0]
            detected_y = max_detection.xywh[0][1]
            self.clicked_source = "frontright_fisheye_image"

        # SET THE HANDLE POSITION
        self.handle_position_side_by_side = (detected_x, detected_y)
        print(
            f"Detected door handle in {detected_side} at ({detected_x}, {detected_y})"
        )

        # Construct side by side image and circle the detected door handle
        self.annotated_side_by_side = np.hstack([rotated_fr_img, rotated_fl_img])
        c = (255, 255, 0)
        cv2.circle(
            self.annotated_side_by_side, (int(detected_x), int(detected_y)), 30, c, 5
        )
        cv2.imshow("Debug", self.annotated_side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_handle_and_hinge(self):
        """Use the model and the side by side image to detect the door handle and hinge."""
        # This method should set self.handle_position_side_by_side and self.hinge_position_side_by_side

        # These images should NOT be rotated.
        fl_img = self.image_dict["frontleft_fisheye_image"][1]
        fr_img = self.image_dict["frontright_fisheye_image"][1]

        rotated_fl_img = cv2.rotate(fl_img, cv2.ROTATE_90_CLOCKWISE)
        rotated_fr_img = cv2.rotate(fr_img, cv2.ROTATE_90_CLOCKWISE)

        fl_detected_handles = self.detect_door_handle_in_image(rotated_fl_img)
        fr_detected_handles = self.detect_door_handle_in_image(rotated_fr_img)

        self.detect_door_handle_in_side_by_side(
            rotated_fl_img, rotated_fr_img, fl_detected_handles, fr_detected_handles
        )

    def get_walk_to_object_in_image_request(self, debug):
        """Convert from touchpoints in side by side image to a WalkToObjectInImage request.
        Optionally show debug image of touch point.

        Args:
            debug (bool): Show intermediate debug image.

        Returns:
            ManipulationApiRequest: Request with WalkToObjectInImage info populated.
        """

        # Figure out which source the user actually clicked.
        height, width, _ = self.side_by_side.shape
        if self.handle_position_side_by_side[0] > width / 2:
            assert self.clicked_source == "frontleft_fisheye_image"
            rotated_pixel = self.handle_position_side_by_side
            rotated_pixel = (rotated_pixel[0] - width / 2, rotated_pixel[1])
        else:
            assert self.clicked_source == "frontright_fisheye_image"
            rotated_pixel = self.handle_position_side_by_side

        # Undo pixel rotation by rotation 90 deg CCW.
        manipulation_cmd = WalkToObjectInImage()
        th = -math.pi / 2
        xm = width / 4
        ym = height / 2
        x = rotated_pixel[0] - xm
        y = rotated_pixel[1] - ym
        manipulation_cmd.pixel_xy.x = math.cos(th) * x - math.sin(th) * y + ym
        manipulation_cmd.pixel_xy.y = math.sin(th) * x + math.cos(th) * y + xm

        # Populate the rest of the Manip API request.
        clicked_image_proto = self.image_dict[self.clicked_source][0]
        manipulation_cmd.frame_name_image_sensor = (
            clicked_image_proto.shot.frame_name_image_sensor
        )
        manipulation_cmd.transforms_snapshot_for_camera.CopyFrom(
            clicked_image_proto.shot.transforms_snapshot
        )
        manipulation_cmd.camera_model.CopyFrom(clicked_image_proto.source.pinhole)
        door_search_dist_meters = 1.25
        manipulation_cmd.offset_distance.value = door_search_dist_meters

        request = ManipulationApiRequest(walk_to_object_in_image=manipulation_cmd)
        return request

    @property
    def vision_tform_sensor(self):
        """Look up vision_tform_sensor for sensor which user clicked.

        Returns:
            math_helpers.SE3Pose
        """
        clicked_image_proto = self.image_dict[self.clicked_source][0]
        frame_name_image_sensor = clicked_image_proto.shot.frame_name_image_sensor
        snapshot = clicked_image_proto.shot.transforms_snapshot
        return frame_helpers.get_a_tform_b(
            snapshot, frame_helpers.VISION_FRAME_NAME, frame_name_image_sensor
        )


def walk_to_object_in_image(robot, request_manager, debug):
    """Command the robot to walk toward user selected point. The WalkToObjectInImage feedback
    reports a raycast result, converting the 2D touchpoint to a 3D location in space.

    Args:
        robot: (Robot) Interface to Spot robot.
        request_manager: (RequestManager) Object for bookkeeping user touch points.
        debug (bool): Show intermediate debug image.

    Returns:
        ManipulationApiResponse: Feedback from WalkToObjectInImage request.
    """
    manip_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    manipulation_api_request = request_manager.get_walk_to_object_in_image_request(
        debug
    )

    # Send a manipulation API request. Using the points selected by the user, the robot will
    # walk toward the door handle.
    robot.logger.info("Walking toward door...")
    response = manip_client.manipulation_api_command(manipulation_api_request)

    # Check feedback to verify the robot walks to the handle. The service will also return a
    # FrameTreeSnapshot that contain a walkto_raycast_intersection point.
    command_id = response.manipulation_cmd_id
    feedback_request = ManipulationApiFeedbackRequest(manipulation_cmd_id=command_id)
    timeout_sec = 15.0
    end_time = time.time() + timeout_sec
    while time.time() < end_time:
        response = manip_client.manipulation_api_feedback_command(feedback_request)
        assert response.manipulation_cmd_id == command_id, (
            "Got feedback for wrong command."
        )
        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            return response
    raise Exception("Manip command timed out. Try repositioning the robot.")


# To open the door,
def open_door(robot, request_manager, snapshot, parameters, feedback):
    """Command the robot to automatically open a door via the door service API.

    Args:
        robot: (Robot) Interface to Spot robot.
        request_manager: (RequestManager) Object for bookkeeping user touch points.
        snapshot: (TransformSnapshot) Snapshot from the WalkToObjectInImage command which contains
            the 3D location reported from a raycast based on the user hinge touch point.
    """
    robot.logger.info("Opening door...")

    # Using the raycast intersection point and the
    vision_tform_raycast = frame_helpers.get_a_tform_b(
        snapshot, frame_helpers.VISION_FRAME_NAME, frame_helpers.RAYCAST_FRAME_NAME
    )
    vision_tform_sensor = request_manager.vision_tform_sensor
    raycast_point_wrt_vision = vision_tform_raycast.get_translation()
    ray_from_camera_to_object = (
        raycast_point_wrt_vision - vision_tform_sensor.get_translation()
    )
    ray_from_camera_to_object_norm = np.sqrt(np.sum(ray_from_camera_to_object**2))
    ray_from_camera_normalized = (
        ray_from_camera_to_object / ray_from_camera_to_object_norm
    )

    auto_cmd = door_pb2.DoorCommand.AutoGraspCommand()
    auto_cmd.frame_name = frame_helpers.VISION_FRAME_NAME
    search_dist_meters = 0.25
    search_ray = search_dist_meters * ray_from_camera_normalized
    search_ray_start_in_frame = raycast_point_wrt_vision - search_ray
    auto_cmd.search_ray_start_in_frame.CopyFrom(
        geometry_pb2.Vec3(
            x=search_ray_start_in_frame[0],
            y=search_ray_start_in_frame[1],
            z=search_ray_start_in_frame[2],
        )
    )

    search_ray_end_in_frame = raycast_point_wrt_vision + search_ray
    auto_cmd.search_ray_end_in_frame.CopyFrom(
        geometry_pb2.Vec3(
            x=search_ray_end_in_frame[0],
            y=search_ray_end_in_frame[1],
            z=search_ray_end_in_frame[2],
        )
    )

    auto_cmd.hinge_side = parameters.hinge_side
    auto_cmd.swing_direction = parameters.swing_direction

    door_command = door_pb2.DoorCommand.Request(auto_grasp_command=auto_cmd)
    request = door_pb2.OpenDoorCommandRequest(door_command=door_command)

    # Command the robot to open the door.
    door_client = robot.ensure_client(DoorClient.default_service_name)
    response = door_client.open_door(request)

    feedback_request = door_pb2.OpenDoorFeedbackRequest()
    feedback_request.door_command_id = response.door_command_id
    feedback_response = door_client.open_door_feedback(feedback_request)

    timeout_sec = 20.0
    end_time = time.time() + timeout_sec

    while time.time() < end_time:
        feedback_response = door_client.open_door_feedback(feedback_request)
        if (
            feedback_response.status
            != basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING
        ):
            raise Exception("Door command reported status ")
        if (
            feedback_response.feedback.status
            == door_pb2.DoorCommand.Feedback.STATUS_COMPLETED
        ):
            robot.logger.info("Opened door.")
            feedback.opened_door = True
            return feedback_response.feedback.status
        time.sleep(0.5)

    # If the door command times out, tell robot to open gripper and give up trying to open the door.
    raise Exception("Door command timed out. Try repositioning the robot.")


def execute_open_door(
    spot,
    model_path,
    parameters=OpenDoorParams(),
    feedback=OpenDoorFeedback(),
    initial_pose=None,
):
    # PHASE 1: Approach the door.
    # S1: Stand the robot.
    spot.stand()

    # S2: Pitch up the robot.
    spot.pitch_up()

    # S3: Capture the images from the two front cameras.
    fl_image_request, fl_img = spot.get_image_RGB(view="frontleft_fisheye_image")
    fr_image_request, fr_img = spot.get_image_RGB(view="frontright_fisheye_image")

    image_dict = {
        "frontleft_fisheye_image": (fl_image_request, fl_img),
        "frontright_fisheye_image": (fr_image_request, fr_img),
    }

    # S4: Load the model and pass it to the request manager, which detects the door handle and hinge
    from ultralytics import YOLOWorld

    model = YOLOWorld(model_path)

    request_manager = RequestManager(image_dict, model)

    try:
        request_manager.get_handle_and_hinge()
    except Exception as ex:
        print(ex)
        spot.stand()
        return
    # assert request_manager.attributes_set(), 'Failed to get user input for handle and hinge.'

    feedback.detected_door = True
    feedback.ego_view = cv2.rotate(fr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    feedback.handle_detection = request_manager.annotated_side_by_side

    robot = spot.robot

    # Tell the robot to walk toward the door.
    try:
        manipulation_feedback = walk_to_object_in_image(
            robot, request_manager, debug=False
        )
    except Exception as ex:
        print(ex)
        execute_recovery_action(spot, recover_arm=False, absolute_pose=initial_pose)
        return

    feedback.walked_to_door = True
    time.sleep(3.0)

    # The ManipulationApiResponse for the WalkToObjectInImage command returns a transform snapshot
    # that contains where user clicked door handle point intersects the world. We use this
    # intersection point to execute the door command.
    snapshot = manipulation_feedback.transforms_snapshot_manipulation_data

    # Execute the door command.
    try:
        open_door(robot, request_manager, snapshot, parameters, feedback)
    except Exception as ex:
        print(ex)
        execute_recovery_action(spot, recover_arm=True, absolute_pose=initial_pose)
        return


def test_detect(model_path):
    from ultralytics import YOLOWorld

    model = YOLOWorld(model_path)

    for idx in range(0, 32):
        image_dict = pkl.load(
            open(f"/home/aaron/spot_tools/data/images/image_dict_{idx}.pkl", "rb")
        )
        # fl_img = pkl.load(open('../../examples/fl_img.pkl', 'rb'))
        # fr_img = pkl.load(open('../../examples/fr_img.pkl', 'rb'))

        # fl_img = cv2.rotate(fl_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # fr_img = cv2.rotate(fr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # image_dict = {"frontleft_fisheye_image": (None, fl_img), "frontright_fisheye_image": (None,fr_img)}

        # Change confidence value of model

        request_manager = RequestManager(image_dict, model)
        request_manager.get_handle_and_hinge()

        # Save side by side image
        cv2.imwrite(
            f"/home/aaron/spot_tools/data/images/side_by_side_{idx}.jpg",
            request_manager.side_by_side,
        )
        # assert request_manager.attributes_set(), 'Failed to get user input for handle and hinge.'


if __name__ == "__main__":
    test_detect("/home/aaron/spot_tools/data/models/yolov8x-worldv2-door.pt")
