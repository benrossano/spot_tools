import numpy as np
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from robot_executor_interface.mid_level_planner import OccupancyMap
from spot_tools_ros.utils import get_tf_pose, pose_to_homo


class OccupancyGridROSUpdater:
    def __init__(
        self,
        node: Node,
        body_frame,
        odom_frame,
        occupancy_map: OccupancyMap,
        feedback,
        tf_buffer,
    ):
        self.node = node
        self.body_frame = body_frame
        self.odom_frame = odom_frame
        self.occupancy_map = occupancy_map
        self.feedback = feedback
        self.tf_buffer = tf_buffer

        # create subscriber to update self.occupancy_map
        self.occupancy_map_subscriber = node.create_subscription(
            OccupancyGrid,
            "~/occupancy_grid",
            self.occupancy_map_callback,
            10,
        )

        # Create publisher for inflated occupancy map
        latching_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.inflated_occupancy_map_publisher = node.create_publisher(
            OccupancyGrid, "~/inflated_occupancy_map", qos_profile=latching_qos
        )
        self.feedback.print(
            "INFO",
            f"Initialized occupancy grid updater {self.body_frame} to {self.odom_frame}.",
        )

    def occupancy_map_callback(self, msg):
        w, h = msg.info.width, msg.info.height
        occupancy_frame_id = msg.header.frame_id
        map_origin = msg.info.origin  # map origin is the lower right corner of the grid in <robot_name>/map frame, with z pinting up
        occ_map = np.array(msg.data, dtype=np.int8).reshape((h, w))

        try:
            robot_pose = get_tf_pose(self.tf_buffer, self.odom_frame, self.body_frame)
            odom_to_robot_map = get_tf_pose(
                self.tf_buffer, self.odom_frame, occupancy_frame_id
            )
        except tf2_ros.TransformException as e:
            # an exception here would take the whole node down; skip this grid instead
            self.feedback.print(
                "WARNING", f"Skipping occupancy update, TF not available: {e}"
            )
            return

        # convert to homogeneous transformation matrices
        robot_pose_homo = pose_to_homo(robot_pose[0], robot_pose[1])
        odom_to_robot_map_homo = pose_to_homo(
            odom_to_robot_map[0], odom_to_robot_map[1]
        )
        map_origin_homo = pose_to_homo(
            [map_origin.position.x, map_origin.position.y, map_origin.position.z],
            map_origin.orientation,
        )

        # transform map origin to map frame
        map_origin_odom_frame = odom_to_robot_map_homo @ map_origin_homo
        # set occupancy grid and robot pose in the mid-level planner
        self.occupancy_map.set_grid(
            occ_map,
            msg.info.resolution,
            map_origin_odom_frame,
            robot_pose_homo,
            self.node.get_clock().now(),
        )

        # Publish the inflated occupancy map
        self.publish_inflated_occupancy_map(msg)

        self.feedback.print(
            "DEBUG",
            f"Received occupancy grid: size=({msg.info.width}, {msg.info.height}), resolution={msg.info.resolution}, origin=({msg.info.origin.position.x}, {msg.info.origin.position.y}, {msg.info.origin.position.z})",
        )

    def publish_inflated_occupancy_map(self, original_msg):
        """
        Publishes the inflated occupancy map stored in the mid-level planner.

        Input:
            - original_msg: OccupancyGrid message with the original grid metadata
        """

        # Create new OccupancyGrid message for inflated grid
        inflated_msg = OccupancyGrid()
        inflated_msg.header = original_msg.header
        inflated_msg.header.stamp = self.node.get_clock().now().to_msg()
        inflated_msg.info = original_msg.info

        # Get inflated grid from mid-level planner and populate message
        inflated_grid = self.occupancy_map.clone_grid()
        inflated_msg.data = inflated_grid.flatten().astype(np.int8).tolist()

        # Publish the inflated occupancy map
        self.inflated_occupancy_map_publisher.publish(inflated_msg)
