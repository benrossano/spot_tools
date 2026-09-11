from types import SimpleNamespace

import numpy as np
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze

import spot_executor as se


def tf_lookup(parent, child):
    """Stand in for spot_tools_ros.utils.get_tf_pose, which returns
    (np.array([x, y, z]), geometry_msgs/Quaternion). transform_command_frame reads the
    rotation as .x/.y/.z/.w, so an array will not do."""
    print(f"Faking lookup for {parent}->{child}")
    identity = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    return np.array([0.0, 0.0, 0.0]), identity


spot = se.FakeSpot(init_pose=np.array([0, 1, 0, 0]))
# SpotExecutor gained `detector` and `planner` arguments; both are optional for a plain
# Follow, which falls through to follow_trajectory_continuous with no mid-level planner.
executor = se.SpotExecutor(spot, None, tf_lookup, None)

collector = se.FeedbackCollector()


# Nx3: transform_command_frame rotates columns 0 and 1 and adds the frame yaw to column 2,
# so each waypoint is (x, y, heading) even though the Follow field is named path2d.
path = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
    ]
)

follow_cmd = Follow("map", path)

gaze_cmd = Gaze("map", np.array([5, 5, 0]), np.array([7, 7, 0]), "obj0", stow_after=True)

seq = ActionSequence("id0", "spot", [follow_cmd, gaze_cmd])

executor.process_action_sequence(seq, collector)
