# Handoff — mid-level planner characterization

Written 2026-09-11. Goal of the session: get this stack running locally in order to do
real-time replanning to goals from live local occupancy, on Spot "Smaug"
(`spot-BD-32330006`, software 5.0.0).

Nothing here has touched the real robot yet. Everything below is ROS-free and robot-free.

## Environment

`spot_tools_env/` at the repo root — a **uv** venv on Python 3.12 with
`--system-site-packages`, so it also sees ROS Jazzy's `rclpy` for the ROS nodes later.

```bash
uv venv --python 3.12 --system-site-packages spot_tools_env
uv pip install --python spot_tools_env/bin/python \
    -e ./robot_executor_interface/robot_executor_interface -e ./spot_tools
uv pip install --python spot_tools_env/bin/python ultralytics
```

Two gotchas worth keeping:

- The README's `python3 -m venv` recipe **fails on this machine**: `python3-venv` is not
  installed and `ensurepip` is unavailable. Use `uv` (already on PATH) or
  `apt install python3.12-venv`.
- `ultralytics` is a **hard import** in `spot_skills/detection_utils.py` (and
  `door_utils.py`), but it is missing from `spot_tools/setup.py`'s `install_requires`.
  Nothing in `spot_executor` can be imported without it, navigation included, because
  `spot_executor/__init__.py` pulls in `spot_executor.py` -> `grasp_utils` ->
  `detection_utils`. It drags in torch (~3 GB). Either add it to `install_requires` or
  make those imports lazy so the navigation path does not need torch.

Detector weights are already at `open_set_sim/weights/yoloe-26m-seg.pt`. `ADT4_WS` is unset
on this machine, so `spot_executor_node.yaml`'s `detector_model_path` will not resolve as-is.

## What now works

`spot_tools/examples/test_spot_executor.py` runs end to end (`Spot reached end of path`,
`Finished 'follow' command with return True`, exit 0). It had drifted from the code in four
places, all fixed:

1. `SpotExecutor(spot, tf_lookup)` — signature gained `detector` and `planner`; both may be
   `None` for a plain `Follow`.
2. `Gaze(...)` — gained a required `object_id`.
3. The `tf_lookup` stub returned a numpy array for rotation, but `transform_command_frame`
   reads `.x/.y/.z/.w`. The real `spot_tools_ros.utils.get_tf_pose` returns
   `(np.array([x,y,z]), geometry_msgs/Quaternion)`.
4. The path was Nx2. `transform_command_frame` does `command[ix, 2] += yaw`, so waypoints
   must be Nx3 `(x, y, heading)` despite the `Follow` field being named `path2d`.

### One library change — review this one

`spot_skills/navigation_utils.py`: the guard `if mid_level_planner is not None:` in
`follow_trajectory_continuous` had been **commented out**, so it called
`mid_level_planner.plan_path()` unconditionally and raised `AttributeError` on `None` — even
though the failed-plan fallback immediately below ("following high-level path directly")
already handles that case. I restored the guard so a `None` planner falls into that fallback.
The signature allows `None` and `SpotExecutor` passes its planner straight through, so this
looks like the original intent, but it is a judgement call about intent — worth your eyes.

## The finding: the planner livelocks when sensing < lookahead

New harness: `spot_tools/examples/test_mid_level_planner.py`. It drives `MidLevelPlanner` in
a closed replan loop against a synthetic grid, with the grid cropped to a sensing radius
(mirroring `fake_occupancy_publisher.crop_around_robot`), so the robot discovers a wall as it
approaches. No ROS, no robot — a failure here is a planner problem and nothing else.

```
./spot_tools_env/bin/python spot_tools/examples/test_mid_level_planner.py --sweep

crop (m)    outcome  replans   length  max |y|
      -1    reached       26    17.4m    4.68m
     8.0    reached       26    17.4m    4.68m
     5.0    reached       29    19.4m    4.68m
     4.0   livelock      120    72.8m    1.20m
     3.0   livelock      120    86.1m    1.08m
     2.0   livelock      120    77.9m    0.48m
```

**The cliff is between 4 m and 5 m of sensing**, with `lookahead_distance_grid: 50` at
0.12 m/cell = 6 m of lookahead. It breaks about where sensing drops below the lookahead.

### Mechanism, confirmed from both sides

A* **never fails**. Instrumenting the 3 m run shows an exact 2-cycle from replan 9 onward:

```
it   robot xy         target xy        waypoints
 9 [-1.56, -1.08]   [ 1.32, -0.24]    44
10 [-1.56, -0.36]   [ 1.44, -0.36]    51
11 [-1.56, -1.08]   [ 1.32, -0.24]    44   <- repeats forever
```

`a_star`'s `is_obstacle` is `occupancy_map[cell] > 0`, so **UNKNOWN (-1) is planned through
as free**. A* optimistically routes straight through the unseen part of the wall and only has
to dodge the ±3 m of wall actually visible. Detouring north or south around that visible stub
costs nearly the same, so each replan flips, and the robot oscillates between two poses.

Marking unknown as occupied instead makes A* **fail immediately** (within 5-9 replans, every
radius), because the lookahead target itself lands in unknown space and `is_obstacle(goal)`
rejects it. So the planner *structurally depends* on optimistic unknown-as-free to have a
goal at all — it cannot simply be made conservative.

What is missing is **commitment**: nothing carries the previous plan forward between replans,
and nothing detects lack of progress. `navigation_utils.py` already admits the second half —
`# TODO: we should probably have a finer-grained check about making progress`.

### Dead end, so you don't repeat it

`project_goal_observed_with_frontier` has **zero call sites** —
`project_goal_to_grid` calls the naive `project_goal_observed`. Routing through the
frontier-aware version changes **nothing**: identical trajectories at every radius. The
livelock is not about where the goal gets projected.

## Next steps, in order

1. **Prototype hysteresis in `plan_path`** — keep the previous target/path unless the new one
   is meaningfully better, so the 2-cycle cannot form. Test against
   `test_mid_level_planner.py --sweep`; success is 4 m and 3 m turning from `livelock` into
   `reached`. This is the cheapest high-value change and is the difference between this
   working and not working on the robot.
2. **Add a progress check** to `follow_trajectory_continuous` (the existing TODO): if
   distance-to-goal has not improved over N seconds, bail rather than burn the whole
   `path_distance * 6` timeout. On the robot the current behavior looks like Spot nosing at a
   wall until timeout.
3. **Decide the unknown-space policy deliberately.** Optimistic (current) livelocks;
   conservative cannot plan at all. Options: a distinct traversal *cost* for unknown rather
   than free/blocked, or clamping the lookahead target to observed space.
4. **Then the colcon build and the ROS flow.** Untouched so far. Per the README:
   `tmuxp load dcist_launch_system/tmux/autogenerated/spot_prior_dsg-spot_prior_dsg.yaml`,
   then `ros2 run spot_tools_ros fake_occupancy_publisher` and
   `ros2 run spot_tools_ros fake_path_publisher -6 1`, watching
   `/hamilton/spot_executor_node/mlp_path_publisher` in RViz. Note the repo README says
   `spot_ros2` supports Ubuntu 22.04 / Humble only; this machine is **Jazzy**, so expect
   trouble there (nothing in `spot_tools` itself imports `spot_ros2` — it uses `bosdyn`
   directly — so this may not bite).
5. **Real robot.** Needs `ADT4_BOSDYN_IP` / `ADT4_BOSDYN_USERNAME` / `ADT4_BOSDYN_PASSWORD`
   and `ADT4_WS`. Robot namespace is `hamilton`. Do not skip step 2 first: a local-minimum
   trap on hardware means Spot walking into a wall repeatedly.

## Open questions

- How much reliable occupancy radius does Hydra's TSDF actually give the planner in real
  conditions? That number decides whether the livelock is academic or the main problem.
- Is `follower_lookahead: 2.1` (metres, the follower) vs `lookahead_distance: 50` (grid
  cells, the planner) intentional, or a units mismatch worth checking?
- `LeaseManager` invalidates the current plan when another client takes the lease. Worth
  understanding before running this alongside the tablet or the `spot-sdk-scripts` tools.
