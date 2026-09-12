# Handoff — Spot navigation stack: test pipeline and on-robot readiness

Written 2026-09-11, updated 2026-09-12 at the end of the session. **Start here.**

## Where things stand (one screen)

**Works, verified today, nothing on the robot yet:**
- Off-robot test pipeline for the mid-level planner + executor on ROS 2 Jazzy: 51 ROS-free tests
  (11 documented xfails), a ROS closed-loop harness with the fake robot (2 tests, pass), a 5-min
  replay of the real Spot bag through YOLOE + Hydra, and an offline planner evaluation on that
  occupancy stream (168/168 plans, 0 collisions, 0 A* failures).
- **End-to-end task pipeline on your real code path:** manifest task skeleton -> `open_set_navigation`
  grounding (SigLIP2 + SAM3 over the Building 45 tour archive) -> OmniPlanner/Fast Downward ->
  compiled `Follow/Pick/Place` -> `SpotExecutor` on the fake robot, pick grounded in the robot camera
  through the same perception service. Cup -> table task: all 6 actions succeeded (47.7 s), pick
  pixel (290,115) score 0.75, 3D estimate 1 cm from the planned point.
  Run dir: `open_set_sim/output/spot_stack_runs/cup_to_table_20260911_222117`.
- Bugs fixed today (all uncommitted): four executor/node crash paths, two logger bugs, doubled
  camera namespace in bag replay, Nx2 Follow paths, weights filename, pick class `UNKNOWN`,
  lookahead target stalling in unknown space (now `allow_unknown_target`).

**Nothing is committed.** Three repos carry the work:

| repo | files |
|---|---|
| `dcist_ws/src/awesome_dcist_t4/spot_tools` (this one, `main` @ c9a76e7) | `HANDOFF.md`, `pytest.ini`, `tests/` (new), `mid_level_planner.py`, `navigation_utils.py`, `spot_executor.py`, `fake_spot.py`, `grasp_utils.py`, `detection_utils.py`, `door_utils.py`, `spot_executor_ros.py`, `occupancy_grid_ros_updater.py`, `fake_occupancy_publisher.py`, `examples/test_mid_level_planner.py` |
| `open_set_sim/open_set_navigation` (has many other uncommitted edits of yours) | `experiments/manifest.py`, `experiments/runner.py`, `planning/omniplanner.py`, `planning/spot_stack.py` (new), `tests/test_spot_stack.py` (new) |
| `open_set_sim` (also has other uncommitted edits of yours) | `manifests/spot_stack_bldg45_cup_to_table.yaml` (new), 4x `spot_executor_node.yaml` (26m -> 26l weights) |

`~/research/spot_tools` (spot-sdk-scripts) was only fast-forwarded to 9c2a97c.

## Resume in a new session

```bash
# environment (every ROS/test command below assumes this)
cd ~/research/openset_nav_dir/dcist_ws
source /opt/ros/jazzy/setup.bash && source ../open_set_sim/install/setup.bash && source install/setup.bash
cd src/awesome_dcist_t4/spot_tools
PY=~/research/openset_nav_dir/.venv/bin/python          # the venv every ROS python node runs in

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg $PY -m pytest -q -m "not ros"   # 25 s
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg $PY -m pytest -q                # + ROS closed loop, ~3 min
$PY tests/ros/fake_nav_harness.py --crop 8.0                                    # watch the fake robot
$PY tests/ros/replay_planner_on_bag.py ~/research/openset_nav_dir/adt4_output/spot_bag_occupancy_20260911_192834/derived_bag

# task pipeline (GPU): perception service first, then the manifest
cd ~/research/openset_nav_dir && source activate-open-set-navigation.sh
open-set-perception-service --device cuda --port 8078 --top-k 32 --sam3-source "$OPEN_SET_SAM3_SOURCE" &
cd open_set_sim && open-set-run manifests/spot_stack_bldg45_cup_to_table.yaml --output output/spot_stack_runs/<name>
cd open_set_navigation && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../.venv/bin/python -m pytest tests/test_spot_stack.py
```

If the colcon overlay is missing (fresh `dcist_ws/install`): see "Environment" below for the
six-package `colcon build` line (10 s).

## What is left, in order

1. ~~Decide and commit.~~ Done 2026-09-12: the frame drop is fixed (see "Bugs still open"), and the
   robot configs (`open_set_sim/dcist_launch_system/{config/default,config_generation/base_params}/spot_executor_node.yaml`
   and the in-repo `example_parms.yaml`) now set `path_commitment_weight: 1.0`,
   `path_commitment_band: 0.5`, `allow_unknown_target: true`, `follow_progress_timeout: 20.0`.
   Code defaults stay off. Still yours: place semantics (`object_point` is ignored).
2. **`robot: ros` dispatch is untested.** `SpotStackExecutor.dispatch_ros` publishes the compiled
   `ActionSequenceMsg` to `/<robot>/omniplanner_node/compiled_plan_out`; run it against the fake
   ROS stack (`tests/ros/fake_nav_harness.py` starts everything except the path publisher) before
   trusting it. Also decide whether the on-robot path is this (open_set_navigation drives the ROS
   executor) or `omniplanner_node` as in the field stack.
3. **Robot-side pick frame for the perception service.** The service needs a keyframe layout
   (rgb + depth + `world_T_body` meta + `camera_calib.json`); `spot_sensors` publishes RGB and
   depth-in-visual-frame, so write a small "current frame -> keyframe dir" adapter and point
   `PerceptionServicePickDetector._frame_source` at it. Alternative: `pick_detector: sam3_http`
   with `sam3_server.py` if GPU memory allows (it did not next to the service on 12 GB).
4. **Kinematic fake + full bag.** The e2e run used the teleporting fake; rerun with
   `spot_stack.motion: kinematic` to see timing/replan behaviour. Replay the whole 25-min bag
   through Hydra at `--rate 0.3` (`tests/ros/record_bag_occupancy.sh --rate 0.3`) for a complete
   occupancy stream; today's derived bag covers ~92 s of bag time.
5. **On-robot order** (section at the end): E-stop rehearsal, `move.py goto`, executor with
   `identity` planner on a short Follow, then `astar`, then Pick/Place — with the frame-drop fix.
6. Known open bugs are listed under "Bugs still open"; each has an `xfail(strict=True)` test that
   flips when fixed.

---

## Environment — what actually works on this machine

- **Python:** `~/research/openset_nav_dir/.venv` (py3.12, uv). This is the venv the launch
  system's `pyenv_node` (ianvs) runs every ROS python node in (`ADT4_ENV/spark_env -> ../.venv`).
  Installed today: `bosdyn-{client,api,core}==5.2.0`, `scikit-image`, `onnxruntime`,
  `transforms3d`, `tf_transformations` (from git; not on PyPI, apt needs sudo), `pygame`,
  `PyOpenGL`, `lark`, and `-e` installs of `spot_tools`, `robot_executor_interface(_ros)`,
  `spot_tools_ros`. The `spot_tools_env/` from the morning handoff never existed here.
- **ROS overlay:** `open_set_sim/install` (Hydra etc., already built for Jazzy) plus
  `dcist_ws/install`, built today with
  `colcon build --symlink-install --packages-select robot_executor_msgs heracles_ros_interfaces
  robot_executor_interface robot_executor_interface_ros spot_tools spot_tools_ros`
  from `dcist_ws` (10 s). Source order: `/opt/ros/jazzy`, `open_set_sim/install`, `dcist_ws/install`.
- **numpy 2 vs Jazzy:** the venv has numpy 2.5; Jazzy's compiled `cv_bridge` needs numpy 1.x.
  The workspace's existing workaround is `ROS_COMPAT_PYTHONPATH` (`~/.local/lib/python3.12/site-packages`
  = numpy 1.26) prepended for ROS python nodes (see `open_set_sim/scripts/run_behavior1k_bag.sh`).
  `spot_executor_ros.py` now imports `cv_bridge` optionally, so the executor runs either way.
- **Missing packages:** `nlu_interface_rviz` (pick-approval UI) and `ros_system_monitor_msgs`
  (heartbeat) exist nowhere on this machine. Both imports are now optional; without them the
  executor auto-approves the detector's pick candidate and skips the heartbeat.
- **Weights:** only `yoloe-26l-seg.pt` exists (`open_set_sim/weights/`). All four
  `spot_executor_node.yaml` copies in `open_set_sim/dcist_launch_system` were switched from
  `26m` to `26l` (same change `docs/setup.md` documents for `instance_seg.yaml`).
- **Env vars for the launch system:** `ADT4_WS=~/research/openset_nav_dir/open_set_sim`,
  `ADT4_ENV=$ADT4_WS/.adt4_env`, `ADT4_ROBOT_NAME=hamilton`, `ADT4_OUTPUT_DIR`. Real
  `ADT4_BOSDYN_IP/USERNAME/PASSWORD` are not in `secrets.env`; they are in the SDK repo's
  git-ignored `env.sh` (`SPOT_IP`, `BOSDYN_CLIENT_*`). Dummies work for fake runs.

## How to run the tests

```bash
cd ~/research/openset_nav_dir/dcist_ws
source /opt/ros/jazzy/setup.bash
source ../open_set_sim/install/setup.bash
source install/setup.bash
cd src/awesome_dcist_t4/spot_tools
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg ../../../../.venv/bin/python -m pytest -q           # everything (~3 min)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg ../../../../.venv/bin/python -m pytest -q -m "not ros"  # ROS-free (~25 s)
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required: Jazzy's `launch_testing` pytest plugins on
`PYTHONPATH` are incompatible with pytest 9. `xfail(strict=True)` marks are used to pin known
defects; when one is fixed the test flips to XPASS and fails until the mark is removed.

| File | What it covers |
|---|---|
| `tests/test_occupancy_map.py` | grid<->metric with rotated map origins, inflation, cost map (incl. UNKNOWN cost) |
| `tests/test_a_star.py` | A* on gaps/walls/unknown, 4- vs 8-connectivity, cost-map bowing |
| `tests/test_plan_path.py` | lookahead target, projection into free space/frontier, off-grid robot, no-grid fallback, path commitment |
| `tests/test_transform_command_frame.py` | SE2 path transform vs scipy, Nx3 requirement |
| `tests/test_follow_trajectory.py` | follower with/without planner on FakeSpot, detour, timeout, lease invalidation, progress timeout |
| `tests/test_replan_livelock.py` | closed-loop replanning sweep from the morning harness, with and without commitment |
| `tests/test_action_msgs.py` | `to_msg`/`from_msg` round trips (needs the built `robot_executor_msgs`) |
| `tests/ros/test_fake_nav.py` | spawns static TFs + fake occupancy + executor + path publisher; fake Spot must reach the goal through an L hallway |
| `tests/ros/fake_nav_harness.py` | the above as a CLI: `--crop`, `--teleport`, `-p name:=value`, `--domain-id` |
| `tests/ros/record_bag_occupancy.sh` | replays the real bag through YOLOE + Hydra, records `/hamilton/hydra/tsdf/occupancy` + TF |
| `tests/ros/replay_planner_on_bag.py` | offline planner evaluation on that derived bag |

## Code changes (all uncommitted)

Fixes:
- `spot_executor_ros.py`: optional imports (above); `self.get_logger.warn` -> `self.get_logger().warn`
  and re-raise in the TF lookup (it used to raise `AttributeError` and then return `None` into
  `transform_command_frame`); detector only built when `detector_model_path` is non-empty
  (navigation-only nodes no longer need torch + weights); new params `follow_progress_timeout`
  (s, <=0 off), `path_commitment_weight` (0 off), `path_commitment_band` (m), `fake_spot_kinematic`.
- `occupancy_grid_ros_updater.py`: a missing TF at the first grid used to raise inside the
  subscription callback and take the node down; it now skips that grid with a warning. On the
  robot Hydra's grid is transient-local, so it arrives before TF is guaranteed.
- `mid_level_planner.py`: `plan_path` before any grid used to crash the follow thread with
  `TypeError: 'NoneType'`; it now returns `(False, high-level path)` so the follower falls back.
- `fake_occupancy_publisher.py`: same logger bug; with `--crop_distance > 0` it now waits for the
  robot TF instead of dying at startup (it starts before the executor).
- `detection_utils.py`, `door_utils.py`: `ultralytics` imported lazily.

Opt-in additions (defaults keep the old behaviour):
- `MidLevelPlanner(path_commitment_weight, path_commitment_band_m)`: extra A* step cost
  `weight * clip(d_prev / band, 0, 1)` for cells `d_prev` from the previous local path
  (stored metric, re-rasterised per grid, reset when the high-level path changes). Hysteresis
  against replan flip-flop.
- `follow_trajectory_continuous(progress_timeout, progress_epsilon)`: abort when the distance to
  the goal has not improved by `epsilon` for `progress_timeout` s (the TODO in the loop).
- `FakeSpot(kinematic=True)`: SE2 goals are tracked at 0.75 m/s / 0.65 rad/s in `step()` instead
  of teleporting. `FakeSpotRos` already calls `step()` at 20 Hz.
- `spot_tools/examples/test_mid_level_planner.py`: `--commit W`.

## Planner livelock — status

Sweep (`test_mid_level_planner.py --sweep --commit W`; 0.12 m cells, lookahead 50 cells = 6 m):

```
crop (m)   w=0        w=0.5      w=1.0            w=2.0
  -1       reached    reached    reached          reached
  8.0      reached    reached    reached          reached
  5.0      reached    reached    reached (25.6m)  reached (27.2m)
  4.0      livelock   reached    reached          reached
  3.0      livelock   livelock   reached          reached
  2.0      livelock   livelock   livelock         livelock
```

`w=1.0` meets the morning handoff's success criterion (4 m and 3 m). Cost: at 5 m sensing the
path is 25.6 m instead of 19.4 m — commitment sticks to the first detour it picked. 2 m sensing
(one third of the lookahead) still 2-cycles; out of the robot's regime (Hydra camera range 10 m,
map window 14 m). Default stays 0; `path_commitment_weight: 1.0` in `spot_executor_node.yaml`
is the recommended on-robot setting once you have seen it on the derived bag.

Also found: with `use_cost_map: true` (the robot config) UNKNOWN cells already carry the full
`nearest_obstacle_cost` per step, because `distance_transform_edt(free_mask)` is 0 there. So the
robot config is not purely "unknown is free"; it is "unknown costs 6x". Pinned by
`test_cost_map_treats_unknown_like_an_obstacle`.

## ROS closed loop (fake robot) — results

`tests/ros/fake_nav_harness.py`: L-shaped hallway (`fake_occupancy_publisher --scenario L_shape_hallway`),
non-identity `hamilton/map -> hamilton/odom` (1, -2, 0.5 rad), robot starts in the vertical arm,
goal in the horizontal arm; the high-level path is the straight line through the wall.

```
config                      reached  executor "follow True"  time    mid-level plans  free-space violations
crop -1 (all visible)        yes      yes                     26.6 s  187              0
crop 8 m                     yes      yes                     26.6 s  192              0
crop 4 m                     yes      yes                     26.8 s  215              0
crop 4 m + commitment 1.0    yes      yes                     26.4 s  218              0
crop 8 m, teleporting fake   yes      yes                      6.3 s   12              0
```

The L hallway has no symmetric detour, so 4 m sensing does not livelock here; the synthetic
wall in `test_replan_livelock.py` is the case that does.

## Real bag through Hydra

`tests/ros/record_bag_occupancy.sh --duration 300 --rate 1.0` replayed the first 5 min of
`~/spot_bags/bulding_1_infinite` (real Spot, building 1) through YOLOE instance segmentation +
Hydra (`conf_name:=default`, `labelspace_name:=instance_seg`) and recorded the planner's inputs
to `adt4_output/spot_bag_occupancy_20260911_192834/derived_bag` (49 MB, 170 grids, TF, odom).
Hydra published `/hamilton/hydra/tsdf/occupancy` at ~0.85 Hz in `hamilton/map`, grids ~240x128
cells at 0.10 m (the active window), and an identity `hamilton/map -> hamilton/odom`.

**Throughput caveat:** the 170 grids span only ~92 s of bag time; Hydra + YOLOE ran at roughly
0.3x real time on this machine (GPU 13 %, CPU-bound). Replay at `--rate 0.3` or lower for full
coverage — the Isaac replay script uses 0.1.

`tests/ros/replay_planner_on_bag.py derived_bag` (plan from every recorded pose toward the pose
the robot reached 8 m later; lookahead 50 cells, inflation 0.2 m, cost map on):

```
plans                168 ok, 0 A* failures, 2 skipped (TF not yet available)
plan time            mean 15.7 ms, max 131 ms
waypoints in OCCUPIED raw cells   0 / 168 plans
path length          mean 3.78 m (163/168 shorter than the 5 m lookahead)
lookahead target in known space   163 / 168
known fraction of cells within    2 m: 0.84 mean (min 0.52)
                                  5 m: 0.43 mean (10th pct 0.35)
                                  8 m: 0.29 mean (min 0.18)
```

Reading: the planner is correct on real geometry (no collisions, no failures, fast), but the
robot only knows ~45 % of the cells within its 5 m lookahead, so the target is pulled back to the
nearest observed free cell almost every replan (`project_goal_observed` treats UNKNOWN as
not-free). That is the regime the synthetic sweep calls sensing < lookahead — it is the normal
operating regime on the robot, not a corner case, which is why the closed-loop commitment /
progress-timeout options matter. Commitment cannot be evaluated offline (no feedback loop):
`--commit 1.0` gives identical numbers here. The four longest plans (25 m, 31 % unknown, t≈91 s)
came from the last grids before Hydra fell behind; worth a look in RViz.

Reproduce: `ros2 bag play derived_bag --clock` on an isolated `ROS_DOMAIN_ID` plus the executor
with `use_fake_spot_interface: true`, `fake_spot_external_pose: true` and
`~/occupancy_grid` remapped to `/hamilton/hydra/tsdf/occupancy` gives a live view of the planner
on the recording; the offline script is the deterministic version of that.


## Task skeleton -> grounding -> OmniPlanner -> Spot executor (`spot_stack`)

Added the same evening, in `open_set_navigation` (checkout `open_set_sim/open_set_navigation`, also
reachable through the `dcist_ws/.../open_set_navigation` symlink):

- `experiments/manifest.py`: `execution.backend: spot_stack` + `execution.spot_stack: SpotStackConfig`
  (`robot: fake|ros`, `motion: teleport|kinematic`, `start_position`, `pick_detector:
  perception_service|sam3_local|sam3_http|none`, `pick_image_from_archive`, `occupancy_from_places`,
  `path_commitment_weight`, `follow_progress_timeout_s`, `allow_unknown_target`, ...).
- `planning/omniplanner.py`: `OfflineOmniPlanner` keeps `last_pddl_plan` / `last_scene` after `solve`.
- `experiments/runner.py`: `_execute_spot_stack` dispatch, `execution.started/completed/failed` events.
- `planning/spot_stack.py`:
  - `compile_action_sequence` — the PDDL plan goes through OmniPlanner's own
    `dsg_pddl_planning_compile.compile_pddl_plan_pure` (what `omniplanner_node` publishes to the
    robot), then Follow paths are normalised to (x, y, heading) and Pick/Place get the skeleton's
    class name as prompt (the DSG labelspace says `UNKNOWN`).
  - `dsg_occupancy_grid` — prior map from the DSG mesh-places layer (free discs + corridors),
    everything else UNKNOWN, so the mid-level planner plans through unknown space.
  - `nearest_archive_frame` — the fake robot's camera shows the archive keyframe recorded closest
    to (and facing) the object it is about to pick.
  - `PerceptionServicePickDetector` — the pick-time grounding is the same SigLIP2+SAM3 service
    that grounded the task: the camera frame is exposed as a one-keyframe frame source and
    searched with `/v1/search`; the returned mask's centroid is the pixel handed to
    `object_grasp` -> `PickObjectInImage`. `Sam3PickDetector` (in-process SAM3) exists too but a
    second SAM3 next to the service does not fit in 12 GB (CUDA OOM, first run).
  - `SpotStackExecutor` — `robot: fake` runs `SpotExecutor.process_action_sequence` in-process on
    `FakeSpot`; `robot: ros` publishes the `ActionSequenceMsg` to
    `/<robot>/omniplanner_node/compiled_plan_out` for the ROS executor node (untested on ROS).
- `tests/test_spot_stack.py` (6 hermetic tests, no GPU).
- Manifest `open_set_sim/manifests/spot_stack_bldg45_cup_to_table.yaml`: real Building 45 tour
  archive, fixed skeleton `visit cup / pick cup / visit table / place cup on table`, archive-only
  grounding, OmniPlanner, fake Spot. Run with the perception service up:

```bash
cd ~/research/openset_nav_dir && source activate-open-set-navigation.sh
open-set-perception-service --device cuda --port 8078 --top-k 32 --sam3-source "$OPEN_SET_SAM3_SOURCE" &
cd open_set_sim && open-set-run manifests/spot_stack_bldg45_cup_to_table.yaml --output output/spot_stack_runs/<name>
```

Spot-side changes for this: `transform_command_frame` accepts Nx2 paths (compiled Follow paths are
Nx2; only the ROS message round trip used to make them Nx3), `FakeSpot(images=..., fake_semantic_class=...)`
so the fake camera can show a chosen frame and `object_grasp` no longer forces class "bag",
`MidLevelPlanner(allow_unknown_target=...)` + ROS param `allow_unknown_target` (default off).

**Result (2026-09-11, run `output/spot_stack_runs/cup_to_table_20260911_222117`)** — task
"Pick up a cup and put it on a table", Building 45 archive, no model calls, 2 grounding searches:

```
grounding   cup -> o15 (10.52, 16.46), table -> o16 / place t13987      (SigLIP2 + SAM3 on 1879 keyframes)
plan        goto-poi(pstart,t10350) goto-poi(t10350,o15) pick-object(o15) goto-poi(o15,o16) goto-poi(o16,t13987) place-object(o15,t13987)
            38 path points, 111 m, fast-downward 0.5 s
execution   Follow 57 m ok | Follow 2 m ok | Pick cup ok | Follow 51 m ok | Follow 1.6 m ok | Place ok   (47.7 s, teleporting fake)
pick        camera = archive keyframe agent_1788139667149585402 (nearest, facing the cup)
            perception service: "cup" score 0.75, 228 mask px, pixel (290, 115), 3D (10.53, 16.47, 1.14) -- 1 cm from the planned pick point
holding     set_robot_holding_state(True, O15) then (False, O15)
```

Two failures on the way there, both instructive for the robot:
1. In-process SAM3 next to the perception service: CUDA OOM (12 GB GPU). Fixed by grounding the
   pick through the service instead (`pick_detector: perception_service`).
2. Pick class arrived as `UNKNOWN` (DSG labelspace) and the second Follow stalled 1.4 m short of
   its goal: the goal lay in UNKNOWN cells of the prior map and `project_goal_observed` pulled the
   lookahead target back into observed free space every replan (progress timeout fired, 3 attempts,
   executor moved on and *placed without having picked*). Fixed by prompting with the skeleton's
   class and `allow_unknown_target: true`. On the robot the same stall happens whenever a goal is
   beyond what Hydra has observed — `allow_unknown_target` is the switch to flip there too.

Not done: `robot: ros` dispatch is written but untested against the ROS executor node; on the real
robot the pick-time frame needs depth + pose + `camera_calib.json` in the keyframe layout for the
perception service (`spot_sensors` publishes depth-in-visual-frame, so that is plumbing, not new
perception); kinematic fake motion was not used for this run (teleport).


## Bugs still open (each pinned by an xfail test unless noted)

- ~~Frames dropped on the wire for Gaze/Pick/Place~~ **fixed 2026-09-12**: `to_msg` now fills
  `gaze_frame`/`pick_frame`/`place_frame` (+ Place `object_class`), and `execute_gaze` transforms
  the gaze point from `command.frame` into Spot's vision frame (`SpotExecutor.point_in_vision_frame`,
  full SE(3) via `transform_point_frame`); a message with an empty frame is still taken as
  vision-frame with a WARNING. Pick/Place do not use their points yet (`object_grasp` detects in
  the image, `object_place` drops in place), so nothing else needed transforming. Still true: the
  heading column of a Follow path is replaced by the segment direction on the wire (the follower
  recomputes heading anyway).
- `follow_trajectory_continuous(feedback=None)` crashes (signature allows None).
- `FakeSpot.get_pose()` is `(x, y, z, yaw)`; `Spot.get_pose()` is `(x, y, yaw)`. The follower's
  end-of-path heading uses index 2, so in fake mode that is z. `FakeStateClient` also always
  reports `is_gripper_holding_item=True` (arm-freeze branch always taken).
- `grid_cell_to_global_pose -> global_position_to_grid_cell` is off by one for ~10 % of columns
  (`int()` truncation on corners). Only matters for exact corner points.
- A* uses a Manhattan heuristic with unit-cost diagonals (inadmissible; fine on open grids).
- Not tested, by inspection: `object_place` backs up 1 m then sidesteps 1 m blind, and failed
  grasps random-walk (`execute_recovery_action`) — no occupancy check. `Place.object_point` is
  ignored. `LeaseManager` re-takes the lease and stands the robot whenever nobody owns it, and
  treats any owner not named `understanding*` as plan-invalid; SDK scripts (`move.py`, `pick.py`)
  take the lease under other names. A failed action is retried twice, then the sequence moves on
  to the next action (a failed Follow -> Pick at the wrong place).
- Units: `lookahead_distance: 50` cells is 6 m on the 0.12 m fake grid but 5 m on Hydra's 0.1 m
  voxels. `follower_lookahead: 2.1` m is in `open_set_sim/dcist_launch_system/config/default/spot_executor_node.yaml`
  (the in-repo `example_parms.yaml` says 2).
- `master.launch.yaml`: the instance-segmentation node sits in the `<robot>_zed` namespace, so
  the relative default `camera_rgb_topic` resolves to `/hamilton/hamilton_zed/hamilton_zed/...`.
  Always pass absolute camera topics (the bag script does).

## On-robot order (nothing below was run)

1. E-stop rehearsal per `~/research/spot_tools/NEXT_STEPS.md`, then `move.py goto 1 0` — same
   `synchro_se2_trajectory_point_command` the executor's `navigate_to_absolute_pose` uses.
2. `spot_sensors` + executor with `mid_level_planner_type: identity`, one short Follow in
   `hamilton/odom` (`fake_path_publisher --map_frame hamilton/odom 2 0`), no Hydra.
3. Hydra live, `astar`, `follow_progress_timeout: 20`, `path_commitment_weight: 1.0`; watch
   `/hamilton/spot_executor_node/mlp_path_publisher` and `.../inflated_occupancy_map` in RViz.
4. Pick/Place: fix the frame drop above first, or keep the robot in a session where
   `map == odom`. `object_grasp` and the SDK's `grasp_at_pixel` use the same `PickObjectInImage`
   call; the SDK version (proven on Smaug) has a 30 s timeout vs 15 s here.
