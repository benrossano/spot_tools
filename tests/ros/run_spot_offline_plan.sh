#!/usr/bin/env bash
# Plan and drive on a RECORDED Hydra map, without Hydra running.
#
# Brings up the offline planning stack against a recorded run directory (ADT4_PRIOR_MAP):
#   [1] spot_sensor_node + robot state + calibration TFs  (real)   |  nothing: FakeSpotRos publishes odom->base_link (fake)
#   [2] static_occupancy_publisher   <map>/occupancy_static.npz  ->  /<robot>/hydra/tsdf/occupancy
#   [3] fiducial_localizer_node      <map>/fiducials.yaml + live AprilTag -> <robot>/map -> <robot>/odom   (real)
#       static_transform_publisher   a hand-picked map->odom                                                (fake)
#   [4] spot_executor_node           A* in the static grid, Follow -> SE2 commands (or the in-process FakeSpot)
#   [5] rviz (unless --no-rviz)
#   [6] open-set-run <manifest>      SAM3 grounding on the recorded keyframes -> PDDL -> Follow sequence published
#                                    to /<robot>/omniplanner_node/compiled_plan_out (manifest: robot: ros, start_position: tf)
# then prints the robot's map-frame pose until Ctrl-C.
#
#   tests/ros/run_spot_offline_plan.sh [--map DIR] [--manifest FILE] [--out DIR] [--robot hamilton] [--platform smaug]
#                                      [--fake] [--fake-anchor "X Y YAW"] [--fake-start "X Y YAW"] [--fake-teleport]
#                                      [--no-rviz] [--no-dispatch] [--watch-s N] [--domain-id N] [--check]
#
# --fake runs everything except the robot: the executor drives an in-process FakeSpot whose odom frame is offset
# from the map by --fake-anchor (default the launch file's verification TF, 5 10 1.57) and which starts at the
# map-frame pose --fake-start (default the recorded tour's end, 4.42 2.43 0). That exercises every frame
# conversion the real run depends on, with nothing that can move.
#
# Needs: the recorded run (bag not required at run time; fiducials.yaml + occupancy_static.npz are), a dcist_ws
# install with spot_tools_ros + dcist_launch_system, the venv, and for the real robot ADT4_BOSDYN_IP /
# ADT4_BOSDYN_USERNAME / ADT4_BOSDYN_PASSWORD (or BOSDYN_CLIENT_* + SPOT_IP from spot-sdk-scripts env.sh).
set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
# .../open_set_sim/spot_tools/tests/ros -> open_set_sim/scripts/lib/paths.sh
source "$_SELF_DIR/../../../scripts/lib/paths.sh"

ROBOT=hamilton
PLATFORM=smaug
MAP="$ADT4_OUTPUT_ROOT/spot_tour_hydra"
MANIFEST="$OPEN_SET_SIM_ROOT/manifests/spot_tour_trash_bin_ros.yaml"
OUT=""
FAKE=0
FAKE_ANCHOR="5 10 1.57"
FAKE_START="4.42 2.43 0"
FAKE_KINEMATIC=true
RVIZ=1
DISPATCH=1
WATCH_S=0
DOMAIN_ID=88
CHECK_ONLY=0
while (($#)); do
  case "$1" in
    --map) MAP="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --robot) ROBOT="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --fake) FAKE=1; shift ;;
    --fake-anchor) FAKE=1; FAKE_ANCHOR="$2"; shift 2 ;;
    --fake-start) FAKE=1; FAKE_START="$2"; shift 2 ;;
    --fake-teleport) FAKE_KINEMATIC=false; shift ;;
    --no-rviz) RVIZ=0; shift ;;
    --no-dispatch) DISPATCH=0; shift ;;
    --watch-s) WATCH_S="$2"; shift 2 ;;
    --domain-id) DOMAIN_ID="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
MAP="$(realpath "$MAP")"
[[ -z "$OUT" ]] && OUT="$ADT4_OUTPUT_ROOT/spot_offline_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------- environment
set +u
source /opt/ros/jazzy/setup.bash
source "$DCIST_INSTALL/setup.bash"
set -u
export ADT4_BOSDYN_IP="${ADT4_BOSDYN_IP:-${SPOT_IP:-192.168.80.3}}"
export ADT4_BOSDYN_USERNAME="${ADT4_BOSDYN_USERNAME:-${BOSDYN_CLIENT_USERNAME:-}}"
export ADT4_BOSDYN_PASSWORD="${ADT4_BOSDYN_PASSWORD:-${BOSDYN_CLIENT_PASSWORD:-}}"
if ((FAKE)); then
  # FakeSpot still reads the credential parameters; give it something non-empty
  export ADT4_BOSDYN_USERNAME="${ADT4_BOSDYN_USERNAME:-fake}" ADT4_BOSDYN_PASSWORD="${ADT4_BOSDYN_PASSWORD:-fake}"
fi
export ADT4_WS ADT4_ENV
export ADT4_PLATFORM_ID="$PLATFORM"
export ADT4_ROBOT_NAME="$ROBOT"
export ADT4_OUTPUT_DIR="$OUT"
export ADT4_PRIOR_MAP="$MAP"
export SPOT_TOOLS_ROS_SHARE="${SPOT_TOOLS_ROS_SHARE:-$(ros2 pkg prefix spot_tools_ros 2>/dev/null || echo /nonexistent)/share/spot_tools_ros}"
LAUNCH_SHARE="$(ros2 pkg prefix dcist_launch_system 2>/dev/null || echo /nonexistent)/share/dcist_launch_system"
# open-set-run shells out to fast-downward, which setup_python_env.sh symlinks into the venv's bin
export PATH="$OPEN_SET_VENV_ROOT/bin:$PATH"
export YOLO_CONFIG_DIR="$OPEN_SET_SIM_ROOT/.ultralytics"
export MPLCONFIGDIR="$ADT4_ENV/matplotlib"
export ROS_LOG_DIR="$OUT/logs/ros"
export ROS_DOMAIN_ID="$DOMAIN_ID"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
# ROS python nodes need numpy 1.x next to Jazzy's compiled bindings; prepend a compat site only if the venv is numpy >= 2
ROS_NUMPY1_SITE="${ROS_NUMPY1_SITE:-$HOME/.local/lib/python3.12/site-packages}"
ROS_COMPAT_PYTHONPATH="${PYTHONPATH:-}"
if "$OPEN_SET_PYTHON" -c 'import numpy, sys; sys.exit(0 if int(numpy.__version__.split(".")[0]) >= 2 else 1)' 2>/dev/null; then
  ROS_COMPAT_PYTHONPATH="$ROS_NUMPY1_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi
PERCEPTION_URL="$(grep -E '^\s*perception_service_url:' "$MANIFEST" 2>/dev/null | head -1 | awk '{print $2}' || true)"

# ---------------------------------------------------------------- preflight
fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi; }
echo "Preflight ($ROBOT, map=$MAP, mode=$( ((FAKE)) && echo fake || echo real), out=$OUT)"
check "venv python $OPEN_SET_PYTHON" "test -x $OPEN_SET_PYTHON"
check "spot_tools_ros + dcist_launch_system installed" "ros2 pkg prefix spot_tools_ros && ros2 pkg prefix dcist_launch_system"
check "new nodes installed (rebuild dcist_ws if not)" "ros2 pkg executables spot_tools_ros | grep -q fiducial_localizer_node && ros2 pkg executables dcist_launch_system | grep -q static_occupancy_publisher_node"
check "recorded occupancy $MAP/occupancy_static.npz" "test -f $MAP/occupancy_static.npz"
check "recorded DSG $MAP/hydra/backend/dsg.json" "test -f $MAP/hydra/backend/dsg.json"
if ((DISPATCH)); then
  check "manifest $MANIFEST" "test -f $MANIFEST"
  check "manifest dispatches to ROS (robot: ros)" "grep -qE '^\s*robot:\s*ros' $MANIFEST"
  [[ -n "$PERCEPTION_URL" ]] && check "perception service $PERCEPTION_URL" "curl -s -m 3 -o /dev/null $PERCEPTION_URL/"
  check "fast-downward on PATH (scripts/setup.sh builds it into the venv)" "command -v fast-downward"
fi
check "venv sees ROS (rclpy + robot_executor_msgs)" "$OPEN_SET_PYTHON -c 'import rclpy, robot_executor_msgs.msg, spot_tools_ros.fiducial_localization'"
if ((FAKE)); then
  check "fake anchor '$FAKE_ANCHOR' and start '$FAKE_START' are 'x y yaw'" "[[ \$(wc -w <<<'$FAKE_ANCHOR') == 3 && \$(wc -w <<<'$FAKE_START') == 3 ]]"
else
  check "fiducials $MAP/fiducials.yaml (scripts/solve_map_fiducial.py)" "test -f $MAP/fiducials.yaml"
  check "robot credentials set (ADT4_BOSDYN_USERNAME/PASSWORD)" '[[ -n "$ADT4_BOSDYN_USERNAME" && -n "$ADT4_BOSDYN_PASSWORD" ]]'
  check "robot reachable: ping $ADT4_BOSDYN_IP" "ping -c1 -W2 $ADT4_BOSDYN_IP"
  check "robot answers gRPC (robot-id)" "$OPEN_SET_PYTHON -c \"import bosdyn.client; r=bosdyn.client.create_standard_sdk('preflight').create_robot('$ADT4_BOSDYN_IP'); print(r.ensure_client('robot-id').get_id().nickname)\""
  check "platform calibration $LAUNCH_SHARE/platforms/$PLATFORM/calibration.yaml" "test -f $LAUNCH_SHARE/platforms/$PLATFORM/calibration.yaml"
  check "spot URDF $SPOT_TOOLS_ROS_SHARE/urdf/spot.urdf.xacro" "test -f $SPOT_TOOLS_ROS_SHARE/urdf/spot.urdf.xacro"
  check "YOLOE weights (Pick) $OPEN_SET_SIM_ROOT/weights/yoloe-26l-seg.pt" "test -f $OPEN_SET_SIM_ROOT/weights/yoloe-26l-seg.pt"
fi
check "launch file resolves" "ros2 launch dcist_launch_system master.launch.yaml conf_name:=default sim_time:=false robot_name:=$ROBOT launch_spot_executor:=true launch_fiducial_localizer:=true launch_static_occupancy_publisher:=true --print"
((fail)) && { echo "preflight failed; fix the FAIL lines above" >&2; exit 1; }
((CHECK_ONLY)) && { echo "preflight passed"; exit 0; }

# ---------------------------------------------------------------- run
mkdir -p "$OUT/logs" "$ROS_LOG_DIR" "$MPLCONFIGDIR"
pids=()
# Background jobs of a non-interactive shell ignore SIGINT, so `kill -INT <ros2 launch>`
# does nothing and its nodes outlive this script (stale static map->odom publishers
# then fight the fiducial localizer on the next run). Signal the whole process tree.
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do descendants "$c"; echo "$c"; done; }
signal_trees() { local sig="$1" pid; for pid in "${pids[@]}"; do for p in $(descendants "$pid") "$pid"; do kill "-$sig" "$p" 2>/dev/null || true; done; done; }
any_alive() { local pid; for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && return 0; for p in $(descendants "$pid"); do kill -0 "$p" 2>/dev/null && return 0; done; done; return 1; }
cleanup() {
  trap - INT TERM EXIT
  echo; echo "Stopping..."
  signal_trees INT
  for _ in {1..20}; do any_alive || break; sleep 0.5; done
  signal_trees TERM
  for _ in {1..20}; do any_alive || break; sleep 0.5; done
  any_alive && signal_trees KILL
  wait 2>/dev/null || true
  echo "Output: $OUT  (plan: $OUT/plan, executor: $OUT/spot_executor, logs: $OUT/logs)"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

launch=(ros2 launch dcist_launch_system master.launch.yaml conf_name:=default sim_time:=false robot_name:=$ROBOT)
# Capture then match: under pipefail a `tf2_echo | grep -q` pipeline reports grep's SIGPIPE, never success.
have_tf() {
  local out
  out="$(timeout 3 ros2 run tf2_ros tf2_echo "$1" "$2" 2>/dev/null || true)"
  [[ "$out" == *Translation* ]]
}
have_topic() { timeout 4 ros2 topic echo --once "$1" >/dev/null 2>&1; }
tf_pose() {  # "x y yaw_deg" of child in parent, or nothing
  timeout 5 ros2 run tf2_ros tf2_echo "$1" "$2" 2>/dev/null | python3 -c '
import re, sys, math
t = sys.stdin.read()
m = re.search(r"Translation: \[([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\]", t)
q = re.search(r"Rotation: in RPY \(degree\) \[([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\]", t)
if m and q:
    print(f"{float(m.group(1)):.2f} {float(m.group(2)):.2f} {float(q.group(3)):.1f}")' 2>/dev/null || true
}

if ((FAKE)); then
  read -r AX AY AYAW <<<"$FAKE_ANCHOR"
  read -r SX SY SYAW <<<"$FAKE_START"
  # FakeSpot lives in the odom frame: odom_T_body = inv(map_T_odom) @ map_T_body
  read -r FX FY FYAW < <(python3 -c "
import math
ax, ay, ayaw, sx, sy, syaw = $AX, $AY, $AYAW, $SX, $SY, $SYAW
dx, dy = sx - ax, sy - ay
c, s = math.cos(-ayaw), math.sin(-ayaw)
print(f'{c*dx - s*dy:.4f} {s*dx + c*dy:.4f} {syaw - ayaw:.4f}')")
  echo "[1/6] fake robot: map->odom = ($AX, $AY, ${AYAW} rad); FakeSpot at odom ($FX, $FY, ${FYAW} rad) = map ($SX, $SY, $SYAW)"
else
  echo "[1/6] Spot sensors + robot state + calibration TFs"
  env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_spot_camera_driver:=true launch_spot_state_publisher:=true \
    launch_calibration_publisher:=true launch_spot_base_link:=true >"$OUT/logs/spot_sensors.log" 2>&1 & pids+=("$!")
  for _ in {1..60}; do have_tf "$ROBOT/odom" "$ROBOT/body" && break; sleep 1; done
  have_tf "$ROBOT/odom" "$ROBOT/body" || { echo "no $ROBOT/odom -> $ROBOT/body TF after 60 s; see $OUT/logs/spot_sensors.log" >&2; exit 1; }
  echo "      robot TF up"
fi

echo "[2/6] static occupancy grid from $MAP"
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_static_occupancy_publisher:=true >"$OUT/logs/occupancy.log" 2>&1 & pids+=("$!")
for _ in {1..30}; do have_topic "/$ROBOT/hydra/tsdf/occupancy" && break; sleep 1; done
have_topic "/$ROBOT/hydra/tsdf/occupancy" || { echo "no /$ROBOT/hydra/tsdf/occupancy after 30 s; see $OUT/logs/occupancy.log" >&2; exit 1; }
echo "      grid publishing"

if ((FAKE)); then
  echo "[3/6] fake anchor: static $ROBOT/map -> $ROBOT/odom"
  ros2 run tf2_ros static_transform_publisher --frame-id "$ROBOT/map" --child-frame-id "$ROBOT/odom" \
    --x "$AX" --y "$AY" --yaw "$AYAW" >"$OUT/logs/anchor.log" 2>&1 & pids+=("$!")
  sleep 1
else
  echo "[3/6] fiducial localizer (reads $MAP/fiducials.yaml)"
  env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_fiducial_localizer:=true >"$OUT/logs/localizer.log" 2>&1 & pids+=("$!")
  echo "      waiting for the robot to see the AprilTag (walk it to within ~3 m, facing the tag)..."
  while ! have_tf "$ROBOT/map" "$ROBOT/odom"; do
    grep -h "REFUSING\|LOCALIZED\|not visible\|observations" "$OUT/logs/localizer.log" 2>/dev/null | tail -1 | sed 's/^/      /' || true
    sleep 2
  done
  grep -h "LOCALIZED" "$OUT/logs/localizer.log" | tail -1 | sed 's/^/      /' || true
fi
if ((FAKE)); then
  echo "      $ROBOT/map -> $ROBOT/odom up (FakeSpot publishes odom -> base_link once the executor starts)"
else
  echo "      $ROBOT/map -> $ROBOT/odom up; robot at map pose: $(tf_pose "$ROBOT/map" "$ROBOT/base_link")"
fi

echo "[4/6] Spot executor"
exec_args=()
if ((FAKE)); then
  exec_args+=(spot_executor_fake:=true spot_executor_fake_x:="$FX" spot_executor_fake_y:="$FY" spot_executor_fake_yaw:="$FYAW"
              spot_executor_fake_kinematic:="$FAKE_KINEMATIC" spot_executor_detector:=none)
fi
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_spot_executor:=true "${exec_args[@]}" >"$OUT/logs/executor.log" 2>&1 & pids+=("$!")
exec_pid="${pids[-1]}"
for _ in {1..90}; do
  have_topic "/$ROBOT/spot_executor_node/inflated_occupancy_map" && break
  kill -0 "$exec_pid" 2>/dev/null || { echo "executor exited during startup; see $OUT/logs/executor.log" >&2; exit 1; }
  sleep 1
done
have_topic "/$ROBOT/spot_executor_node/inflated_occupancy_map" \
  || { echo "executor never published its inflated grid after 90 s; see $OUT/logs/executor.log" >&2; exit 1; }
echo "      executor has the grid (in $ROBOT/odom via TF); robot at map pose: $(tf_pose "$ROBOT/map" "$ROBOT/base_link")"

if ((RVIZ)); then
  echo "[5/6] RViz"
  "${launch[@]}" launch_rviz:=true >"$OUT/logs/rviz.log" 2>&1 & pids+=("$!")
else
  echo "[5/6] RViz skipped"
fi

if ((DISPATCH)); then
  echo "[6/6] grounding + planning: $MANIFEST"
  echo "      (SAM3 over the recorded keyframes, PDDL, then the Follow sequence is published to the executor)"
  if env -u PYTHONHOME "$OPEN_SET_VENV_ROOT/bin/open-set-run" "$MANIFEST" --output "$OUT/plan" >"$OUT/logs/plan.log" 2>&1; then
    echo "      plan dispatched"
  else
    echo "      open-set-run failed; see $OUT/logs/plan.log" >&2
    tail -20 "$OUT/logs/plan.log" >&2 || true
    exit 1
  fi
  grep -h "dispatched\|subscribers\|Follow\|start" "$OUT/plan/artifacts/execution.json" 2>/dev/null | head -5 | sed 's/^/      /' || true
  if [[ -f "$MAP/occupancy_static.npz" ]]; then
    env -u PYTHONPATH -u PYTHONHOME "$OPEN_SET_PYTHON" "$OPEN_SET_SIM_ROOT/scripts/plot_plan_on_occupancy.py" \
      "$OUT/plan" "$MAP/occupancy_static.npz" -o "$OUT/plan_on_occupancy.png" 2>/dev/null | sed 's/^/     /' || true
  fi
else
  echo "[6/6] dispatch skipped (--no-dispatch); publish an ActionSequenceMsg to /$ROBOT/omniplanner_node/compiled_plan_out"
fi

echo
echo "Robot pose in $ROBOT/map (Ctrl-C to stop everything; executor log: $OUT/logs/executor.log)"
t0=$(date +%s)
while kill -0 "$exec_pid" 2>/dev/null; do
  now=$(date +%s)
  status="$(grep -h "Navigating to\|Finished\|reached\|Executing\|WARN\|ERROR" "$OUT/logs/executor.log" 2>/dev/null | tail -1 | sed 's/.*spot_executor_node\]: //' | cut -c1-90)"
  echo "  t+$((now - t0))s  map pose (x y yaw_deg): $(tf_pose "$ROBOT/map" "$ROBOT/base_link")   $status"
  if ((WATCH_S > 0)) && ((now - t0 >= WATCH_S)); then echo "  watch window over"; break; fi
  sleep 3
done
