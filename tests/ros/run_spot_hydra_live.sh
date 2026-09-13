#!/usr/bin/env bash
# Live Hydra mapping on the real Spot while someone teleops (tablet or SDK move.py).
#
# Runs on the machine that has the ZED plugged in and network access to the robot:
#   spot_sensor_node (odometry TF + joint states over gRPC, read-only, no lease)
#   robot_state_publisher + platform calibration TFs (ZED mount) + body->base_link
#   zed_wrapper                      (unless --zed none: ZED topics come from elsewhere)
#   YOLOE instance segmentation, Hydra, DSG saver, RViz (unless --no-rviz)
# and optionally records the planner inputs. Ctrl-C saves the DSG and shuts Hydra down
# cleanly so the keyframe archive (<out>/agents) is complete for open-set grounding.
#
#   tests/ros/run_spot_hydra_live.sh [--out DIR] [--zed local|none] [--no-rviz] [--record] [--record-images]
#                                    [--robot hamilton] [--platform smaug] [--domain-id N] [--check]
#
# Needs: ADT4_BOSDYN_IP / ADT4_BOSDYN_USERNAME / ADT4_BOSDYN_PASSWORD (or BOSDYN_CLIENT_* + SPOT_IP from
# the spot-sdk-scripts env.sh) and a sourced install that provides hydra_ros, semantic_inference_ros,
# spot_tools_ros, ianvs, dcist_launch_system and zed_wrapper.
#   split layout (this desktop):  ADT4_WS=open_set_sim  DCIST_WS=dcist_ws  ZED_WS=~/zed_ws
#   field layout (one dcist_ws):  ADT4_WS=DCIST_WS=/path/to/dcist_ws, ADT4_ENV=$ADT4_WS/.adt4_env, ZED_WS unset
set -euo pipefail

# Fall back to the checkout this script lives in rather than a fixed path:
# .../<workspace>/open_set_sim/spot_tools/tests/ros/ -> <workspace>
_SELF_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
OPEN_SET_NAV=${OPEN_SET_NAV_WORKSPACE:-$(cd "$_SELF_DIR/../../../.." && pwd)}
ADT4_WS_DIR=${ADT4_WS:-$OPEN_SET_NAV/open_set_sim}
DCIST_WS=${DCIST_WS:-$OPEN_SET_NAV/dcist_ws}
ZED_WS=${ZED_WS:-$HOME/zed_ws}
ROBOT=hamilton
PLATFORM=smaug
OUT=""
ZED=local
RVIZ=1
RECORD=0
RECORD_IMAGES=0
DOMAIN_ID=88
CHECK_ONLY=0
while (($#)); do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --zed) ZED="$2"; shift 2 ;;
    --no-rviz) RVIZ=0; shift ;;
    --record) RECORD=1; shift ;;
    --record-images) RECORD=1; RECORD_IMAGES=1; shift ;;
    --robot) ROBOT="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --domain-id) DOMAIN_ID="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$OUT" ]] && OUT="$OPEN_SET_NAV/adt4_output/spot_live_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------- environment
set +u
source /opt/ros/jazzy/setup.bash
[[ -f "$ADT4_WS_DIR/install/setup.bash" ]] && source "$ADT4_WS_DIR/install/setup.bash"
source "$DCIST_WS/install/setup.bash"
[[ -f "$ZED_WS/install/setup.bash" ]] && source "$ZED_WS/install/setup.bash"
set -u
# robot credentials: accept the spot-sdk-scripts names too
export ADT4_BOSDYN_IP="${ADT4_BOSDYN_IP:-${SPOT_IP:-192.168.80.3}}"
export ADT4_BOSDYN_USERNAME="${ADT4_BOSDYN_USERNAME:-${BOSDYN_CLIENT_USERNAME:-}}"
export ADT4_BOSDYN_PASSWORD="${ADT4_BOSDYN_PASSWORD:-${BOSDYN_CLIENT_PASSWORD:-}}"
export ADT4_WS="$ADT4_WS_DIR"
export ADT4_ENV="${ADT4_ENV:-$ADT4_WS_DIR/.adt4_env}"
export ADT4_PLATFORM_ID="$PLATFORM"
export ADT4_ROBOT_NAME="$ROBOT"
export ADT4_OUTPUT_DIR="$OUT"
# resolve shares through the installed packages so merged and isolated installs both work
export SPOT_TOOLS_ROS_SHARE="${SPOT_TOOLS_ROS_SHARE:-$(ros2 pkg prefix spot_tools_ros 2>/dev/null || echo /nonexistent)/share/spot_tools_ros}"
export ZED_WRAPPER_SHARE="${ZED_WRAPPER_SHARE:-$(ros2 pkg prefix zed_wrapper 2>/dev/null || echo /nonexistent)/share/zed_wrapper}"
LAUNCH_SHARE="$(ros2 pkg prefix dcist_launch_system 2>/dev/null || echo /nonexistent)/share/dcist_launch_system"
export YOLO_CONFIG_DIR="$ADT4_WS_DIR/.ultralytics"
export MPLCONFIGDIR="$ADT4_ENV/matplotlib"
export ROS_LOG_DIR="$OUT/logs/ros"
export ROS_DOMAIN_ID="$DOMAIN_ID"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
# ROS python nodes (sensor node, YOLOE) need numpy 1.x next to Jazzy's compiled bindings; only
# prepend a compat site when the venv itself ships numpy >= 2
ROS_NUMPY1_SITE="${ROS_NUMPY1_SITE:-$HOME/.local/lib/python3.12/site-packages}"
VENV_PY="$ADT4_ENV/spark_env/bin/python"
NEED_NUMPY1_COMPAT=0
ROS_COMPAT_PYTHONPATH="${PYTHONPATH:-}"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c 'import numpy, sys; sys.exit(0 if int(numpy.__version__.split(".")[0]) >= 2 else 1)' 2>/dev/null; then
  NEED_NUMPY1_COMPAT=1
  ROS_COMPAT_PYTHONPATH="$ROS_NUMPY1_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

# ---------------------------------------------------------------- preflight
fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi; }
echo "Preflight ($ROBOT @ $ADT4_BOSDYN_IP, ZED=$ZED, out=$OUT)"
check "robot credentials set (ADT4_BOSDYN_USERNAME/PASSWORD)" '[[ -n "$ADT4_BOSDYN_USERNAME" && -n "$ADT4_BOSDYN_PASSWORD" ]]'
check "robot reachable: ping $ADT4_BOSDYN_IP" "ping -c1 -W2 $ADT4_BOSDYN_IP"
check "venv python $VENV_PY" "test -x $VENV_PY"
check "robot answers gRPC (robot-id)" "$VENV_PY -c \"import bosdyn.client; r=bosdyn.client.create_standard_sdk('preflight').create_robot('$ADT4_BOSDYN_IP'); print(r.ensure_client('robot-id').get_id().nickname)\""
check "GPU visible" "nvidia-smi"
check "YOLOE weights $ADT4_WS/weights/yoloe-26l-seg.pt" "test -f $ADT4_WS/weights/yoloe-26l-seg.pt"
check "hydra_ros / semantic_inference_ros / spot_tools_ros / ianvs built" "ros2 pkg prefix hydra_ros && ros2 pkg prefix semantic_inference_ros && ros2 pkg prefix spot_tools_ros && ros2 pkg prefix ianvs"
check "spot_tools_ros + bosdyn importable in the venv" "$VENV_PY -c 'import spot_tools_ros.spot_sensors, bosdyn.client'"
if ((NEED_NUMPY1_COMPAT)); then
  check "venv has numpy>=2: numpy 1.x compat site present ($ROS_NUMPY1_SITE)" "test -f $ROS_NUMPY1_SITE/numpy/__init__.py"
elif [[ -x "$VENV_PY" ]]; then
  echo "  ok    venv numpy < 2, no compat site needed"
fi
check "platform calibration $LAUNCH_SHARE/platforms/$PLATFORM/calibration.yaml" "test -f $LAUNCH_SHARE/platforms/$PLATFORM/calibration.yaml"
check "spot URDF $SPOT_TOOLS_ROS_SHARE/urdf/spot.urdf.xacro" "test -f $SPOT_TOOLS_ROS_SHARE/urdf/spot.urdf.xacro"
if [[ "$ZED" == "local" ]]; then
  check "zed_wrapper built ($ZED_WRAPPER_SHARE)" "test -f $ZED_WRAPPER_SHARE/launch/zed_camera.launch.py"
  check "ZED SDK installed (/usr/local/zed)" "test -d /usr/local/zed"
  check "ZED on USB (vendor 2b03)" '[[ "$(lsusb)" == *2b03* || "$(lsusb)" == *2B03* ]]'
fi
check "launch file resolves" "ros2 launch dcist_launch_system master.launch.yaml conf_name:=default sim_time:=false robot_name:=$ROBOT launch_hydra:=true launch_spot_camera_driver:=true --print"
((fail)) && { echo "preflight failed; fix the FAIL lines above" >&2; exit 1; }
((CHECK_ONLY)) && { echo "preflight passed"; exit 0; }

# ---------------------------------------------------------------- run
mkdir -p "$OUT/logs" "$ROS_LOG_DIR" "$MPLCONFIGDIR"
pids=()
hydra_pid=""
cleanup() {
  trap - INT TERM EXIT
  echo; echo "Stopping: saving DSG and shutting Hydra down cleanly..."
  timeout 30 ros2 service call /$ROBOT/dsg_saver/save_dsg dcist_launch_system_msgs/srv/SaveDsg \
    "{save_path: '$OUT/dsg_snapshot_with_mesh.json', include_mesh: true}" >"$OUT/logs/snapshot.log" 2>&1 || true
  if [[ -n "$hydra_pid" ]] && kill -0 "$hydra_pid" 2>/dev/null; then
    timeout 30 ros2 service call /$ROBOT/shutdown std_srvs/srv/Empty '{}' >"$OUT/logs/shutdown.log" 2>&1 || true
    for _ in {1..240}; do kill -0 "$hydra_pid" 2>/dev/null || break; sleep 0.5; done   # Hydra can take minutes to flush
  fi
  for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  for _ in {1..60}; do alive=0; for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done; ((alive)) || break; sleep 0.5; done
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "Output: $OUT  (Hydra archive: $OUT/agents, $OUT/hydra; logs: $OUT/logs)"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

launch=(ros2 launch dcist_launch_system master.launch.yaml conf_name:=default sim_time:=false robot_name:=$ROBOT)

echo "[1/6] Spot sensors + robot state + calibration TFs"
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_spot_camera_driver:=true launch_spot_state_publisher:=true \
  launch_calibration_publisher:=true launch_spot_base_link:=true >"$OUT/logs/spot_sensors.log" 2>&1 & pids+=("$!")
# Capture first, then match. Under `set -o pipefail` a `tf2_echo | grep -q`
# pipeline can never report success: tf2_echo streams until killed, so grep -q
# always exits on the first match and SIGPIPEs it, and pipefail surfaces that
# 141 as the pipeline's status. The check then fails even though the TF is up.
have_robot_tf() {
  local out
  out="$(timeout 3 ros2 run tf2_ros tf2_echo "$ROBOT/odom" "$ROBOT/body" 2>/dev/null || true)"
  [[ "$out" == *Translation* ]]
}
for _ in {1..60}; do
  have_robot_tf && break; sleep 1
done
have_robot_tf \
  || { echo "no $ROBOT/odom -> $ROBOT/body TF after 60 s; see $OUT/logs/spot_sensors.log" >&2; exit 1; }
echo "      robot TF up"

if [[ "$ZED" == "local" ]]; then
  echo "[2/6] ZED"
  "${launch[@]}" launch_zed:=true >"$OUT/logs/zed.log" 2>&1 & pids+=("$!")
else
  echo "[2/6] ZED: expecting /$ROBOT/${ROBOT}_zed/* from elsewhere"
fi
# The RGB topic moved between zed-ros2-wrapper versions: older builds publish
# rgb/camera_info, current ones rgb/color/rect/camera_info. Accept either, so
# this does not silently wait out its timeout after a wrapper upgrade.
ZED_RGB_INFO_CANDIDATES=(
  "/$ROBOT/${ROBOT}_zed/rgb/color/rect/camera_info"
  "/$ROBOT/${ROBOT}_zed/rgb/camera_info"
)
have_zed_info() {
  local topic
  for topic in "${ZED_RGB_INFO_CANDIDATES[@]}"; do
    if timeout 3 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
      ZED_RGB_INFO="$topic"
      return 0
    fi
  done
  return 1
}
for _ in {1..90}; do have_zed_info && break; sleep 1; done
have_zed_info \
  || { echo "no ZED camera_info after 90 s (tried: ${ZED_RGB_INFO_CANDIDATES[*]}); see $OUT/logs/zed.log" >&2; exit 1; }
echo "      ZED publishing ($ZED_RGB_INFO)"

echo "[3/6] YOLOE instance segmentation"
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_instance_segmentation:=true >"$OUT/logs/semantic.log" 2>&1 & pids+=("$!")
echo "[4/6] Hydra"
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_hydra:=true labelspace_name:=instance_seg >"$OUT/logs/hydra.log" 2>&1 & pids+=("$!")
hydra_pid="${pids[-1]}"
echo "[5/6] DSG saver"
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_dsg_saver:=true >"$OUT/logs/saver.log" 2>&1 & pids+=("$!")
if ((RVIZ)); then
  echo "[6/6] RViz"
  "${launch[@]}" launch_rviz:=true >"$OUT/logs/rviz.log" 2>&1 & pids+=("$!")
fi
if ((RECORD)); then
  topics=(/tf /tf_static /$ROBOT/odom /$ROBOT/joint_states /$ROBOT/hydra/tsdf/occupancy /$ROBOT/hydra/backend/dsg)
  ((RECORD_IMAGES)) && topics+=(/$ROBOT/${ROBOT}_zed/rgb/color/rect/image /$ROBOT/${ROBOT}_zed/depth/depth_registered "$ZED_RGB_INFO" /$ROBOT/${ROBOT}_zed/depth/camera_info)
  ros2 bag record -o "$OUT/bag" --storage mcap "${topics[@]}" >"$OUT/logs/record.log" 2>&1 & pids+=("$!")
fi

echo "Waiting for Hydra and YOLOE..."
for _ in {1..90}; do
  nodes="$(ros2 node list 2>/dev/null || true)"
  grep -qx "/$ROBOT/hydra" <<<"$nodes" && grep -qx "/$ROBOT/${ROBOT}_zed/semantic_inference" <<<"$nodes" && break
  kill -0 "$hydra_pid" 2>/dev/null || { echo "Hydra exited during startup; see $OUT/logs/hydra.log" >&2; exit 1; }
  sleep 1
done
echo
echo "Hydra is mapping. Teleop now (tablet, or in the SDK repo: estop.py + move.py teleop)."
echo "Watch: ros2 topic hz /$ROBOT/hydra/tsdf/occupancy      RViz: /$ROBOT/hydra/backend/dsg, /$ROBOT/hydra/tsdf/occupancy"
echo "Ctrl-C here when done: the DSG is saved and Hydra shuts down cleanly (can take a few minutes)."
while kill -0 "$hydra_pid" 2>/dev/null; do sleep 5; done
echo "Hydra exited; see $OUT/logs/hydra.log" >&2
