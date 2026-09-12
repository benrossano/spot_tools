#!/usr/bin/env bash
# Replay the real Spot bag through instance segmentation + Hydra and record the planner's
# inputs (/<robot>/hydra/tsdf/occupancy, /tf, /tf_static, /<robot>/odom) into a small derived
# bag for offline planner evaluation (tests/ros/replay_planner_on_bag.py).
#
# Mirrors open_set_sim/scripts/run_behavior1k_bag.sh, minus the TF bridge: this bag is already
# namespaced (hamilton/odom -> hamilton/body -> hamilton/base_link -> ZED chain).
#
#   tests/ros/record_bag_occupancy.sh [--bag DIR] [--out DIR] [--rate R] [--duration S] [--domain-id N]
set -euo pipefail

OPEN_SET_NAV=/home/ben-rossano/research/openset_nav_dir
ADT4_WS_DIR=$OPEN_SET_NAV/open_set_sim
DCIST_WS=$OPEN_SET_NAV/dcist_ws
BAG=/home/ben-rossano/spot_bags/bulding_1_infinite/recorded_data
OUT=$OPEN_SET_NAV/adt4_output/spot_bag_occupancy_$(date +%Y%m%d_%H%M%S)
RATE=1.0
DURATION=""
DOMAIN_ID=88
ROBOT=hamilton
while (($#)); do
  case "$1" in
    --bag) BAG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --domain-id) DOMAIN_ID="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -d "$BAG" ]] || { echo "bag not found: $BAG" >&2; exit 1; }
mkdir -p "$OUT/logs"

set +u
source /opt/ros/jazzy/setup.bash
source "$ADT4_WS_DIR/install/setup.bash"
source "$DCIST_WS/install/setup.bash"
set -u
export ADT4_WS="$ADT4_WS_DIR"
export ADT4_ENV="$ADT4_WS_DIR/.adt4_env"
export ADT4_PLATFORM_ID=smaug
export ADT4_ROBOT_NAME=$ROBOT
export ADT4_OUTPUT_DIR="$OUT"
export YOLO_CONFIG_DIR="$ADT4_WS_DIR/.ultralytics"
export MPLCONFIGDIR="$ADT4_WS_DIR/.adt4_env/matplotlib"
export ROS_LOG_DIR="$OUT/logs/ros"
export ROS_DOMAIN_ID="$DOMAIN_ID"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
# ROS python nodes need numpy 1.x next to Jazzy's compiled bindings (see run_behavior1k_bag.sh)
ROS_NUMPY1_SITE="${ROS_NUMPY1_SITE:-$HOME/.local/lib/python3.12/site-packages}"
[[ -f "$ROS_NUMPY1_SITE/numpy/__init__.py" ]] || { echo "numpy-1 compat site missing: $ROS_NUMPY1_SITE" >&2; exit 1; }
ROS_COMPAT_PYTHONPATH="$ROS_NUMPY1_SITE${PYTHONPATH:+:$PYTHONPATH}"

pids=()
cleanup() {
  trap - INT TERM EXIT
  echo "Stopping..."
  for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  for _ in {1..60}; do
    alive=0; for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    ((alive)) || break; sleep 0.5
  done
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# camera topics must be absolute: the instance-segmentation node lives in the <robot>_zed
# namespace, so the relative default would resolve to /<robot>/<robot>_zed/<robot>_zed/...
launch=(ros2 launch dcist_launch_system master.launch.yaml conf_name:=default sim_time:=true robot_name:=$ROBOT
  camera_rgb_topic:=/$ROBOT/${ROBOT}_zed/rgb/image_rect_color
  camera_depth_topic:=/$ROBOT/${ROBOT}_zed/depth/depth_registered
  camera_info_topic:=/$ROBOT/${ROBOT}_zed/rgb/camera_info)
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_instance_segmentation:=true >"$OUT/logs/semantic.log" 2>&1 & pids+=("$!")
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_hydra:=true labelspace_name:=instance_seg >"$OUT/logs/hydra.log" 2>&1 & pids+=("$!")
hydra_pid="${pids[-1]}"

echo "Waiting for Hydra and YOLOE..."
ready=0
for _ in {1..90}; do
  nodes="$(ros2 node list 2>/dev/null || true)"
  if grep -qx "/$ROBOT/hydra" <<<"$nodes" && grep -qx "/$ROBOT/${ROBOT}_zed/semantic_inference" <<<"$nodes"; then ready=1; break; fi
  kill -0 "$hydra_pid" 2>/dev/null || { echo "Hydra exited during startup; see $OUT/logs/hydra.log" >&2; exit 1; }
  sleep 1
done
((ready)) || { echo "mapper did not become ready; see $OUT/logs" >&2; exit 1; }
sleep 5

ros2 bag record -o "$OUT/derived_bag" --storage mcap \
  "/$ROBOT/hydra/tsdf/occupancy" /tf /tf_static "/$ROBOT/odom" /clock \
  >"$OUT/logs/record.log" 2>&1 & pids+=("$!")
sleep 2

limit=()
[[ -n "$DURATION" ]] && limit=(--playback-duration "$DURATION")
echo "Playing $BAG at ${RATE}x"
ros2 run ianvs play_rosbag "$BAG" \
  --topics /tf /tf_static "/$ROBOT/odom" \
    "/$ROBOT/${ROBOT}_zed/rgb/image_rect_color" "/$ROBOT/${ROBOT}_zed/depth/depth_registered" \
    "/$ROBOT/${ROBOT}_zed/rgb/camera_info" "/$ROBOT/${ROBOT}_zed/depth/camera_info" \
  --clock --delay 2.0 -r "$RATE" --disable-keyboard-controls "${limit[@]}" \
  >"$OUT/logs/bag.log" 2>&1 &
play_pid=$!; pids+=("$play_pid")
while kill -0 "$play_pid" 2>/dev/null; do
  kill -0 "$hydra_pid" 2>/dev/null || { echo "Hydra exited during playback; see $OUT/logs/hydra.log" >&2; exit 1; }
  sleep 5
done
sleep 10  # let Hydra drain and the recorder flush
echo "Derived bag: $OUT/derived_bag"
