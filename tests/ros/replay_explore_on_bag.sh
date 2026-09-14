#!/usr/bin/env bash
# Offline test of the Hydra-free exploration nodes on the recorded tour bag.
#
# Plays the run's bag (ZED RGB-D + camera_info + /tf incl. Hydra's map->odom) through
#   occupancy_mapper_node   prior occupancy_static.npz + live depth -> hydra/tsdf/occupancy, occupancy_live.npz
#   keyframe_recorder_node  RGB-D + TF -> live_agents/ in the archive layout
# then compares the result with what Hydra produced for the same run
# (tests/ros/evaluate_explore_replay.py): grid agreement / growth, keyframe poses vs
# Hydra's agents/, camera_calib.json equality.
#
#   tests/ros/replay_explore_on_bag.sh [--map DIR] [--out DIR] [--rate R] [--no-prior] [--domain-id N]
#
# --no-prior starts the mapper from an empty grid (pure live map, for comparison).
set -euo pipefail
_SELF_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck disable=SC1091
source "$_SELF_DIR/../../../scripts/lib/paths.sh"
MAP="${ADT4_PRIOR_MAP:-$ADT4_OUTPUT_ROOT/spot_tour_hydra}"
OUT=""
RATE=2.0
PRIOR=1
ROBOT="${ADT4_ROBOT_NAME:-hamilton}"
DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
while (($#)); do
  case "$1" in
    --map) MAP="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --no-prior) PRIOR=0; shift ;;
    --domain-id) DOMAIN_ID="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -d "$MAP/bag" ]] || { echo "no bag at $MAP/bag" >&2; exit 1; }
OUT="${OUT:-$ADT4_OUTPUT_ROOT/explore_replay_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/logs"

set +u
source /opt/ros/jazzy/setup.bash
source "$DCIST_INSTALL/setup.bash"
set -u
# shellcheck disable=SC1091
source "$OPEN_SET_SIM_ROOT/scripts/lib/ros_python_env.sh"
ROS_COMPAT_PYTHONPATH="$(ros_node_pythonpath "$OPEN_SET_PYTHON")" || exit 1
export ADT4_WS ADT4_ENV
export ADT4_PLATFORM_ID="${ADT4_PLATFORM_ID:-smaug}" ADT4_ROBOT_NAME="$ROBOT"
export ADT4_OUTPUT_DIR="$OUT" ADT4_PRIOR_MAP="$MAP"
export ROS_LOG_DIR="$OUT/logs/ros" ROS_DOMAIN_ID="$DOMAIN_ID" ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY

pids=()
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do descendants "$c"; echo "$c"; done; }
stop_tree() {  # TERM the launch and every node under it (snapshot first: orphans re-parent to init), then KILL
  local root="$1" all=() p
  for p in $(descendants "$root") "$root"; do all+=("$p"); done
  for p in "${all[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  for _ in {1..20}; do local alive=0; for p in "${all[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive)) || break; sleep 1; done
  for p in "${all[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
}
cleanup() { for p in "${pids[@]:-}"; do [[ -n "$p" ]] && stop_tree "$p"; done; }
trap cleanup EXIT

launch=(ros2 launch dcist_launch_system master.launch.yaml conf_name:=default sim_time:=false robot_name:="$ROBOT")
extra=()
((PRIOR)) || extra+=(occupancy_mapper_prior:=none)
echo "[1/3] mapper + recorder (domain $DOMAIN_ID, out $OUT, prior=$PRIOR)"
env PYTHONPATH="$ROS_COMPAT_PYTHONPATH" "${launch[@]}" launch_occupancy_mapper:=true launch_keyframe_recorder:=true \
  "${extra[@]}" >"$OUT/logs/nodes.log" 2>&1 & pids+=("$!")
for _ in {1..30}; do ros2 node list 2>/dev/null | grep -q occupancy_mapper && break; sleep 1; done
ros2 node list 2>/dev/null | grep -q keyframe_recorder || { echo "nodes did not start; see $OUT/logs/nodes.log" >&2; exit 1; }
echo "      nodes up"

echo "[2/3] playing $MAP/bag at x$RATE"
t0=$(date +%s)
ros2 bag play "$MAP/bag" --rate "$RATE" >"$OUT/logs/bag_play.log" 2>&1
echo "      playback done in $(( $(date +%s) - t0 )) s; letting the nodes drain"
sleep 3
# background jobs of a non-interactive shell ignore SIGINT (and their children inherit that),
# so stop the launch with SIGTERM; the nodes treat it like Ctrl-C and save on the way out.
stop_tree "${pids[0]}"
pids=()
grep -h 'keyframe [0-9]*:' "$OUT/logs/nodes.log" | tail -1 | sed 's/^/      /' || true
grep -h 'integrated [0-9]* depth' "$OUT/logs/nodes.log" | tail -1 | sed 's/^/      /' || true

echo "[3/3] evaluation"
"$OPEN_SET_PYTHON" "$_SELF_DIR/evaluate_explore_replay.py" "$OUT" --prior "$MAP" --png "$OUT/explore_replay.png" | tee "$OUT/evaluation.txt"
echo "Output: $OUT"
