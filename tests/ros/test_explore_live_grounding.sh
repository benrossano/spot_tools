#!/usr/bin/env bash
# End-to-end exploration test without hardware: ground an object that is NOT in the prior
# archive from keyframes "recorded while exploring", plan to it, drive the fake robot there.
#
# Uses the output of tests/ros/replay_explore_on_bag.sh (live_agents/) and the recorded tour:
#   prior archive  = Hydra's agents/ keyframes recorded BEFORE --cut-s seconds of bag time
#   live stream    = the replay's keyframes recorded AT/AFTER the cut (what keyframe_recorder_node
#                    would have written since the session began), served by a perception service
#                    started here with --stream robot=<dir>
#   task           = fixed skeleton "visit <query>" (default: elevator, seen only in the corridor
#                    the tour walks at bag t 145-162 s), robot: ros to the fake executor through
#                    run_spot_offline_plan.sh --fake --explore
# Order: (4) CONTROL, the same task with the live tier disabled, must NOT ground (otherwise the
# prior already contains a match: raise --threshold, move --cut-s earlier or change --query; the
# test stops with exit 3); (5) the exploration run must ground through the live tier, plan,
# dispatch, and the fake executor must report success (asserted from events.jsonl / result.json).
#
#   tests/ros/test_explore_live_grounding.sh [--replay DIR] [--prior DIR] [--cut-s 140] [--query elevator]
#                                            [--threshold 0.30] [--port 8079] [--sam3-server URL] [--out DIR]
#                                            [--control-only]
set -euo pipefail
_SELF_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck disable=SC1091
source "$_SELF_DIR/../../../scripts/lib/paths.sh"
PRIOR="${ADT4_PRIOR_MAP:-$ADT4_OUTPUT_ROOT/spot_tour_hydra}"
REPLAY="$(ls -dt "$ADT4_OUTPUT_ROOT"/explore_replay_* 2>/dev/null | head -1 || true)"
CUT_S=140
QUERY=elevator
PORT=8079
SAM3_SERVER="${OPEN_SET_SAM3_SERVER_URL:-http://128.30.224.75:8000}"
OUT=""
THRESHOLD=0.30
CONTROL_ONLY=0
while (($#)); do
  case "$1" in
    --replay) REPLAY="$2"; shift 2 ;;
    --prior) PRIOR="$2"; shift 2 ;;
    --cut-s) CUT_S="$2"; shift 2 ;;
    --query) QUERY="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --sam3-server) SAM3_SERVER="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --control-only) CONTROL_ONLY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -d "$REPLAY/live_agents" ]] || { echo "no replay live_agents under '$REPLAY' (run tests/ros/replay_explore_on_bag.sh first)" >&2; exit 1; }
OUT="${OUT:-$ADT4_OUTPUT_ROOT/explore_grounding_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/logs"
if [[ -f "$OPEN_SET_WORKSPACE_ROOT/secrets.env" ]]; then set -a; source "$OPEN_SET_WORKSPACE_ROOT/secrets.env"; set +a; fi
export PATH="$OPEN_SET_VENV_ROOT/bin:$PATH"

echo "[1/6] split archives at bag t = ${CUT_S} s"
"$OPEN_SET_PYTHON" - "$PRIOR" "$REPLAY" "$OUT" "$CUT_S" <<'PYEOF'
import json, pathlib, sys
prior, replay, out, cut_s = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), float(sys.argv[4])
import re
meta = prior / "bag" / "metadata.yaml"
m = re.search(r"nanoseconds_since_epoch:\s*(\d+)", meta.read_text()) if meta.exists() else None
if m:
    bag_t0 = int(m.group(1))  # bag start; --cut-s counts from here (matches `ros2 bag info`)
else:  # no bag: fall back to the earliest recorded keyframe
    bag_t0 = min(int(x.name.split("_")[1]) for x in (replay / "live_agents").glob("agent_*_meta.json"))
cut_ns = bag_t0 + int(cut_s * 1e9)
def link_subset(src, dst, keep):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for meta in sorted(src.glob("agent_*_meta.json")):
        ts = int(meta.name.split("_")[1])
        if not keep(ts):
            continue
        for f in (meta, src / meta.name.replace("_meta.json", "_rgb.png"), src / meta.name.replace("_meta.json", "_depth.png")):
            (dst / f.name).symlink_to(f.resolve())
        n += 1
    (dst / "camera_calib.json").symlink_to((src / "camera_calib.json").resolve())
    return n
n_prior = link_subset(prior / "agents", out / "prior_agents", lambda ts: ts < cut_ns)
n_live = link_subset(replay / "live_agents", out / "live_agents_after_cut", lambda ts: ts >= cut_ns)
json.dump({"cut_ns": cut_ns, "bag_t0_ns": bag_t0, "n_prior": n_prior, "n_live": n_live}, open(out / "split.json", "w"), indent=2)
print(f"      prior archive {n_prior} keyframes (hydra agents before the cut), live stream {n_live} keyframes (replay at/after the cut)")
assert n_prior > 0 and n_live > 0
PYEOF
LIVE_DIR="$OUT/live_agents_after_cut"

echo "[2/6] manifest for '$QUERY'"
MANIFEST="$OUT/explore_${QUERY// /_}.yaml"
cat > "$MANIFEST" <<YEOF
manifest_version: 1
name: explore-live-grounding-${QUERY// /-}
expected_outcome: plan
allow_test_backends: true
strict_stack: false
scene:
  source: ${PRIOR}
  labelspace: {1: chair, 2: cone, 4: table}
task:
  task: Go to the ${QUERY}.
  scene_id: spot-tour-20260913
  budget: {max_acquisitions: 8, max_model_calls: 0, max_wall_time_s: 900}
  skeletons:
    - skeleton_id: visit-target
      name: Go to the ${QUERY}
      stages:
        - {action: visit, arguments: {target: ${QUERY}}}
      requirements:
        - {requirement_id: target-exists, predicate: exists, args: ["${QUERY}"]}
database: {backend: memory}
planner: {backend: omniplanner, domain_name: OpenSetRearrangementDomain, robot_id: spot, fd_search: "lazy_wastar([ff()],w=1)"}
evidence:
  archive: real_rgbd
  archive_frame_source: ${OUT}/prior_agents
  live: real_rgbd
  live_stream_id: robot
  live_frame_source: ${LIVE_DIR}
  perception_service_url: http://127.0.0.1:${PORT}
  admission_threshold: ${THRESHOLD}
  retrieval_top_k: 32
  segmenter_confidence: 0.25
  max_segments_per_frame: 5
  max_fused_candidates: 5
  observation_radius_m: 3.0
agent: {backend: none}
execution:
  backend: spot_stack
  execute_final_plan: true
  spot_stack:
    robot: ros
    dispatch_wait_s: 15.0
    wait_for_completion_s: 600.0
    robot_name: hamilton
    frame_id: hamilton/map
    start_position: tf
    occupancy_file: ${PRIOR}/occupancy_static.npz
    occupancy_from_places: false
    occupancy_resolution_m: 0.1
    path_commitment_weight: 1.0
    allow_unknown_target: true
    follow_progress_timeout_s: 20.0
tools:
  required_capabilities: [admitted_only, geometric_path, pddl, rgbd, scene_graph]
robot:
  urdf: ${OPEN_SET_SPOT_TOOLS_ROOT}/spot_tools_ros/urdf/spot.urdf.xacro
YEOF
echo "      $MANIFEST"

pids=()
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do descendants "$c"; echo "$c"; done; }
cleanup() {
  trap - EXIT INT TERM
  local all=() p pid
  for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] || continue; for p in $(descendants "$pid") "$pid"; do all+=("$p"); done; done
  for p in "${all[@]:-}"; do [[ -n "$p" ]] && kill -INT "$p" 2>/dev/null || true; done
  for _ in {1..20}; do local alive=0; for p in "${all[@]:-}"; do [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && alive=1; done; ((alive)) || break; sleep 1; done
  for p in "${all[@]:-}"; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true; done
  sleep 2
  for p in "${all[@]:-}"; do [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "[3/6] perception service on :$PORT with --stream robot=$LIVE_DIR (SAM3 at $SAM3_SERVER)"
(cd "$OPEN_SET_SIM_ROOT" && env -u PYTHONHOME -u PYTHONPATH open-set-perception-service --device cuda --port "$PORT" --top-k 32 --confidence 0.25 \
    --sam3-mode server --sam3-server-url "$SAM3_SERVER" --stream "robot=$LIVE_DIR") >"$OUT/logs/perception.log" 2>&1 & pids+=("$!")
for _ in {1..240}; do curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"robot"' || { echo "perception service not up with stream 'robot'; see $OUT/logs/perception.log" >&2; exit 1; }
echo "      up"

echo "[4/6] control: same task, live tier disabled (must NOT ground)"
CTRL="$OUT/control_${QUERY// /_}.yaml"
sed -e 's/^  live: real_rgbd/  live: none/' -e '/^  live_stream_id:/d' -e '/^  live_frame_source:/d' \
    -e 's/^  execute_final_plan: true/  execute_final_plan: false/' -e 's/^    start_position: tf/    start_position: [4.42, 2.43, 0.0]/' \
    -e 's/^name: /name: control-/' "$MANIFEST" > "$CTRL"
(cd "$OPEN_SET_SIM_ROOT" && env -u PYTHONHOME open-set-run "$CTRL" --output "$OUT/control") >"$OUT/logs/control.log" 2>&1 || true
ctrl_summary="$("$OPEN_SET_PYTHON" - "$OUT/control" <<'PYEOF2'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); met = None; conf = []
if (d / "result.json").exists():
    met = json.loads((d / "result.json").read_text()).get("expectation_met")
if (d / "events.jsonl").exists():
    for l in (d / "events.jsonl").read_text().splitlines():
        e = json.loads(l); p = e.get("payload") or {}
        if str(p.get("tool_id", "")).startswith("evidence.") and p.get("graph_patch"):
            conf.append(((p.get("evidence") or {}).get("confidence"), p["tool_id"]))
print(f"{met} {conf}")
PYEOF2
)"
echo "      planned=${ctrl_summary%% *}  admissions=${ctrl_summary#* }"
if [[ "${ctrl_summary%% *}" == "True" ]]; then
  echo "      PRECONDITION FAILED: the prior archive already grounds '$QUERY' at threshold $THRESHOLD." >&2
  echo "      Raise --threshold above the confidence shown, move --cut-s earlier, or pick another --query." >&2
  exit 3
fi
echo "      ok    without live keyframes '$QUERY' is not groundable"
((CONTROL_ONLY)) && { echo "Output: $OUT"; exit 0; }

echo "[5/6] fake robot stack (--explore) + task; dispatch to the fake executor"
export OPEN_SET_LIVE_AGENTS="$LIVE_DIR"
STACK="$OPEN_SET_SPOT_TOOLS_ROOT/tests/ros/run_spot_offline_plan.sh"
set +e
"$STACK" --fake --explore --no-rviz --manifest "$MANIFEST" --out "$OUT/run" --watch-s 5 >"$OUT/logs/run.log" 2>&1
rc=$?
set -e
grep -E '^\[|ok$|success|admitt|FAIL|failed' "$OUT/logs/run.log" | grep -v '^  ok' | tail -12 | sed 's/^/      /'

echo "[6/6] assertions"
set +e
"$OPEN_SET_PYTHON" - "$OUT/run/plan" "$QUERY" <<'PYEOF'
import json, pathlib, sys
plan = pathlib.Path(sys.argv[1]); query = sys.argv[2]
ok = True
events = [json.loads(l) for l in (plan / "events.jsonl").read_text().splitlines() if l.strip()]
tools = [e for e in events if (e.get("payload") or {}).get("tool_id", "").startswith("evidence.")]
by_tier = {}
for e in tools:
    p = e["payload"]; tier = p["tool_id"].split(".")[1]
    ev = p.get("evidence") or {}
    by_tier.setdefault(tier, []).append({"admitted": p.get("graph_patch") is not None, "conf": ev.get("confidence"), "status": ev.get("admission_status") or ev.get("status")})
for tier, items in by_tier.items():
    print(f"      evidence.{tier}: {len(items)} search(es): " + ", ".join(f"admitted={i['admitted']} conf={i['conf']}" for i in items))
if any(i["admitted"] for i in by_tier.get("archive", [])):
    print("      FAIL the prior archive admitted the target; the cut did not remove it"); ok = False
if not any(i["admitted"] for i in by_tier.get("live", [])):
    print("      FAIL the live stream did not admit the target"); ok = False
res = json.loads((plan / "result.json").read_text()) if (plan / "result.json").exists() else {}
met = res.get("expectation_met")
print(f"      result.json expectation_met: {met}")
ok &= bool(met)
# where the target was grounded, from the graph patch
target = None
for e in tools:
    patch = (e.get("payload") or {}).get("graph_patch")
    if patch:
        for op in patch.get("operations", []):
            props = op.get("properties") or {}
            if props.get("position"):
                target = [round(float(v), 2) for v in props["position"]]
print(f"      grounded '{query}' at {target}")
exe = res.get("execution") or {}
rr = exe.get("robot_result") or {}
print(f"      execution: dispatched={exe.get('dispatched')} robot success={rr.get('success')} actions={[a.get('action') for a in rr.get('actions', [])]}")
ok &= bool(rr.get("success"))
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
result=$?
set -e
echo "Output: $OUT"
exit $result
