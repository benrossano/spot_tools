"""Drive MidLevelPlanner in a closed replan loop against a synthetic occupancy grid.

No ROS, no robot. The robot only ever sees cells within --crop metres (everything else is
UNKNOWN), so it discovers a wall as it approaches and has to route around it across
successive replans. That is the real behaviour: MidLevelPlanner is a *local* planner with a
lookahead of `lookahead_distance_grid` cells, so one call never solves the whole problem.

A failure here is a planner problem, isolated from ROS and from the robot.

Run:  ./spot_tools_env/bin/python test_mid_level_planner.py [out.png]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_executor_interface.mid_level_planner import MidLevelPlanner, OccupancyMap

# Grid conventions, from spot_tools_ros/occupancy_grid_ros_updater.py and
# fake_occupancy_publisher.py: (h, w) int8, 0 free / 100 occupied / -1 unknown.
FREE, OCCUPIED, UNKNOWN = 0, 100, -1
RESOLUTION = 0.12  # m per cell, the fake publisher's default
WIDTH = HEIGHT = 200  # 24 m x 24 m
ORIGIN_XY = (-12.0, -12.0)  # grid lower-left corner in odom, so odom (0,0) is centred

START_XY = (-7.0, 0.0)
GOAL_XY = (7.0, 0.0)


class Feedback:
    """Minimal stand-in for FeedbackCollector: the planner only ever calls .print()."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def print(self, level, s):
        if self.verbose and level != "DEBUG":
            print(f"{level} {s}")


def homo(x=0.0, y=0.0, z=0.0):
    """4x4 homogeneous transform with no rotation; poses here are axis-aligned."""
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def world_to_cell(x, y):
    return int((y - ORIGIN_XY[1]) / RESOLUTION), int((x - ORIGIN_XY[0]) / RESOLUTION)


def build_world():
    """Ground truth the robot does not get to see all at once."""
    grid = np.full((HEIGHT, WIDTH), FREE, dtype=np.int8)
    # Wall at x = 0, from the bottom edge up to y = +3.6 m, so the only way past is north.
    grid[0:130, 98:103] = OCCUPIED
    # Two freestanding blocks so the detour is not a clean straight line.
    grid[135:155, 60:78] = OCCUPIED
    grid[150:170, 120:140] = OCCUPIED
    return grid


def visible(world, robot_xy, crop_m):
    """What the robot can see right now: cells beyond crop_m are UNKNOWN.

    Mirrors fake_occupancy_publisher.crop_around_robot, which is how the real stack
    limits the planner to local information.
    """
    if crop_m <= 0:
        return world.copy()
    grid = np.full_like(world, UNKNOWN)
    ri, rj = world_to_cell(*robot_xy)
    r = int(crop_m / RESOLUTION)
    i0, i1 = max(0, ri - r), min(HEIGHT, ri + r + 1)
    j0, j1 = max(0, rj - r), min(WIDTH, rj + r + 1)
    ii, jj = np.ogrid[i0:i1, j0:j1]
    mask = (ii - ri) ** 2 + (jj - rj) ** 2 <= r * r
    window = grid[i0:i1, j0:j1]
    window[mask] = world[i0:i1, j0:j1][mask]
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="mid_level_planner.png")
    ap.add_argument("--crop", type=float, default=5.0, help="Sensing radius, m (-1 = see all)")
    ap.add_argument("--lookahead", type=int, default=50, help="Planner lookahead, grid cells")
    ap.add_argument("--step", type=float, default=0.6, help="Distance advanced per replan, m")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="Run a sensing-radius sweep instead of one run, and skip the plot")
    args = ap.parse_args()

    if args.sweep:
        print(f"{'crop (m)':>9} {'outcome':>10} {'replans':>8} {'length':>8} {'max |y|':>8}")
        for crop in (-1, 8.0, 5.0, 4.0, 3.0, 2.0):
            traj, failures = simulate(crop, args.lookahead, args.step, args.iters)
            reached = np.linalg.norm(traj[-1] - np.array(GOAL_XY)) < 0.6
            length = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
            outcome = "reached" if reached else ("A* failed" if failures else "livelock")
            print(f"{crop:>9} {outcome:>10} {len(traj) - 1:>8} {length:>7.1f}m "
                  f"{np.abs(traj[:, 1]).max():>7.2f}m")
        return

    traj, failures = simulate(args.crop, args.lookahead, args.step, args.iters, args.verbose)
    reached = np.linalg.norm(traj[-1] - np.array(GOAL_XY)) < 0.6
    crossed = traj[:, 0].max() > 0.5  # got past the wall at x = 0
    print(f"replans:            {len(traj) - 1}")
    print(f"A* failures:        {failures}")
    print(f"final position:     {traj[-1]}")
    print(f"reached goal:       {reached}")
    print(f"got past the wall:  {crossed}")
    print(f"max |y| (detour):   {np.abs(traj[:, 1]).max():.2f} m")
    print(f"path length:        {np.linalg.norm(np.diff(traj, axis=0), axis=1).sum():.2f} m "
          f"(straight line would be {np.linalg.norm(np.subtract(GOAL_XY, START_XY)):.2f} m)")
    render(traj, reached, args)


def simulate(crop, lookahead, step, iters, verbose=False):
    """Closed replan loop. Returns (trajectory Nx2, number of A* failures)."""
    world = build_world()
    feedback = Feedback(verbose)
    occ = OccupancyMap(
        feedback,
        inflate_radius_meters=0.2,  # occupancy_inflation_radius from spot_executor_node.yaml
        use_cost_map=True,
        safe_distance=1.0,
        nearest_obstacle_cost=5.0,
    )

    high_level = np.array([list(START_XY), list(GOAL_XY)])  # straight through the wall
    robot = np.array(START_XY, dtype=float)
    trajectory = [robot.copy()]
    failures = 0

    planner = None
    for it in range(iters):
        occ.set_grid(visible(world, robot, crop), RESOLUTION, homo(*ORIGIN_XY), homo(*robot), 0.0)
        if planner is None:
            planner = MidLevelPlanner(occ, feedback, lookahead_distance_grid=lookahead)

        ok, out = planner.plan_path(high_level)
        if not ok:
            failures += 1
            break

        wp = np.asarray(out.path_waypoints_metric).reshape(-1, 2)
        if len(wp) < 2:
            break

        # Advance `step` along the returned path, the way the follower's lookahead does.
        seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        travelled, idx = 0.0, 1
        while idx < len(wp) and travelled + seg[idx - 1] < step:
            travelled += seg[idx - 1]
            idx += 1
        robot = wp[min(idx, len(wp) - 1)].astype(float)
        trajectory.append(robot.copy())

        if np.linalg.norm(robot - np.array(GOAL_XY)) < 0.6:
            break

    return np.array(trajectory), failures


def render(traj, reached, args):
    world = build_world()
    extent = [ORIGIN_XY[0], ORIGIN_XY[0] + WIDTH * RESOLUTION,
              ORIGIN_XY[1], ORIGIN_XY[1] + HEIGHT * RESOLUTION]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(world, origin="lower", extent=extent, cmap="Greys", vmin=-1, vmax=100)
    ax.plot(*zip(START_XY, GOAL_XY), "--", color="tab:orange", lw=2, label="high-level path")
    ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:blue", lw=2.5, label="executed (replanned)")
    ax.plot(*START_XY, "o", color="tab:green", ms=11, label="start")
    ax.plot(*GOAL_XY, "*", color="tab:red", ms=17, label="goal")
    ax.set_title(f"MidLevelPlanner closed loop — sensing {args.crop} m, reached={reached}")
    ax.set_xlabel("x (m, odom)")
    ax.set_ylabel("y (m, odom)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
