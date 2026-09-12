"""Closed-loop replanning against a partially observed synthetic wall (HANDOFF.md)."""

import numpy as np
import pytest
from helpers import RES, load_example_harness

h = load_example_harness()
REACHED = 0.6


@pytest.mark.slow
@pytest.mark.parametrize("crop", [-1, 8.0, 5.0])
def test_reaches_goal_when_sensing_covers_the_lookahead(crop):
    traj, failures = h.simulate(crop, 50, 0.6, 120)
    assert failures == 0
    assert np.linalg.norm(traj[-1] - np.array(h.GOAL_XY)) < REACHED


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True, reason="livelock when sensing radius < lookahead; see HANDOFF.md"
)
@pytest.mark.parametrize("crop", [4.0, 3.0, 2.0])
def test_reaches_goal_when_sensing_is_shorter_than_the_lookahead(crop):
    traj, failures = h.simulate(crop, 50, 0.6, 120)
    assert failures == 0
    assert np.linalg.norm(traj[-1] - np.array(h.GOAL_XY)) < REACHED


@pytest.mark.slow
def test_full_visibility_trajectory_never_enters_an_obstacle():
    world = h.build_world()
    traj, _ = h.simulate(-1, 50, 0.6, 120)
    for x, y in traj:
        i, j = h.world_to_cell(x + RES / 2, y + RES / 2)
        assert world[i, j] == h.FREE


COMMIT = {"path_commitment_weight": 1.0}


@pytest.mark.slow
@pytest.mark.parametrize("crop", [-1, 5.0, 4.0, 3.0])
def test_path_commitment_breaks_the_livelock(crop):
    traj, failures = h.simulate(crop, 50, 0.6, 120, planner_kwargs=COMMIT)
    assert failures == 0
    assert np.linalg.norm(traj[-1] - np.array(h.GOAL_XY)) < REACHED


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="2 m sensing vs 6 m lookahead still livelocks even with commitment",
)
def test_path_commitment_does_not_rescue_very_short_sensing():
    traj, failures = h.simulate(2.0, 50, 0.6, 120, planner_kwargs=COMMIT)
    assert np.linalg.norm(traj[-1] - np.array(h.GOAL_XY)) < REACHED
