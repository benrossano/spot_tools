import numpy as np
import pytest
from helpers import FREE, OCC, RES, UNK, homo, make_map


@pytest.mark.parametrize("yaw", [0.0, 0.7, np.pi / 2, -2.0])
def test_cell_centres_map_back_to_their_cell_under_rotated_origin(yaw):
    origin = homo(3.0, -2.0, yaw)
    om = make_map(np.zeros((40, 60)), origin=origin)
    rng = np.random.default_rng(0)
    for i, j in zip(rng.integers(0, 40, 50), rng.integers(0, 60, 50)):
        corner = om.grid_cell_to_global_pose((int(i), int(j)))
        assert np.allclose(
            corner, origin @ np.array([j * RES, i * RES, 0, 1]).reshape(4, 1)
        )
        centre = origin @ np.array([(j + 0.5) * RES, (i + 0.5) * RES, 0, 1]).reshape(
            4, 1
        )
        assert om.global_position_to_grid_cell(centre) == (i, j)


@pytest.mark.xfail(
    strict=True,
    reason="int() truncation: cell->corner->cell is off by one for some cells",
)
def test_cell_corner_roundtrip_is_idempotent():
    om = make_map(np.zeros((10, 200)))
    for j in range(200):
        assert om.global_position_to_grid_cell(om.grid_cell_to_global_pose((0, j))) == (
            0,
            j,
        )


def test_inflation_dilates_free_cells_only_and_keeps_unknown():
    grid = np.zeros((21, 21), np.int8)
    grid[10, 10] = OCC
    grid[:, :3] = UNK
    om = make_map(grid, inflate=0.25)  # ceil(0.25 / 0.1) = 3 cells
    g = om.get_grid()
    ii, jj = np.ogrid[:21, :21]
    d2 = (ii - 10) ** 2 + (jj - 10) ** 2
    assert (g[d2 <= 9] == OCC).all()
    assert (g[(d2 > 9) & (grid == FREE)] == FREE).all()
    assert (g[:, :3] == UNK).all()


def test_zero_inflation_returns_grid_unchanged():
    grid = np.zeros((5, 5), np.int8)
    grid[2, 2] = OCC
    assert (make_map(grid, inflate=0.0).get_grid() == grid).all()


def test_cost_map_decays_to_zero_beyond_safe_distance():
    grid = np.zeros((30, 30), np.int8)
    grid[15, 15] = OCC
    om = make_map(grid, cost=True, safe_distance=0.5, nearest_obstacle_cost=5.0)
    c = om.proximity_cost_map
    sigma = (0.5 / RES) / 3.0
    assert c[15, 15] == pytest.approx(5.0)
    assert c[15, 16] == pytest.approx(5.0 * np.exp(-1 / (2 * sigma**2)), rel=1e-5)
    assert c[15, 21] == 0.0  # 6 cells > 5-cell safe distance
    assert c[0, 0] == 0.0


def test_cost_map_treats_unknown_like_an_obstacle():
    """Characterisation: with use_cost_map, stepping into UNKNOWN costs the full obstacle penalty."""
    grid = np.zeros((30, 30), np.int8)
    grid[:, 25:] = UNK
    c = make_map(
        grid, cost=True, safe_distance=0.5, nearest_obstacle_cost=5.0
    ).proximity_cost_map
    assert np.allclose(c[:, 25:], 5.0)
    assert c[15, 24] > 0.0 and c[15, 19] == 0.0
