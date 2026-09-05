import numpy as np

import minimage


def test_from_vesin_accepts_numpy_rows():
    box = np.eye(3) * 10.0
    cell = minimage.Cell.from_vesin(box)
    assert cell.is_ortho()
    assert abs(cell.dist2(np.array([0.2, 0.0, 0.0]), np.array([9.4, 0.0, 0.0])) - 0.64) < 1e-12


def test_wrap_many_numpy_diffs():
    cell = minimage.Cell.ortho(10.0, 10.0, 10.0)
    diffs = np.array([[9.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    out = cell.wrap_many(diffs)
    assert abs(out[0][0] + 0.8) < 1e-12
    assert abs(out[1][0] - 0.2) < 1e-12


def test_dist2_many_numpy_qs():
    cell = minimage.Cell.ortho(10.0, 10.0, 10.0)
    qs = np.array([[9.4, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = cell.dist2_many(np.array([0.2, 0.0, 0.0]), qs)
    assert abs(out[0] - 0.64) < 1e-12
    assert abs(out[1] - 0.64) < 1e-12
