import math

import numpy as np
import pytest

from minimage import PeriodicSpace


def test_scalar_wrap_distance_and_half_open_tie():
    metric = PeriodicSpace([360.0, None, 2.0 * math.pi])

    assert metric.dimensions == 3
    assert metric.periods == [360.0, None, 2.0 * math.pi]
    assert metric.wrap_point([-10.0, -10.0, 2.0 * math.pi]) == [
        350.0,
        -10.0,
        0.0,
    ]
    assert metric.wrap_difference([180.0, 4.0, math.pi]) == [
        -180.0,
        4.0,
        -math.pi,
    ]
    assert metric.dist2([350.0, 1.0, 0.0], [10.0, 4.0, 0.0]) == 409.0
    assert metric.distance([350.0, 1.0, 0.0], [10.0, 4.0, 0.0]) == math.sqrt(
        409.0
    )


def test_batches_and_box_bound():
    metric = PeriodicSpace([10.0, None])

    assert metric.wrap_points([[11.0, 2.0], [-1.0, 3.0]]) == [
        [1.0, 2.0],
        [9.0, 3.0],
    ]
    assert metric.wrap_differences([[6.0, 2.0], [-5.0, 3.0]]) == [
        [-4.0, 2.0],
        [-5.0, 3.0],
    ]
    assert metric.dist2_many([9.5, 0.0], [[0.5, 2.0], [5.0, 0.0]]) == [
        5.0,
        20.25,
    ]
    left = [[9.0, 1.0], [2.0, -1.0]]
    right = [[1.0, 4.0], [8.0, 3.0]]
    assert metric.dist2_pairs(left, right) == [13.0, 32.0]
    assert metric.dist2_pairs(
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
    ) == [13.0, 32.0]
    assert metric.axis_min_dist2(0, 9.5, 0.0, 1.0) == 0.25
    assert metric.aabb_min_dist2([9.5, 4.0], [0.0, 1.0], [1.0, 3.0]) == 1.25


def test_invalid_periods_shapes_and_bounds():
    for periods in ([], [0.0], [-1.0], [math.nan]):
        with pytest.raises(ValueError):
            PeriodicSpace(periods)

    metric = PeriodicSpace([10.0, None])
    with pytest.raises(ValueError):
        metric.dist2([0.0], [1.0])
    with pytest.raises(ValueError):
        metric.dist2_many([0.0, 0.0], [[1.0]])
    with pytest.raises(ValueError):
        metric.dist2_pairs([[0.0, 0.0]], [])
    with pytest.raises(ValueError):
        metric.aabb_min_dist2([0.0, 0.0], [2.0, 0.0], [1.0, 1.0])
