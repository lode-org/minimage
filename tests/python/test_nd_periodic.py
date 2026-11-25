import math
import sys

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
    assert metric.wrap_point([-1e-20, 2.0]) == [0.0, 2.0]
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
    listed = metric.dist2_pairs(left, right)
    assert listed.dtype == np.float64
    np.testing.assert_array_equal(listed, [13.0, 32.0])
    paired = metric.dist2_pairs(
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
    )
    assert paired.dtype == np.float64
    np.testing.assert_array_equal(paired, [13.0, 32.0])
    empty = metric.dist2_pairs(
        np.empty((0, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
    )
    assert empty.dtype == np.float64 and empty.shape == (0,)
    assert metric.axis_min_dist2(0, 9.5, 0.0, 1.0) == 0.25
    assert metric.aabb_min_dist2([9.5, 4.0], [0.0, 1.0], [1.0, 3.0]) == 1.25


def test_zero_dimension_space_reports_empty_input():
    with pytest.raises(ValueError, match="^empty input$"):
        PeriodicSpace([])


def test_invalid_periods_shapes_and_bounds():
    for periods in ([0.0], [-1.0], [math.nan]):
        with pytest.raises(ValueError, match="period"):
            PeriodicSpace(periods)

    metric = PeriodicSpace([10.0, None])
    with pytest.raises(ValueError):
        metric.dist2([0.0], [1.0])
    with pytest.raises(ValueError):
        metric.dist2_many([0.0, 0.0], [[1.0]])
    with pytest.raises(ValueError):
        metric.dist2_pairs([[0.0, 0.0]], [])
    with pytest.raises(ValueError):
        metric.dist2_pairs(np.empty((0, 3)), np.empty((0, 3)))
    with pytest.raises(ValueError):
        metric.aabb_min_dist2([0.0, 0.0], [2.0, 0.0], [1.0, 1.0])


def test_extreme_magnitudes_negatives_and_mixed_axes():
    big = sys.float_info.max
    metric = PeriodicSpace([10.0, None, 3.0])
    p = [-big, -4.5, big]
    q = [big, -1.5, -big]

    disp = metric.displacement(p, q)
    assert all(math.isfinite(x) for x in disp)
    assert -5.0 <= disp[0] < 5.0
    assert disp[1] == 3.0
    assert -1.5 <= disp[2] < 1.5

    d2 = metric.dist2(p, q)
    dist = metric.distance(p, q)
    assert math.isfinite(d2) and d2 >= 9.0
    assert dist == math.sqrt(d2)

    wp = metric.wrap_point(p)
    wq = metric.wrap_point(q)
    assert metric.displacement(wp, wq) == disp
    assert metric.dist2(wp, wq) == d2

    assert metric.dist2_many(p, [q, p]) == [d2, 0.0]
    np.testing.assert_array_equal(
        metric.dist2_pairs(np.asarray([p, q]), np.asarray([q, p])),
        [d2, d2],
    )

    assert metric.axis_min_dist2(0, big, -1.0, 1.0) == 1.0
    box = metric.aabb_min_dist2(p, [-1.0, -2.0, -1.0], [1.0, 0.0, 1.0])
    assert math.isfinite(box) and box >= 4.0


def test_huge_image_counts_keep_a_representable_separation():
    metric = PeriodicSpace([9.664251935223536e-9, None])
    a, b = 35602323.55526906, 35602323.55526907
    expected = -2.2136713382997084e-9

    assert metric.displacement([a, 1.0], [b, 3.0]) == [expected, 2.0]
    assert metric.dist2([a, 0.0], [b, 0.0]) == expected * expected
    assert metric.axis_min_dist2(0, b, a, a) == expected * expected
    assert metric.aabb_min_dist2([b, 0.0], [a, -1.0], [a, 1.0]) == expected * expected


def test_distances_use_the_exact_minimum_image():
    micro = PeriodicSpace([1e-6])
    far = 4.525188817411374e-16
    assert micro.displacement([-10.0], [0.0]) == [far]
    assert micro.dist2([-10.0], [0.0]) == far * far
    assert micro.axis_min_dist2(0, 0.0, -10.0, -10.0) == far * far

    two_30 = 2.0**30
    near = 4.76837158203125e-7
    assert micro.displacement([two_30], [two_30 + 4e-7]) == [near]

    ten = PeriodicSpace([10.0])
    two_54 = 2.0**54
    assert ten.displacement([two_54], [two_54 - 4.0]) == [-4.0]
    assert ten.dist2([two_54], [two_54 - 4.0]) == 16.0
    assert ten.axis_min_dist2(0, two_54, two_54 + 4.0, two_54 + 4.0) == 16.0


def test_property_extreme_and_negative_coordinates():
    rng = np.random.default_rng(0x50524F50)
    big = sys.float_info.max
    for dim in (1, 2, 4, 6):
        for case in range(40):
            periods = [
                None if (axis + case) % 3 == 0 else float(rng.uniform(0.25, 20.0))
                for axis in range(dim)
            ]
            metric = PeriodicSpace(periods)
            p = []
            q = []
            for axis, period in enumerate(periods):
                if period is None:
                    p.append(float(rng.uniform(-100.0, 100.0)))
                    q.append(float(rng.uniform(-100.0, 100.0)))
                else:
                    choices = [big, -big, float(-rng.uniform(0.0, 1e6 * period)), float(rng.uniform(-10.0, 10.0))]
                    p.append(choices[(case + axis) % len(choices)])
                    q.append(choices[(case + axis + 1) % len(choices)])
            disp = metric.displacement(p, q)
            d2 = metric.dist2(p, q)
            assert math.isfinite(d2) and d2 >= 0.0
            for axis, period in enumerate(periods):
                if period is None:
                    assert disp[axis] == q[axis] - p[axis]
                else:
                    assert -0.5 * period <= disp[axis] < 0.5 * period
            assert math.isclose(d2, sum(x * x for x in disp), rel_tol=1e-12, abs_tol=1e-12)
