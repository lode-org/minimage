import minimage


def test_ortho_wrap():
    cell = minimage.Cell.ortho(10.0, 10.0, 10.0)
    assert cell.is_ortho()
    assert abs(cell.dist2([0.2, 0.0, 0.0], [9.4, 0.0, 0.0]) - 0.64) < 1e-12


def test_sheared_lammps():
    cell = minimage.Cell.from_lammps_bounds(
        15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    assert not cell.is_ortho()
    assert abs(cell.dist2([0.2, 0.1, 1.0], [9.7, 0.1, 1.0]) - 0.25) < 1e-12


def test_reduce_pairs():
    kept = minimage.reduce_image_pairs([(0, 1), (0, 0), (0, 1), (2, 3)])
    assert kept == [(0, 1), (2, 3)]
