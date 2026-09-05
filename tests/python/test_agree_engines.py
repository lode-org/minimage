"""minimage vs eOn, LAMMPS, and GROMACS wrap oracles."""

from __future__ import annotations

import numpy as np
import pytest

import minimage


def eon_numpy_wrap(diff, box):
    ibox = np.linalg.inv(box)
    frac = np.dot(diff, ibox)
    frac = (frac % 1.0 + 1.5) % 1.0 - 0.5
    return np.dot(frac, box)


def eon_cpp_wrap(diff, box):
    ibox = np.linalg.inv(box)
    frac = np.dot(np.asarray(diff, dtype=float), ibox)
    frac = frac - np.floor(frac + 0.5)
    return np.dot(frac, box)


def lammps_ortho_wrap(diff, lx, ly, lz):
    out = np.asarray(diff, dtype=float).copy()
    half = np.array([0.5 * lx, 0.5 * ly, 0.5 * lz])
    span = np.array([lx, ly, lz])
    for i in range(3):
        if abs(out[i]) > half[i]:
            out[i] -= np.copysign(span[i], out[i])
    return out


def lammps_triclinic_wrap(diff, xprd, yprd, zprd, xy, xz, yz):
    h = np.array([[xprd, xy, xz], [0.0, yprd, yz], [0.0, 0.0, zprd]], dtype=float)
    lamda = np.linalg.solve(h, np.asarray(diff, dtype=float))
    lamda = lamda - np.floor(lamda + 0.5)
    return h @ lamda


def gromacs_pbc_dx(x1, x2, box_vectors):
    dx = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    box = np.asarray(box_vectors, dtype=float)
    for d in (2, 1, 0):
        hbox = 0.5 * box[d, d]
        while dx[d] > hbox:
            dx = dx - box[d]
        while dx[d] < -hbox:
            dx = dx + box[d]
    return dx


def hoomd_minimage(diff, lx, ly, lz, xy=0.0, xz=0.0, yz=0.0):
    """HOOMD BoxDim::minImage on a restricted box (z, then y, then x)."""
    w = np.asarray(diff, dtype=float).copy()
    if w[2] >= 0.5 * lz:
        w[2] -= lz
        w[1] -= lz * yz
        w[0] -= lz * xz
    elif w[2] < -0.5 * lz:
        w[2] += lz
        w[1] += lz * yz
        w[0] += lz * xz
    if w[1] >= 0.5 * ly:
        w[1] -= ly
        w[0] -= ly * xy
    elif w[1] < -0.5 * ly:
        w[1] += ly
        w[0] += ly * xy
    if w[0] >= 0.5 * lx:
        w[0] -= lx
    elif w[0] < -0.5 * lx:
        w[0] += lx
    return w


def lammps_minimum_image_triclinic(diff, xprd, yprd, zprd, xy, xz, yz):
    """LAMMPS Domain::minimum_image triclinic walk (z, then y, then x)."""
    dx, dy, dz = [float(v) for v in diff]
    while abs(dz) > 0.5 * zprd:
        if dz < 0.0:
            dz += zprd
            dy += yz
            dx += xz
        else:
            dz -= zprd
            dy -= yz
            dx -= xz
    while abs(dy) > 0.5 * yprd:
        if dy < 0.0:
            dy += yprd
            dx += xy
        else:
            dy -= yprd
            dx -= xy
    while abs(dx) > 0.5 * xprd:
        if dx < 0.0:
            dx += xprd
        else:
            dx -= xprd
    return np.array([dx, dy, dz])


@pytest.mark.parametrize(
    "diff",
    [
        np.array([0.2, 0.0, 0.0]),
        np.array([9.2, 0.0, 0.0]),
        np.array([-9.2, 0.1, 0.0]),
        np.array([4.9, 5.4, -5.9]),
    ],
)
def test_ortho_matches_eon_lammps_gromacs(diff):
    box = np.diag([10.0, 11.0, 12.0])
    got = np.asarray(minimage.Cell.ortho(10.0, 11.0, 12.0).wrap(diff))
    np.testing.assert_allclose(got, eon_numpy_wrap(diff, box), atol=1e-12)
    np.testing.assert_allclose(got, eon_cpp_wrap(diff, box), atol=1e-12)
    np.testing.assert_allclose(got, lammps_ortho_wrap(diff, 10.0, 11.0, 12.0), atol=1e-12)
    np.testing.assert_allclose(got, gromacs_pbc_dx(diff, np.zeros(3), box), atol=1e-12)


def test_sheared_matches_lammps_triclinic_and_eon():
    rows = np.array([[10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0]])
    p = np.array([0.2, 0.1, 1.0])
    q = np.array([9.7, 0.1, 1.0])
    got = np.asarray(minimage.Cell.from_vesin(rows).displacement(p, q))
    lam = lammps_triclinic_wrap(q - p, 10.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0)
    np.testing.assert_allclose(got, lam, atol=1e-12)
    np.testing.assert_allclose(got, eon_cpp_wrap(q - p, rows), atol=1e-12)
    np.testing.assert_allclose(gromacs_pbc_dx(q, p, rows), eon_cpp_wrap(q - p, rows), atol=1e-12)


def test_restricted_cell_fractional_matches_27_image():
    cell = minimage.Cell.from_vesin(
        [[10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0]]
    )
    assert cell.fractional_matches_cartesian()


def test_moderate_skew_fractional_already_short():
    cell = minimage.Cell.from_vectors([1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0])
    frac = np.asarray(cell.displacement([0.0, 0.0, 0.0], [0.95, 0.05, 0.0]))
    cart = np.asarray(cell.displacement_cartesian([0.0, 0.0, 0.0], [0.95, 0.05, 0.0]))
    np.testing.assert_allclose(frac, cart, atol=1e-12)
    assert float(np.dot(cart, cart)) < 0.01


def test_sheared_matches_hoomd_and_lammps_walk():
    rows = np.array([[10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0]])
    p = np.array([0.2, 0.1, 1.0])
    q = np.array([9.7, 0.1, 1.0])
    got = np.asarray(minimage.Cell.from_vesin(rows).displacement(p, q))
    hoomd = hoomd_minimage(q - p, 10.0, 8.660254037844386, 10.0, xy=5.0)
    lam = lammps_minimum_image_triclinic(q - p, 10.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0)
    np.testing.assert_allclose(got, hoomd, atol=1e-12)
    np.testing.assert_allclose(got, lam, atol=1e-12)


def test_hex_body_diagonal_euclidean_beats_fractional():
    cell = minimage.Cell.from_vesin(
        [[10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0]]
    )
    assert cell.tilts_reduced()
    assert cell.is_minkowski_reduced()
    cart = (
        0.49 * np.array([10.0, 0.0, 0.0])
        + 0.49 * np.array([5.0, 8.660254037844386, 0.0])
        + 0.49 * np.array([0.0, 0.0, 10.0])
    )
    frac = np.asarray(cell.displacement([0.0, 0.0, 0.0], cart.tolist()))
    euc = np.asarray(cell.displacement_euclidean([0.0, 0.0, 0.0], cart.tolist()))
    assert float(np.dot(euc, euc)) + 1e-8 < float(np.dot(frac, frac))


def test_unreduced_tilt_euclidean_is_lattice_zero():
    cell = minimage.Cell.from_vectors([1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 0.0, 1.0])
    assert cell.is_restricted()
    assert not cell.tilts_reduced()
    q = [0.02, -0.02, 0.0]
    euc = np.asarray(cell.displacement_euclidean([0.0, 0.0, 0.0], q))
    cart = np.asarray(cell.displacement_cartesian([0.0, 0.0, 0.0], q))
    assert float(np.dot(euc, euc)) < 1e-24
    assert float(np.dot(cart, cart)) > 1e-8
    red = cell.reduce_tilts()
    assert red.tilts_reduced()
