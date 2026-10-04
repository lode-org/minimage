# Changelog

## Unreleased

Orthorhombic wrap is `d - L floor(d/L + 1/2)` for every image, and
keeps the `-L/2` tie. Restricted triclinic engine wrap is the
triangular lamda step. The structure-of-arrays orthorhombic kernel and the shifted bin
kernel run AVX when the CPU has it.
`displacement_euclidean` is Smith's half-altitude test, then the
McKilliam–Grant–Clarkson closest point on the Selling superbasis
cached at construction. Lagrange size reduction runs before the
Delone step, so a near-parallel cell does not take one iteration
per reciprocal of the angle. C `mi_dist2` / `mi_displacement` /
`mi_wrap_many` do not rebuild that superbasis. Orthorhombic
`mi_dist2`, `mi_displacement`, `mi_wrap_many`, and `mi_dist2_many`
skip `Hinv` and use the per-axis wrap; the batch feeds the AVX kernel. `dist2_shifted_many`
and `mi_dist2_shifted_many` are Rapaport's one-shift bin pair for
linkcell. Constructors that only publish `mi_cell` skip Selling. A repeated
C call on the same twelve doubles reuses that inverse.
SymPy, Sollya, and Lean (mathlib) each check the wrap and the Delone
step; `ljos-consensus` settles those ballots.
`scripts/profile-pop3.sh` times the hot path on the POP3 hierarchy.

## 0.1.2 - 2026-09-27

Python constructors and `dist2` / `displacement` accept numpy arrays
(`tolist` fallback). `wrap` / `wrap_many` batch difference vectors.
Orthorhombic `wrap_many` uses the signed `[-L/2, L/2)` kernel, not a
per-row `displacement` call.
`is_restricted` / `tilts_reduced` / `reduce_tilts` / `to_restricted`
follow GROMACS `correct_box` and LAMMPS general-to-restricted.
`displacement_euclidean` uses the Smith 1989 half-edge test, then
Minkowski reduction (Nguyen-Stehle 2009) plus a 27-image: that is
the nearest image for cutoff-free k-NN, where a hex-prism body
diagonal is a fractional wrap that is not nearest.
`displacement_cartesian` is the 27-image check on the caller's H.
Agreement tests cover LAMMPS `minimum_image`, HOOMD `minImage`,
GROMACS `pbc_dx`, and eOn.

## 0.1.1

Orthorhombic signed wrap keeps `-L/2` and maps `+L/2` onto `-L/2`,
matching dump `relDist`.

## 0.1.0

First release. `Cell` holds H and a dump-cell origin. Constructors
accept LAMMPS bounds plus `xy, xz, yz` tilts, a CON lattice or
length-angle box, an ASE-style 3x3 cell, and a vesin box.
Fractional minimum-image displacement and squared distance, batched
ortho and general pair lists, and reduction of a vesin image pair
list to one minimum-image pair with the self image excluded. C ABI
(`mi_*`), C++ header, Meson and CMake consumers, thin Python module.
