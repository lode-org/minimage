# Changelog

## Unreleased

Python constructors and `dist2` / `displacement` accept numpy arrays
(`tolist` fallback). `wrap` / `wrap_many` batch difference vectors.
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
