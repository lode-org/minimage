# Changelog

<!-- towncrier release notes start -->

## 0.2.1 - 2026-11-16

### Added

Release wheels cover Linux x86_64 and aarch64 (manylinux_2_28), macOS arm64 and Windows x86_64, each tested with pytest before upload, and the sdist is installed and tested too.

### Changed

The crates.io job runs `cargo publish --locked` without `--no-verify`, so the packaged crate builds before it uploads. Every workflow pins its actions by commit, and Dependabot proposes one grouped update a month for the actions and one for `Cargo.lock`. CI audits the release library's instructions against `isa-allow.toml` with rgpot's ISA audit on the x86-64 baseline; each dispatched SIMD kernel has an allow entry with its reason.

### Fixed

CITATION.cff gives 2026-11-15, the 0.2.0 release date. `sync-version.py --release` now sets date-released on the release commit.

The Python module builds against the stable ABI for CPython 3.10 and newer. `pip install minimage` from the 0.2.0 sdist failed on Python 3.14, which PyO3 0.23 rejects outside the stable ABI. The `test` extra now names numpy, which three of the four Python test files import.

The README asks CMake for `find_package(minimage 0.2)`. The package config accepts the same minor version only, so the 0.1 line it showed failed against an installed 0.2.0. The version sync and check now cover that line.

## 0.2.0 - 2026-11-15

### Added

`PartialCell` is periodic along some axes only: a slab periodic in two
directions, a wire in one, or an open system. The Euclidean nearest
image is searched over the lattice of the periodic vectors, with no
image along an open axis; a half-box tie keeps the engine wrap. With all
three axes periodic it is a `Cell` and returns the same bits. The C ABI
adds `mi_pbc_cell`, `mi_pbc_cell_from_ase`,
`mi_displacement_euclidean_pbc` and `mi_dist2_euclidean_pbc`; the
`mi_cell` functions and layout are unchanged.

`filter_shifted` keeps candidate rows whose shifted squared distance is
inside a cutoff. `ring_centroid` averages the beads of a ring polymer
after unwrapping each bead onto bead 0. The C ABI exposes both as
`mi_filter_shifted` and `mi_ring_centroid`.

Batch kernels choose one SIMD path per call: AVX-512F and DQ, AVX2,
AVX, NEON, or scalar. `MINIMAGE_SIMD` overrides the choice. NEON runs
two lanes of the scalar arithmetic for `dist2_many`, `dist2_pairs`,
`wrap_many`, `dist2_ortho_diffs`, and the fixed and fixed32 batches.

### Changed

The version is written by hand only in `[workspace.package]`.
Meson, CMake and maturin read it from `Cargo.toml` at build time.
`scripts/sync-version.py` regenerates the other copies.
The four Meson wrap snippets name the newest release tag, which `[workspace.metadata.release].tag` in `Cargo.toml` sets.
`cog bump` writes the version and that tag, then runs the sync and towncrier.
It checks the result before it tags the release commit.

`floor_fast` and `round_away` now use the x86_64 truncation on every
architecture. `floor_fast(-0.0)` is `+0.0`, so the fractional part of
`-0.0` is `-0.0`. On aarch64 this used to follow `f64::floor` and
`f64::round`, which keep the sign of zero. x86_64 results of these
two functions are unchanged.

## 0.1.4 - 2026-10-09

`fixed` and `fixed32` store a position as a fraction of the cell and
wrap by integer subtraction. `dist2_euclidean_many`,
`dist2_euclidean_pairs`, and `dist2_euclidean_pairs_warm` match
`dist2_euclidean` per pair, on the C, C++, and Python APIs.
Python batches read a DLPack array in place and return a numpy array.
`minimage-burn` is a separate crate and is not in the default build.

## 0.1.3 - 2026-10-04

Orthorhombic wrap is `d - L floor(d/L + 1/2)` for every image, and
keeps the `-L/2` tie. Restricted triclinic engine wrap is the
triangular lamda step. The structure-of-arrays orthorhombic kernel and the shifted bin
kernel run AVX when the CPU has it.
`displacement_euclidean` is Smith's half-altitude test, then the
McKilliam-Grant-Clarkson closest point on the Selling superbasis.
The superbasis is cached on the calling thread, keyed by H, and
built on the first query that fails the Smith test. Constructors
do not build it. A restricted cell keeps the three lamda reciprocals.
The ortho and restricted tests scale by the face widths, which
construction already computes, and do not take a second set of
square roots. Lagrange size reduction runs before the
Delone step, so a near-parallel cell does not take one iteration
per reciprocal of the angle. C `mi_dist2` / `mi_displacement` /
`mi_wrap_many` do not build that superbasis. Orthorhombic
`mi_dist2`, `mi_displacement`, `mi_wrap_many`, and `mi_dist2_many`
skip `Hinv` and use the per-axis wrap; the batch feeds the AVX kernel. `dist2_shifted_many`
and `mi_dist2_shifted_many` are Rapaport's one-shift bin pair for
linkcell. Constructors that only publish `mi_cell` skip Selling. A repeated
C call on the same twelve doubles reuses that inverse.
Orthorhombic `dist2` uses two comparisons inside one neighbouring
image and the floor form past that image. `dist2_pairs` and `mi_dist2_pairs` use that orthorhombic
structure-of-arrays kernel, and a repeated orthorhombic `mi_cell`
reuses the lengths and the reciprocals. `dist2_shifted_indexed`
gathers a linked-cell index list into the shifted kernel. The
gather was slower than the inlined subtract on a long index list,
so the production walk keeps that subtract.

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
