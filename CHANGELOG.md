# Changelog

## Unreleased

Every skewed-cell wrap rounds half away from zero without the libm
call: add the largest double below one half with the sign of the
argument and truncate with `cvttsd2si`, the same integer for every
double. `dist2`, `displacement`, and the Smith-hit Euclidean query
inline into the caller. Orthorhombic `dist2` squares
`min(|d|, L - |d|)`, the signed wrap's square bit for bit. A diagonal
H builds with three reciprocals and no square roots; a restricted H
takes its inverse from the lamda reciprocals, and its c-face width is
`|c_z|`. The hand-placed lattice loads are gone.
The Euclidean far query rounds in three Selling vectors (Babai) and
runs the Sommer-Feder-Shalvi slicer over the seven Voronoi-relevant
classes of the obtuse superbasis, in place of three McKilliam rounds.
A restricted cell with reduced tilts starts the slicer from the engine
wrap. `dist2_many`, `dist2_pairs`, and `wrap_many`, and their C
entries, are one fused AVX pass over the packed rows for every cell
shape; a squared distance equals the per-pair call bit for bit, and a
wrapped vector equals it up to the sign of a zero. A C call
looks up its cell once and borrows it in place.
`Cell::fixed` stores positions as 64-bit fixed-point fractions of
the cell. `dist2_fixed`, `displacement_fixed`, `dist2_many_fixed`,
`dist2_pairs_fixed`, and `fixed_many`, with C entries `mi_fixed_many`,
`mi_dist2_fixed`, `mi_dist2_many_fixed`, and `mi_dist2_pairs_fixed`,
wrap with integer subtraction modulo `2^64`, exact, then one product
with H for any cell shape (the Ozaki integer split).
Batch kernels run eight lanes wide where the processor has AVX-512F
and DQ and the compiler has the intrinsics (Rust 1.89; `build.rs`
checks, older compilers keep AVX2): 1.5 to 1.7 times faster for
triclinic and fixed-point batches, 1.2 to 1.5 for orthorhombic
distances, bit for bit the same results. A short group loads and
stores under a lane mask, so no batch ends in a scalar tail.
Python batch methods read any DLPack producer (numpy, PyTorch, JAX,
or CuPy host arrays) in place, with no `tolist`, and return numpy
arrays that own their buffers through a DLPack 1.0 capsule; a list in
still gives a list out. `dist2_pairs`, `fixed`, `fixed_many`,
`dist2_fixed`, `dist2_many_fixed`, and `dist2_pairs_fixed` are bound.
`minimage-burn` (in `burn/`, outside the workspace, Rust 1.95) puts
the wrap on Burn tensors, whose device picks the backend at run time:
CPU, wgpu, Vulkan, Metal, CUDA, or ROCm. Fractions folded on the CPU in
double precision wrap exactly on a single- or half-precision device,
since `ds - round(ds)` is exact for `|ds| < 1`.

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
