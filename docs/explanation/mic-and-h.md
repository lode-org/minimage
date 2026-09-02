# Minimum image and H

A periodic dump is a parallelepiped whose edges are the columns of H.
Cartesian coordinates fold to fractional coordinates `s = Hinv (r -
origin)`, wrap each component into the central image, then map back
`r = H s`. Squared distance is `|H wrap(Hinv (q - p))|^2`.

Orthorhombic boxes are the diagonal case. Each axis wraps on its own
length. The batched kernel hoists one reciprocal per axis and wraps
precomputed differences, the same arithmetic as a Highway
`BatchPeriodicDistSq` loop.

LAMMPS dump bounds are not H. The ITEM BOX BOUNDS line stores bound
spans plus tilt `xy, xz, yz`. `Cell::from_lammps_bounds` recovers the
restricted-triclinic H and origin:

```
xlo_bound = xlo + min(0, xy, xz, xy+xz)
```

A CON header may already hold the 3x3 lattice, or lengths and angles
in the crystallographic convention. ASE and vesin pass rows `(a, b,
c)`.

vesin enumerates periodic images. A particle can appear as its own
neighbour through an image, and one neighbour can arrive through
several images. `reduce_pairs` keeps each ordered pair once and drops
the self image.
