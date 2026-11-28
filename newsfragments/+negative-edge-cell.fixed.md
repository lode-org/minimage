A diagonal cell built with a negative edge, such as
`Cell::ortho(-15.0, -15.0, 15.0)` or the same rows through
`minimage_cell_from_vectors` and `minimage_cell_from_ase`, kept the signed
edges in `H` and its inverse while the orthorhombic wrap, the fold, and
`lattice_shift` counted in `|edge|`. `cartesian(fractional(r))` then
returned a point that was no image of `r`, `lattice_shift(1, 0, 0)` was
not `a()`, and `filter_shifted` applied a shift along the opposite vector
to the one `H` named. An orthorhombic cell now stores each lattice vector
with a positive diagonal entry, which is the same lattice, so every stored
quantity and the `mi_cell` the C constructors write back agree. Sheared
cells were already consistent for any sign or handedness and keep the
vectors they were given. A brute-force scan over lattice images now checks
both.
