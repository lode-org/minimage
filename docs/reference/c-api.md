# C API

Prefix `mi_`. Every buffer is caller-owned. A nonzero return means
failure; read `mi_last_error()` on the same thread.

| Symbol | Role |
| --- | --- |
| `mi_cell` | Lattice vectors a, b, c and origin |
| `mi_cell_ortho` | Header-only diagonal box |
| `mi_cell_from_vectors` | Columns a, b, c |
| `mi_cell_from_lammps` | `xlo xhi ylo yhi zlo zhi` plus tilts |
| `mi_cell_from_lammps_bounds` | Dump bound spans plus tilts |
| `mi_cell_from_ase` | Row-major 3x3 |
| `mi_cell_from_con` | CON 3x3 |
| `mi_cell_from_con_box` | Lengths and angles in degrees |
| `mi_cell_from_vesin` | vesin 3x3 |
| `mi_displacement` | Wrapped `q - p` |
| `mi_dist2` | Squared minimum-image distance |
| `mi_dist2_many` | One source, packed candidates |
| `mi_dist2_pairs` | Packed pair lists |
| `mi_dist2_ortho_diffs` | Highway-shaped ortho wrap |
| `mi_reduce_pairs` | Drop self images, unique pairs |
| `mi_last_error` | Thread-local error string |
| `mi_version` | Process-static version |
