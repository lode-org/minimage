# minimage

<p align="center">
  <img src="assets/branding/minimage-logo-light.svg" width="360" alt="minimage">
</p>

**Minimum-image convention** for a periodic parallelepiped.

linkcell walks k-nearest neighbours. vesin builds cutoff pair lists.
Both need the same wrap: fold through H, pick one image, never the
self image. This crate is that wrap. `Cell` holds the 3x3 lattice and
a dump-cell origin. Constructors accept LAMMPS bounds plus `xy, xz,
yz` tilts, a CON lattice or length-angle box, an ASE-style 3x3 cell,
and a vesin box. `PartialCell` handles slabs, wires, and open systems
where only some axes are periodic.

It is a LODE library. The Rust crate is the implementation. The C ABI
(`minimage_*`) is the hourglass waist, the same shape as
[linkcell](https://github.com/d-SEAMS/linkcell) and
[readcon-core](https://github.com/lode-org/readcon-core). C++ is a
header-only wrapper over that ABI.

Docs (orgmode first, HTML export):
[lode-org.github.io/minimage](https://lode-org.github.io/minimage/).
Org catalog: [lode-org.github.io](https://lode-org.github.io/).
Rust API docs: [docs.rs/minimage](https://docs.rs/minimage).

## Install

Rust:

```
cargo add minimage
```

C and C++ consumers take the `staticlib` (`--features capi`, on by
default) plus `include/minimage.h` or `include/minimage.hpp`. Meson,
CMake, and pkg-config all install that archive, the shared library, and
those headers.
The function prefix spells out the crate name (`minimage_*`) so it
cannot collide with mimalloc's `mi_*` ABI. The non-default
`capi-compat-names` feature adds deprecated thin aliases for the old
`mi_*` function names while consumers migrate; those aliases can still
collide with mimalloc.

```
pip install minimage
```

Python 3.10 or newer. Each release has one abi3 wheel per platform,
for Linux x86_64 and aarch64 (manylinux_2_28), macOS arm64 and
Windows x86_64, which covers every CPython from 3.10 on. Elsewhere pip
builds the sdist, which needs a Rust toolchain.

```python
import minimage

cell = minimage.Cell.ortho(10.0, 10.0, 10.0)
assert abs(cell.dist2([0.2, 0.0, 0.0], [9.4, 0.0, 0.0]) - 0.64) < 1e-12
```

### Meson

```
meson setup build
meson compile -C build
meson install -C build
```

As a wrap, Meson exposes `minimage_dep`:

```
[wrap-git]
url = https://github.com/lode-org/minimage.git
revision = v0.2.3
depth = 1

[provide]
minimage = minimage_dep
```

```meson
minimage_dep = dependency('minimage', fallback: ['minimage', 'minimage_dep'])
```

### CMake

```
cmake -B build -DCMAKE_INSTALL_PREFIX=$PREFIX
cmake --build build
cmake --install build
```

```cmake
find_package(minimage 0.2 REQUIRED)
target_link_libraries(app PRIVATE minimage::minimage)
```

### pkg-config

```
pkg-config --cflags --libs minimage
```

## Rust

```rust
use minimage::{dist2_many, reduce_pairs, Cell, Error, PartialCell};

fn main() -> Result<(), Error> {
    let sim = Cell::ortho(10.0, 10.0, 10.0)?;
    let sheared = Cell::from_lammps_bounds(
        15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )?;
    let left = [0.2, 0.0, 0.0];
    let right = [9.4, 0.0, 0.0];
    assert!((sim.dist2(left, right) - 0.64).abs() < 1e-12);

    // Hex-prism body diagonal: Euclidean MIC finds a shorter image than the engine wrap.
    let q_diag = sheared.cartesian([0.49, 0.49, 0.49]);
    let euc = sheared.displacement_euclidean([0.0; 3], q_diag);
    assert!(euc.iter().map(|x| x * x).sum::<f64>() < sheared.dist2([0.0; 3], q_diag));

    let qs = [[9.4, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut out = [0.0; 2];
    dist2_many(&sim, left, &qs, &mut out)?;

    let slab = PartialCell::from_ase(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]],
        [true, true, false],
    )?;
    assert!((slab.displacement_euclidean([0.5, 0.5, 0.0], [9.5, 0.5, 20.0])[2] - 20.0).abs() < 1e-12);

    let pairs = [[0, 1], [0, 0], [0, 1]];
    let mut kept = Vec::new();
    reduce_pairs(&pairs, &mut kept)?;
    assert_eq!(kept, vec![[0, 1]]);
    Ok(())
}
```

## C ABI (`minimage_*`)

All exported C ABI functions use the `minimage_` prefix, including
`minimage_cell_from_lammps_bounds`, `minimage_displacement`,
`minimage_displacement_euclidean`, `minimage_dist2`, `minimage_dist2_many`,
`minimage_dist2_pairs`, `minimage_wrap_many`, `minimage_pbc_cell_from_ase`,
`minimage_displacement_euclidean_pbc`, `minimage_filter_shifted`,
`minimage_ring_centroid`, and `minimage_reduce_pairs`.

```c
#include "minimage.h"
#include <stdio.h>

mi_cell cell;
if (minimage_cell_from_lammps_bounds(
        15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, &cell) != 0) {
    fprintf(stderr, "%s\n", minimage_last_error());
}
const double p[3] = {0.0, 0.0, 0.0};
const double q[3] = {
    0.49 * (cell.ax + cell.bx + cell.cx),
    0.49 * (cell.ay + cell.by + cell.cy),
    0.49 * (cell.az + cell.bz + cell.cz),
};
double dr[3] = {0.0};
if (minimage_displacement_euclidean(&cell, p, q, dr) != 0) {
    fprintf(stderr, "%s\n", minimage_last_error());
}
```

## Conventions and error behaviour

- **Units and storage**: coordinates, lattice vectors, and origins share
  one caller-chosen length unit; squared distances and cutoffs use that
  unit squared; `from_con_box` / `minimage_cell_from_con_box` take angles
  in degrees. `Cell::from_vectors` and `Cell::h` store `a, b, c` as the
  columns of `H` (`r = H s + origin`), while `from_ase`, `from_con`,
  `from_vesin`, and `PartialCell::from_ase` take row-major `[a, b, c]`.
- **Displacements and triclinic cells**: `displacement(p, q)` returns
  `wrap(q - p)`. Orthorhombic boxes wrap each axis into `[-L/2, L/2)`,
  keeping `-L/2` and mapping `+L/2` to `-L/2`. Restricted triclinic boxes
  (`a` along x, `b` in the xy plane) use the upper-triangular `lamda`
  wrap matching LAMMPS, HOOMD, OpenMM, and GROMACS; `reduce_tilts`
  applies GROMACS `correct_box`, `to_restricted` rotates general cells
  into that frame, and `displacement_euclidean` (`minimage_displacement_euclidean`)
  checks Smith's half-altitude bound and falls back to the Selling
  superbasis slicer for exact Voronoi nearest images.
- **N-dimensional features**: `PeriodicSpace` assigns a positive
  period or `None` to each independent feature axis. It wraps points into
  `[0, L)`, differences into `[-L/2, L/2)`, handles contiguous row-major
  batches, and computes periodic point-to-box lower bounds for tree
  pruning. It is separate from coupled triclinic lattice geometry.
- **Errors**: Rust returns `Result<_, Error>` (`BadBox`, `BufferSize`,
  `Empty`). C ABI functions return `0` on success and `1` on failure and
  record a thread-local message in `minimage_last_error()` (cleared on
  the next success on that thread; `minimage_version()` leaves it
  untouched). C++ throws `minimage::Error`, and Python raises
  `ValueError`, `TypeError`, or `BufferError`.

Batch kernels select AVX-512, AVX2, AVX, NEON, or scalar at run time.
`MINIMAGE_SIMD` overrides that choice. `simd_path()` returns the name.

## License

MIT. See [LICENSE](LICENSE).
