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
and a vesin box.

It is a LODE library. The Rust crate is the implementation. The C ABI
(`mi_*`) is the hourglass waist, the same shape as
[linkcell](https://github.com/d-SEAMS/linkcell) and
[readcon-core](https://github.com/lode-org/readcon-core). C++ is a
RAII header over that ABI.

## Install

Rust:

```
cargo add minimage
```

C and C++ consumers take the `staticlib` (`--features capi`, on by
default) plus `include/minimage.h` or `include/minimage.hpp`. Meson,
CMake, and pkg-config all install that archive and those headers.

```
pip install minimage
```

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
revision = v0.1.1
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
find_package(minimage 0.1 REQUIRED)
target_link_libraries(app PRIVATE minimage::minimage)
```

### pkg-config

```
pkg-config --cflags --libs minimage
```

## Rust

```rust
use minimage::{dist2_many, reduce_pairs, Cell};

let sim = Cell::ortho(10.0, 10.0, 10.0)?;
let sheared = Cell::from_lammps_bounds(
    15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)?;
let left = [0.2, 0.0, 0.0];
let right = [9.4, 0.0, 0.0];
assert!((sim.dist2(left, right) - 0.64).abs() < 1e-12);

let qs = [[9.4, 0.0, 0.0], [1.0, 0.0, 0.0]];
let mut out = [0.0; 2];
dist2_many(&sim, left, &qs, &mut out)?;

let pairs = [[0, 1], [0, 0], [0, 1]];
let mut kept = Vec::new();
reduce_pairs(&pairs, &mut kept)?;
assert_eq!(kept, vec![[0, 1]]);
```

`Cell::from_ase`, `Cell::from_con`, `Cell::from_con_box`, and
`Cell::from_vesin` take the same row-major 3x3 lattice. Displacement
is the wrapped vector from `p` to `q`.

## License

MIT. See [LICENSE](LICENSE).
