# A wrapped pair

Two points near opposite faces of a 10-wide box sit 9.2 apart in
Cartesian space and 0.8 apart under the minimum image.

```rust
use minimage::Cell;

let cell = Cell::ortho(10.0, 10.0, 10.0)?;
assert!((cell.dist2([0.2, 0.0, 0.0], [9.4, 0.0, 0.0]) - 0.64).abs() < 1e-12);
```

A sheared LAMMPS dump uses bound spans, not `lx, ly, lz`. The a-image
pair `(0.2, 0.1, 1)` and `(9.7, 0.1, 1)` on

```
box = 15  8.660254037844386  10  5  0  0
```

has squared distance 0.25.
