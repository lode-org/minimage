//! Minimum-image convention for a periodic parallelepiped.
//!
//! [`Cell`] holds the 3x3 lattice H and a dump-cell origin. Constructors
//! accept a LAMMPS bound box plus tilts, a CON lattice or
//! length-angle box, an ASE-style 3x3 cell, and a vesin box. Distances
//! are the fractional wrap `ds = wrap(Hinv (q - p))`, then `dr = H ds`.
//! That is the LAMMPS lamda / eOn / HOOMD engine convention.
//! Orthorhombic boxes use `d - L floor(d/L + 1/2)` on every image.
//! A restricted triclinic box uses the triangular lamda step.
//! [`Cell::reduce_tilts`] is GROMACS `correct_box`;
//! [`Cell::to_restricted`] is the LAMMPS general-to-restricted rotation.
//! [`Cell::displacement_euclidean`] is Smith's half-altitude test, then
//! Babai's rounding and the Sommer–Feder–Shalvi slicer on the Selling
//! superbasis cached on the calling thread. Lagrange size reduction runs before
//! that Delone step. [`dist2_many`], [`dist2_pairs`], [`wrap_many`], and
//! [`dist2_ortho_diffs`] batch the engine wrap. The orthorhombic SoA
//! kernel is AVX when the CPU has it. [`Cell::fixed`] stores a position
//! as 64-bit fixed-point fractions, and the wrap between two of them is
//! exact integer arithmetic ([`Cell::dist2_fixed`], [`dist2_many_fixed`],
//! [`dist2_pairs_fixed`]).
//! [`Cell::dist2_shifted_indexed`] gathers a linked-cell bin's index
//! list into that shifted kernel. [`reduce_pairs`] turns
//! a vesin image pair list into one minimum-image pair and drops the
//! self image.
//!
//! It is a LODE library. The Rust crate is the implementation. The C
//! ABI (`mi_*`) is the hourglass waist, the same shape as
//! [linkcell](https://github.com/d-SEAMS/linkcell) and
//! [readcon-core](https://github.com/lode-org/readcon-core). C++ is a
//! RAII header over that ABI.
//!
//! ```
//! use minimage::Cell;
//!
//! # fn main() -> Result<(), minimage::Error> {
//! let sim = Cell::ortho(10.0, 10.0, 10.0)?;
//! let left = [0.2, 0.0, 0.0];
//! let right = [9.4, 0.0, 0.0];
//! assert!((sim.dist2(left, right) - 0.64).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

mod batch;
mod cell;
mod error;
mod fixed;
mod fused;
mod kernel;
mod minkowski;
mod pairs;
mod selling;
mod simd;

pub use batch::{
    dist2_many, dist2_many_fixed, dist2_ortho_diffs, dist2_pairs, dist2_pairs_fixed, wrap_many,
};
pub use cell::{dump_bounds_to_h, Cell};
pub use error::Error;
pub use minkowski::{is_minkowski_reduced, minkowski_reduce};
pub use pairs::{reduce_pairs, reduce_pairs_packed};

#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "capi")]
pub use capi::*;
