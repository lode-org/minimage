//! Minimum-image convention for a periodic parallelepiped.
//!
//! [`Cell`] holds the 3x3 lattice H and a dump-cell origin. Constructors
//! accept a LAMMPS bound box plus tilts, a CON lattice or
//! length-angle box, an ASE-style 3x3 cell, and a vesin box. Distances
//! are the fractional wrap `ds = wrap(Hinv (q - p))`, then `dr = H ds`.
//! That is the LAMMPS lamda / eOn / HOOMD engine convention and is the
//! Euclidean MIC on a restricted, tilt-reduced cell. [`Cell::reduce_tilts`]
//! is GROMACS `correct_box`; [`Cell::to_restricted`] is the LAMMPS
//! general-to-restricted rotation; [`Cell::displacement_euclidean`]
//! applies those once, then wraps. [`dist2_many`], [`wrap_many`], and
//! [`dist2_ortho_diffs`] batch the engine wrap. [`reduce_pairs`] turns a
//! vesin image pair list into one minimum-image pair and drops the self
//! image.
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
mod minkowski;
mod pairs;

pub use batch::{dist2_many, dist2_ortho_diffs, dist2_pairs, wrap_many};
pub use cell::{dump_bounds_to_h, Cell};
pub use error::Error;
pub use minkowski::{is_minkowski_reduced, minkowski_reduce};
pub use pairs::{reduce_pairs, reduce_pairs_packed};

#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "capi")]
pub use capi::*;
