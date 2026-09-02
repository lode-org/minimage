//! Minimum-image convention for a periodic parallelepiped.
//!
//! [`Cell`] holds the 3x3 lattice H and a dump-cell origin. Constructors
//! accept a LAMMPS bound box plus tilts, a CON lattice or
//! length-angle box, an ASE-style 3x3 cell, and a vesin box. Distances
//! are the fractional wrap `ds = wrap(Hinv (q - p))`, then `dr = H ds`.
//! [`dist2_many`] and [`dist2_ortho_diffs`] batch that wrap over
//! candidate lists. [`reduce_pairs`] turns a vesin image pair list into
//! one minimum-image pair and drops the self image.
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
mod pairs;

pub use batch::{dist2_many, dist2_ortho_diffs, dist2_pairs};
pub use cell::{dump_bounds_to_h, Cell};
pub use error::Error;
pub use pairs::{reduce_pairs, reduce_pairs_packed};

#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "capi")]
pub use capi::*;
