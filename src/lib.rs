//! Minimum-image convention for a periodic parallelepiped.
//!
//! [`Cell`] holds the 3x3 lattice `H` and a dump-cell origin.
//! Constructors accept a LAMMPS bound box plus tilts, a CON lattice or
//! length-angle box, an ASE-style 3x3 cell, and a vesin box.
//!
//! # Units and conventions
//!
//! - **Length units**: arbitrary, provided `H`, `origin`, positions, and
//!   cutoffs share one unit (squared cutoffs use that unit squared).
//!   [`Cell::from_con_box`] takes crystallographic angles in degrees.
//!   Fractional coordinates from [`Cell::fractional`] are dimensionless
//!   in `[0, 1)`.
//! - **Matrix storage**: [`Cell::from_vectors`] and [`Cell::h`] store
//!   lattice vectors `a`, `b`, `c` as the **columns** of `H`, so
//!   `r = H s + origin`. [`Cell::from_ase`], [`Cell::from_con`],
//!   [`Cell::from_vesin`], and [`PartialCell::from_ase`] accept
//!   row-major 3x3 arrays `[a, b, c]`.
//! - **Displacement sign**: [`Cell::displacement`] returns the wrapped
//!   vector from `p` to `q`, i.e., `wrap(q - p)`. The origin cancels in
//!   every displacement and distance call.
//! - **Engine wrap vs Euclidean MIC**: [`Cell::displacement`] and
//!   [`Cell::dist2`] compute the fractional wrap `ds = wrap(Hinv (q - p))`,
//!   then `dr = H ds` (the LAMMPS `lamda` / eOn / HOOMD convention).
//!   Orthorhombic boxes use `d - L floor(d/L + 1/2)` on every image,
//!   keeping `-L/2` and mapping `+L/2` to `-L/2`. Restricted triclinic
//!   boxes use the triangular `lamda` step. [`Cell::reduce_tilts`] is
//!   GROMACS `correct_box`; [`Cell::to_restricted`] is the LAMMPS
//!   general-to-restricted rotation.
//! - **Euclidean nearest image**: [`Cell::displacement_euclidean`] runs
//!   Smith's half-altitude test, then Babai's rounding and the
//!   Sommer-Feder-Shalvi slicer on the Selling superbasis cached on the
//!   calling thread (with Lagrange size reduction before the Delone step).
//!   On a tie it keeps the engine wrap.
//!
//! # Batch kernels, fixed-point, and partial periodicity
//!
//! [`dist2_many`], [`dist2_pairs`], [`wrap_many`], and
//! [`dist2_ortho_diffs`] batch the engine wrap. Each batch reads
//! [`simd_path`] once and runs AVX-512, AVX2, AVX, NEON, or scalar.
//! [`Cell::fixed`] stores a position as 64-bit fixed-point fractions,
//! and the wrap between two of them is exact integer arithmetic
//! ([`Cell::dist2_fixed`], [`dist2_many_fixed`], [`dist2_pairs_fixed`]);
//! [`Cell::fixed32`] keeps the top 32 bits for half the memory traffic.
//! [`Cell::dist2_shifted_indexed`] gathers a linked-cell bin's index
//! list into the shifted kernel. [`reduce_pairs`] turns a vesin image
//! pair list into one minimum-image pair and drops the self image.
//! [`PartialCell`] is periodic along some axes only: a slab, a wire, or
//! an open system, with no image along an open axis.
//!
//! # Error behaviour
//!
//! Fallible functions return [`Result<_, Error>`](Error). [`Error::BadBox`]
//! signals a singular, non-finite, or degenerate cell; [`Error::BufferSize`]
//! signals mismatched slice lengths or out-of-range atom indices;
//! [`Error::Empty`] signals an empty or misaligned ring-polymer input.
//! Individual distance and displacement methods on [`Cell`] and
//! [`PartialCell`] are infallible once the cell has been constructed.
//!
//! It is a LODE library. The Rust crate is the implementation. The C
//! ABI (`minimage_*`) is the hourglass waist, the same shape as
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
mod filter;
mod fixed;
mod fixed32;
mod fused;
mod images;
mod kernel;
mod minkowski;
mod pairs;
mod partial;
mod selling;
mod simd;

#[cfg(test)]
mod isa_golden;

pub use batch::{
    dist2_euclidean_many, dist2_euclidean_pairs, dist2_euclidean_pairs_warm, dist2_many,
    dist2_many_fixed, dist2_many_fixed32, dist2_ortho_diffs, dist2_pairs, dist2_pairs_fixed,
    dist2_pairs_fixed32, fixed32_many, fixed_many, wrap_many,
};
pub use cell::{dump_bounds_to_h, Cell};
pub use error::Error;
pub use filter::{filter_shifted, ring_centroid, FilterOut};
pub use images::{
    cutoff_exceeds_half_width, image_extents, pairs_within_images, visit_pairs_within_images,
    ImagePair, MAX_IMAGE_COUNT, MAX_IMAGE_EXTENT,
};
pub use minkowski::{is_minkowski_reduced, minkowski_reduce};
pub use pairs::{reduce_pairs, reduce_pairs_packed};
pub use partial::PartialCell;
pub use simd::simd_path;

#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "capi")]
pub use capi::*;
