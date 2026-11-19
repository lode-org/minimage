//! Recoverable failures from cell construction or batch/pair calls.

use std::fmt;

/// Why a cell construction, reduction, or batch call could not run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// Singular or degenerate cell (`|det| < 1e-18`, non-finite entry,
    /// non-positive perpendicular face width, collinear periodic basis
    /// vectors, or non-positive box lengths where required).
    BadBox,
    /// Caller buffer lengths do not match, or an atom index exceeds the
    /// coordinate slice length.
    BufferSize,
    /// A bead count or coordinate list is empty or not evenly divisible.
    Empty,
    /// Cutoff is negative or not finite.
    BadCutoff,
    /// A coordinate or fractional separation is not finite.
    NonFinite,
    /// Fractional separation rounded to the nearest lattice index plus
    /// the image extent overflows `i32`.
    ImageShiftOverflow,
    /// Per-axis image extent `ceil(cutoff / spacing)` exceeds the cap.
    ImageExtent {
        /// Lattice axis (`0`, `1`, or `2`).
        axis: usize,
        /// Requested shell count `ceil(cutoff / spacing)`.
        extent: f64,
        /// Per-axis extent cap.
        cap: i32,
        /// Cutoff distance.
        cutoff: f64,
        /// Perpendicular face spacing on `axis`.
        spacing: f64,
    },
    /// Total shift count `(2 * e_a + 1)(2 * e_b + 1)(2 * e_c + 1)`
    /// exceeds the cap.
    ImageCount {
        /// Per-axis extents `[e_a, e_b, e_c]`.
        extents: [i32; 3],
        /// Shift-count cap.
        cap: i64,
    },
}

impl Eq for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadBox => write!(f, "singular or non-positive cell"),
            Error::BufferSize => write!(
                f,
                "out buffer length must match the pair count, or atom index >= n_atoms"
            ),
            Error::Empty => write!(f, "no pairs"),
            Error::BadCutoff => write!(f, "cutoff must be finite and non-negative"),
            Error::NonFinite => write!(f, "non-finite fractional separation"),
            Error::ImageShiftOverflow => {
                write!(f, "fractional separation exceeds lattice index range")
            }
            Error::ImageExtent {
                axis,
                cap,
                cutoff,
                spacing,
                ..
            } => write!(
                f,
                "image extent on axis {axis} exceeds cap {cap} (cutoff={cutoff}, spacing={spacing})"
            ),
            Error::ImageCount { extents, cap } => write!(
                f,
                "image count exceeds cap {cap} (extents {} {} {})",
                extents[0], extents[1], extents[2]
            ),
        }
    }
}

impl std::error::Error for Error {}
