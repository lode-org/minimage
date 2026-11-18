//! Recoverable failures from cell construction or batch/pair calls.

use std::fmt;

/// Why a cell construction, reduction, or batch call could not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadBox => write!(f, "singular or non-positive cell"),
            Error::BufferSize => write!(
                f,
                "out buffer length must match the pair count, or atom index >= n_atoms"
            ),
            Error::Empty => write!(f, "no pairs"),
        }
    }
}

impl std::error::Error for Error {}
