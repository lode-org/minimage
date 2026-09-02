//! Recoverable failures from a cell or a distance call.

use std::fmt;

/// Why a cell or a distance call could not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// A box length is not strictly positive, or H is singular.
    BadBox,
    /// Caller output length does not match the candidate count.
    BufferSize,
    /// A pair list or candidate list is empty when a count is required.
    Empty,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadBox => write!(f, "singular or non-positive cell"),
            Error::BufferSize => write!(f, "out buffer length must match the pair count"),
            Error::Empty => write!(f, "no pairs"),
        }
    }
}

impl std::error::Error for Error {}
