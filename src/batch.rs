//! Batched minimum-image squared distances.
//!
//! Orthorhombic differences are a structure-of-arrays kernel: one
//! reciprocal per axis, then `abs` and `round`, the Highway
//! `BatchPeriodicDistSq` arithmetic, issued as AVX when the CPU has
//! it. Restricted triclinic batches use the triangular
//! lamda step. A general `H` hoists the inverse and applies it per
//! row. [`dist2_shifted_many`](crate::Cell::dist2_shifted_many) is
//! Rapaport's linked-cell pair, one lattice shift for a whole bin.

use crate::kernel::{self, n2};
use crate::{Cell, Error};

/// Minimum-image wrap of packed difference vectors.
///
/// `diffs` and `out` are row-major `n` triples. Each row is `q - p`
/// (the cell origin does not enter).
pub fn wrap_many(cell: &Cell, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) -> Result<(), Error> {
    if out.len() != diffs.len() {
        return Err(Error::BufferSize);
    }
    if cell.is_ortho() {
        crate::simd::wrap_many_ortho(cell.widths(), diffs, out);
    } else {
        for (d, o) in diffs.iter().zip(out.iter_mut()) {
            *o = cell.wrap_diff(*d);
        }
    }
    Ok(())
}

/// Squared MIC distances from `p` to each packed candidate in `qs`.
///
/// `qs` is row-major `n` triples. `out` has length `n`.
pub fn dist2_many(cell: &Cell, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) -> Result<(), Error> {
    if out.len() != qs.len() {
        return Err(Error::BufferSize);
    }
    if cell.is_ortho() {
        dist2_many_ortho(cell.widths(), p, qs, out);
    } else if cell.is_restricted() {
        for (q, o) in qs.iter().zip(out.iter_mut()) {
            let d = cell.wrap_diff([q[0] - p[0], q[1] - p[1], q[2] - p[2]]);
            *o = n2(d);
        }
    } else {
        let h = cell.h();
        let hinv = cell.hinv();
        for (q, o) in qs.iter().zip(out.iter_mut()) {
            let d = kernel::general_wrap(h, hinv, [q[0] - p[0], q[1] - p[1], q[2] - p[2]]);
            *o = n2(d);
        }
    }
    Ok(())
}

/// Squared MIC distances for packed pair lists `ps` and `qs`.
///
/// Each slice is row-major `n` triples. `out` has length `n`.
pub fn dist2_pairs(
    cell: &Cell,
    ps: &[[f64; 3]],
    qs: &[[f64; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if ps.len() != qs.len() || out.len() != ps.len() {
        return Err(Error::BufferSize);
    }
    if cell.is_ortho() {
        let l = cell.widths();
        for i in 0..ps.len() {
            let d = kernel::ortho_wrap(
                l,
                [
                    qs[i][0] - ps[i][0],
                    qs[i][1] - ps[i][1],
                    qs[i][2] - ps[i][2],
                ],
            );
            out[i] = n2(d);
        }
    } else {
        for i in 0..ps.len() {
            out[i] = cell.dist2(ps[i], qs[i]);
        }
    }
    Ok(())
}

/// Orthorhombic wrap of precomputed differences, one reciprocal per axis.
///
/// `dx`, `dy`, `dz`, and `out` have length `n`. This is the Highway
/// kernel: `dr -= box * round(dr * (1 / box))` after taking `abs`.
pub fn dist2_ortho_diffs(
    dx: &[f64],
    dy: &[f64],
    dz: &[f64],
    bx: f64,
    by: f64,
    bz: f64,
    out: &mut [f64],
) -> Result<(), Error> {
    let n = dx.len().min(dy.len()).min(dz.len()).min(out.len());
    if n == 0 {
        return Ok(());
    }
    if !(bx > 0.0 && by > 0.0 && bz > 0.0) {
        return Err(Error::BadBox);
    }
    crate::simd::dist2_ortho_diffs(&dx[..n], &dy[..n], &dz[..n], bx, by, bz, &mut out[..n]);
    Ok(())
}

pub(crate) fn dist2_many_ortho(l: [f64; 3], p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
    // Squares match the signed wrap, so the SoA kernel is the same
    // distance. Chunks stay on the stack.
    const CHUNK: usize = 64;
    let mut dx = [0.0; CHUNK];
    let mut dy = [0.0; CHUNK];
    let mut dz = [0.0; CHUNK];
    let mut start = 0;
    while start < qs.len() {
        let n = (qs.len() - start).min(CHUNK);
        for (k, q) in qs[start..start + n].iter().enumerate() {
            dx[k] = q[0] - p[0];
            dy[k] = q[1] - p[1];
            dz[k] = q[2] - p[2];
        }
        crate::simd::dist2_ortho_diffs(
            &dx[..n],
            &dy[..n],
            &dz[..n],
            l[0],
            l[1],
            l[2],
            &mut out[start..start + n],
        );
        start += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_scalar_ortho() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let p = [0.2, 0.0, 0.0];
        let qs = [[9.4, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let mut out = [0.0; 3];
        dist2_many(&cell, p, &qs, &mut out).unwrap();
        for i in 0..3 {
            assert!((out[i] - cell.dist2(p, qs[i])).abs() < 1e-12);
        }
        assert!((out[0] - 0.64).abs() < 1e-12);
        let wide: Vec<[f64; 3]> = (0..70).map(|i| [f64::from(i) - 20.0, 0.0, 0.0]).collect();
        let mut wide_out = vec![0.0; wide.len()];
        dist2_many(&cell, p, &wide, &mut wide_out).unwrap();
        for (q, got) in wide.iter().zip(wide_out.iter()) {
            assert!((got - cell.dist2(p, *q)).abs() < 1e-12);
        }
    }

    #[test]
    fn diffs_kernel_matches_highway_shape() {
        let dx = [9.2_f64, 0.8, 5.0, 11.0, -3.0];
        let dy = [0.0; 5];
        let dz = [0.0; 5];
        let mut out = [0.0; 5];
        dist2_ortho_diffs(&dx, &dy, &dz, 10.0, 10.0, 10.0, &mut out).unwrap();
        assert!((out[0] - 0.64).abs() < 1e-12);
        assert!((out[1] - 0.64).abs() < 1e-12);
        assert!((out[2] - 25.0).abs() < 1e-12);
        assert!((out[3] - 1.0).abs() < 1e-12);
        assert!((out[4] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn sheared_batch_quarter() {
        let cell =
            Cell::from_lammps_bounds(15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap();
        let p = [0.2, 0.1, 1.0];
        let qs = [[9.7, 0.1, 1.0]];
        let mut out = [0.0];
        dist2_many(&cell, p, &qs, &mut out).unwrap();
        assert!((out[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn wrap_many_ortho_face() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let diffs = [[9.2, 0.0, 0.0], [0.2, 0.0, 0.0]];
        let mut out = [[0.0; 3]; 2];
        wrap_many(&cell, &diffs, &mut out).unwrap();
        assert!((out[0][0] + 0.8).abs() < 1e-12);
        assert!((out[1][0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn wrap_many_keeps_negative_half() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let diffs = [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let mut out = [[0.0; 3]; 2];
        wrap_many(&cell, &diffs, &mut out).unwrap();
        assert!((out[0][0] + 5.0).abs() < 1e-12);
        assert!((out[1][0] + 5.0).abs() < 1e-12);
        for (d, o) in diffs.iter().zip(out.iter()) {
            let s = cell.displacement([0.0, 0.0, 0.0], *d);
            assert!((o[0] - s[0]).abs() < 1e-12);
        }
    }
}
