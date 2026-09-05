//! Batched minimum-image squared distances.
//!
//! The orthorhombic kernel hoists one reciprocal per axis and wraps
//! each component independently, the same arithmetic as the Highway
//! `BatchPeriodicDistSq` path. The general path applies
//! [`crate::Cell::dist2`] per pair. Loops are written as packed
//! coordinate streams so a vectorising compiler emits SIMD for the
//! ortho remainder.

use crate::{Cell, Error};

/// Minimum-image wrap of packed difference vectors.
///
/// `diffs` and `out` are row-major `n` triples. Each row is `q - p`
/// (the cell origin does not enter).
pub fn wrap_many(
    cell: &Cell,
    diffs: &[[f64; 3]],
    out: &mut [[f64; 3]],
) -> Result<(), Error> {
    if out.len() != diffs.len() {
        return Err(Error::BufferSize);
    }
    for (d, o) in diffs.iter().zip(out.iter_mut()) {
        *o = cell.displacement([0.0, 0.0, 0.0], *d);
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
    } else {
        for (q, o) in qs.iter().zip(out.iter_mut()) {
            *o = cell.dist2(p, *q);
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
        let rbx = 1.0 / l[0];
        let rby = 1.0 / l[1];
        let rbz = 1.0 / l[2];
        for i in 0..ps.len() {
            out[i] = dist2_ortho_one(ps[i], qs[i], l, rbx, rby, rbz);
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
    let rbx = 1.0 / bx;
    let rby = 1.0 / by;
    let rbz = 1.0 / bz;
    // Four-wide unroll so the wrap is a contiguous SIMD-shaped loop.
    let mut i = 0;
    while i + 4 <= n {
        for k in 0..4 {
            let mut ddx = dx[i + k].abs();
            let mut ddy = dy[i + k].abs();
            let mut ddz = dz[i + k].abs();
            ddx -= bx * (ddx * rbx).round();
            ddy -= by * (ddy * rby).round();
            ddz -= bz * (ddz * rbz).round();
            out[i + k] = ddx * ddx + ddy * ddy + ddz * ddz;
        }
        i += 4;
    }
    while i < n {
        let mut ddx = dx[i].abs();
        let mut ddy = dy[i].abs();
        let mut ddz = dz[i].abs();
        ddx -= bx * (ddx * rbx).round();
        ddy -= by * (ddy * rby).round();
        ddz -= bz * (ddz * rbz).round();
        out[i] = ddx * ddx + ddy * ddy + ddz * ddz;
        i += 1;
    }
    Ok(())
}

fn dist2_many_ortho(l: [f64; 3], p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
    let rbx = 1.0 / l[0];
    let rby = 1.0 / l[1];
    let rbz = 1.0 / l[2];
    for (q, o) in qs.iter().zip(out.iter_mut()) {
        *o = dist2_ortho_one(p, *q, l, rbx, rby, rbz);
    }
}

#[inline]
fn dist2_ortho_one(p: [f64; 3], q: [f64; 3], l: [f64; 3], rbx: f64, rby: f64, rbz: f64) -> f64 {
    let mut ddx = (q[0] - p[0]).abs();
    let mut ddy = (q[1] - p[1]).abs();
    let mut ddz = (q[2] - p[2]).abs();
    ddx -= l[0] * (ddx * rbx).round();
    ddy -= l[1] * (ddy * rby).round();
    ddz -= l[2] * (ddz * rbz).round();
    ddx * ddx + ddy * ddy + ddz * ddz
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
}
