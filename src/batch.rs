//! Batched minimum-image squared distances.
//!
//! [`dist2_many`], [`dist2_pairs`], and [`wrap_many`] are one fused AVX
//! pass over the packed rows when the CPU has it, for every cell shape,
//! and a squared distance equals the per-pair call bit for bit; a
//! wrapped vector equals it up to the sign of a zero.
//! [`dist2_ortho_diffs`] keeps the Highway `BatchPeriodicDistSq` shape:
//! precomputed differences, one reciprocal per axis, then `abs` and
//! `round`. [`dist2_shifted_many`](crate::Cell::dist2_shifted_many) is
//! Rapaport's linked-cell pair, one lattice shift for a whole bin.

use crate::{Cell, Error};

/// Minimum-image wrap of packed difference vectors.
///
/// `diffs` and `out` are row-major `n` triples. Each row is `q - p`
/// (the cell origin does not enter).
pub fn wrap_many(cell: &Cell, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) -> Result<(), Error> {
    if out.len() != diffs.len() {
        return Err(Error::BufferSize);
    }
    crate::fused::wrap_many(cell.frame(), diffs, out);
    Ok(())
}

/// Squared MIC distances from `p` to each packed candidate in `qs`.
///
/// `qs` is row-major `n` triples. `out` has length `n`.
pub fn dist2_many(cell: &Cell, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) -> Result<(), Error> {
    if out.len() != qs.len() {
        return Err(Error::BufferSize);
    }
    crate::fused::dist2_many(cell.frame(), p, qs, out);
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
    crate::fused::dist2_pairs(cell.frame(), ps, qs, out);
    Ok(())
}

/// [`Cell::dist2_euclidean`] from `p` to each row of `qs`, bit for bit.
/// The superbasis is looked up once per call rather than once per pair.
pub fn dist2_euclidean_many(
    cell: &Cell,
    p: [f64; 3],
    qs: &[[f64; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if out.len() != qs.len() {
        return Err(Error::BufferSize);
    }
    euclidean_many(cell, p, qs, out);
    Ok(())
}

/// [`dist2_euclidean_many`] once the lengths are known to agree.
pub(crate) fn euclidean_many(cell: &Cell, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
    if cell.is_ortho() {
        return crate::fused::dist2_many(cell.frame(), p, qs, out);
    }
    let (frame, room) = (cell.frame(), cell.smith_room());
    crate::selling::with_basis(cell.h(), |b| {
        crate::fused::dist2_euclidean_many(frame, room, b, p, qs, out)
    });
}

/// [`Cell::dist2_euclidean`] for paired rows, bit for bit, with one
/// superbasis lookup per call.
pub fn dist2_euclidean_pairs(
    cell: &Cell,
    ps: &[[f64; 3]],
    qs: &[[f64; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if ps.len() != qs.len() || out.len() != ps.len() {
        return Err(Error::BufferSize);
    }
    euclidean_pairs(cell, ps, qs, out);
    Ok(())
}

/// [`dist2_euclidean_pairs`] once the lengths are known to agree.
pub(crate) fn euclidean_pairs(cell: &Cell, ps: &[[f64; 3]], qs: &[[f64; 3]], out: &mut [f64]) {
    if cell.is_ortho() {
        return crate::fused::dist2_pairs(cell.frame(), ps, qs, out);
    }
    let (frame, room) = (cell.frame(), cell.smith_room());
    crate::selling::with_basis(cell.h(), |b| {
        crate::fused::dist2_euclidean_pairs(frame, room, b, ps, qs, out)
    });
}

/// Euclidean nearest-image squared distances for pairs that persist
/// across frames.
///
/// `images[k]` is the cell-basis lattice shift `n` of pair `k`'s image,
/// `x = (q - p) - H n`, from the previous call; start with zeros. A pair
/// whose stored image still lies in the Voronoi cell costs a subtraction
/// and seven dot products. The others run the full search, bit for bit
/// [`Cell::dist2_euclidean`], and get their new shift. The image is the
/// one `dist2_euclidean` finds; for a stored image the distance equals
/// it to rounding, since `H n` is subtracted directly rather than
/// through the engine wrap. An orthorhombic cell takes the plain batch
/// and leaves `images` alone.
pub fn dist2_euclidean_pairs_warm(
    cell: &Cell,
    ps: &[[f64; 3]],
    qs: &[[f64; 3]],
    images: &mut [[i32; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if ps.len() != qs.len() || out.len() != ps.len() || images.len() != ps.len() {
        return Err(Error::BufferSize);
    }
    euclidean_pairs_warm(cell, ps, qs, images, out);
    Ok(())
}

/// [`dist2_euclidean_pairs_warm`] once the lengths are known to agree.
pub(crate) fn euclidean_pairs_warm(
    cell: &Cell,
    ps: &[[f64; 3]],
    qs: &[[f64; 3]],
    images: &mut [[i32; 3]],
    out: &mut [f64],
) {
    if cell.is_ortho() {
        return crate::fused::dist2_pairs(cell.frame(), ps, qs, out);
    }
    crate::selling::with_basis(cell.h(), |basis| {
        let w = crate::fused::Warm {
            h: cell.h(),
            hinv: cell.hinv(),
            frame: cell.frame(),
            room: cell.smith_room(),
            basis,
        };
        crate::fused::dist2_euclidean_pairs_warm(&w, ps, qs, images, out)
    });
}

/// [`Cell::fixed`] for packed positions: one AVX2 pass folds, floors, and
/// takes the mantissa of `t + 1` for four rows at once.
pub fn fixed_many(cell: &Cell, rs: &[[f64; 3]], out: &mut [[u64; 3]]) -> Result<(), Error> {
    if out.len() != rs.len() {
        return Err(Error::BufferSize);
    }
    crate::fixed::fixed_many(&cell.fold(), rs, out);
    Ok(())
}

/// [`Cell::fixed32`] for packed positions.
pub fn fixed32_many(cell: &Cell, rs: &[[f64; 3]], out: &mut [[u32; 3]]) -> Result<(), Error> {
    if out.len() != rs.len() {
        return Err(Error::BufferSize);
    }
    crate::fixed32::fixed32_many(&cell.fold(), rs, out);
    Ok(())
}

/// [`Cell::dist2_fixed32`] from `p` to each row of `qs`, bit for bit:
/// sixteen rows per AVX-512 pass, eight per AVX2 pass.
pub fn dist2_many_fixed32(
    cell: &Cell,
    p: [u32; 3],
    qs: &[[u32; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if out.len() != qs.len() {
        return Err(Error::BufferSize);
    }
    crate::fixed32::dist2_many(cell.fixed32_lattice(), p, qs, out);
    Ok(())
}

/// [`Cell::dist2_fixed32`] for paired rows, bit for bit.
pub fn dist2_pairs_fixed32(
    cell: &Cell,
    ps: &[[u32; 3]],
    qs: &[[u32; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if ps.len() != qs.len() || out.len() != ps.len() {
        return Err(Error::BufferSize);
    }
    crate::fixed32::dist2_pairs(cell.fixed32_lattice(), ps, qs, out);
    Ok(())
}

/// Squared engine-wrap distances from fixed-point `p` to each fixed-point
/// `qs` ([`Cell::fixed`]). Integer wrap, one product with `H`.
pub fn dist2_many_fixed(
    cell: &Cell,
    p: [u64; 3],
    qs: &[[u64; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if out.len() != qs.len() {
        return Err(Error::BufferSize);
    }
    crate::fixed::dist2_many(cell.fixed_lattice(), p, qs, out);
    Ok(())
}

/// Squared engine-wrap distances for fixed-point pair lists.
pub fn dist2_pairs_fixed(
    cell: &Cell,
    ps: &[[u64; 3]],
    qs: &[[u64; 3]],
    out: &mut [f64],
) -> Result<(), Error> {
    if ps.len() != qs.len() || out.len() != ps.len() {
        return Err(Error::BufferSize);
    }
    crate::fixed::dist2_pairs(cell.fixed_lattice(), ps, qs, out);
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
    fn pairs_match_scalar_including_later_images() {
        let cell = Cell::ortho(10.0, 11.0, 12.0).unwrap();
        let mut ps = vec![[0.0; 3]; 70];
        let mut qs = vec![[0.0; 3]; 70];
        for i in 0..70 {
            let t = f64::from(i as i32) - 20.0;
            ps[i] = [0.2, 0.0, 0.0];
            qs[i] = [t, -18.0, 3.0];
        }
        let mut out = vec![0.0; 70];
        dist2_pairs(&cell, &ps, &qs, &mut out).unwrap();
        for i in 0..70 {
            assert!((out[i] - cell.dist2(ps[i], qs[i])).abs() < 1e-9);
        }
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
