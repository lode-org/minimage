//! Shifted-pair filter and the ring-polymer centroid.
//!
//! A Verlet skin and a ring-polymer bead share one candidate list built
//! at `cutoff + skin`. Each later call keeps the rows whose shifted
//! distance is still inside the real cutoff. The arithmetic is
//! [`Cell::dist2_shifted`]: `|q + H S - p|^2`, the same image the
//! linked-cell stencil stored.

use crate::cell::Cell;
use crate::Error;

/// Rows that survived [`filter_shifted`].
///
/// Capacities are kept across calls. `vector` is empty unless the call
/// asked for vectors, and then it is one Cartesian `q + H S - p` per row.
#[derive(Clone, Debug, Default)]
pub struct FilterOut {
    /// Source index.
    pub i: Vec<u32>,
    /// Target index.
    pub j: Vec<u32>,
    /// Integer cell shift of the target.
    pub shift: Vec<[i32; 3]>,
    /// Squared distance after the shift.
    pub dist2: Vec<f64>,
    /// Shifted Cartesian vector. Empty when the call did not ask for it.
    pub vector: Vec<[f64; 3]>,
}

/// Drop candidates whose shifted distance is not strictly below `cut2`.
///
/// `ii`, `jj`, and `shift` are the same length. An index past `xyz`
/// is [`Error::BufferSize`]. `out` is cleared and filled; its capacity
/// stays. `want_vector` stores `q + H S - p` beside each kept row.
#[allow(clippy::too_many_arguments)]
pub fn filter_shifted(
    cell: &Cell,
    xyz: &[[f64; 3]],
    ii: &[u32],
    jj: &[u32],
    shift: &[[i32; 3]],
    cut2: f64,
    want_vector: bool,
    out: &mut FilterOut,
) -> Result<(), Error> {
    if ii.len() != jj.len() || ii.len() != shift.len() {
        return Err(Error::BufferSize);
    }
    out.i.clear();
    out.j.clear();
    out.shift.clear();
    out.dist2.clear();
    out.vector.clear();
    let n = ii.len();
    out.i.reserve(n / 4 + 1);
    out.j.reserve(n / 4 + 1);
    out.shift.reserve(n / 4 + 1);
    out.dist2.reserve(n / 4 + 1);
    if want_vector {
        out.vector.reserve(n / 4 + 1);
    }
    if n == 0 {
        return Ok(());
    }
    let widths = cell.widths();
    let h = cell.h();
    let table = if cell.is_ortho() {
        None
    } else {
        Some(shift_table(h))
    };
    filter_runs(
        xyz,
        ii,
        jj,
        shift,
        cut2,
        want_vector,
        out,
        widths,
        cell.is_ortho(),
        table.as_deref(),
        h,
    )
}

const SHIFT_MAX: i32 = 3;

fn shift_table(h: [[f64; 3]; 3]) -> Vec<[f64; 3]> {
    let n = (2 * SHIFT_MAX + 1) as usize;
    let mut table = vec![[0.0; 3]; n * n * n];
    for sx in -SHIFT_MAX..=SHIFT_MAX {
        for sy in -SHIFT_MAX..=SHIFT_MAX {
            for sz in -SHIFT_MAX..=SHIFT_MAX {
                let fa = f64::from(sx);
                let fb = f64::from(sy);
                let fc = f64::from(sz);
                table[shift_slot(sx, sy, sz)] = [
                    fa * h[0][0] + fb * h[1][0] + fc * h[2][0],
                    fa * h[0][1] + fb * h[1][1] + fc * h[2][1],
                    fa * h[0][2] + fb * h[1][2] + fc * h[2][2],
                ];
            }
        }
    }
    table
}

fn shift_slot(sx: i32, sy: i32, sz: i32) -> usize {
    let n = (2 * SHIFT_MAX + 1) as usize;
    let x = (sx + SHIFT_MAX) as usize;
    let y = (sy + SHIFT_MAX) as usize;
    let z = (sz + SHIFT_MAX) as usize;
    (z * n + y) * n + x
}

/// One source stays in registers for its run. An orthorhombic box scales
/// `S` by the three widths. Any other box loads `H S` from the 7³ table
/// when every component lies in `[-3, 3]`.
#[allow(clippy::too_many_arguments)]
fn filter_runs(
    xyz: &[[f64; 3]],
    ii: &[u32],
    jj: &[u32],
    shift: &[[i32; 3]],
    cut2: f64,
    want_vector: bool,
    out: &mut FilterOut,
    widths: [f64; 3],
    ortho: bool,
    table: Option<&[[f64; 3]]>,
    h: [[f64; 3]; 3],
) -> Result<(), Error> {
    let n = ii.len();
    let natoms = xyz.len();
    let mut k = 0;
    while k < n {
        let i = ii[k] as usize;
        if i >= natoms {
            return Err(Error::BufferSize);
        }
        let p = xyz[i];
        while k < n && ii[k] as usize == i {
            let j = jj[k] as usize;
            if j >= natoms {
                return Err(Error::BufferSize);
            }
            // The index was checked. The load is the whole cost of a candidate.
            let q = unsafe { *xyz.get_unchecked(j) };
            let s = unsafe { *shift.get_unchecked(k) };
            let t = if ortho {
                [
                    f64::from(s[0]) * widths[0],
                    f64::from(s[1]) * widths[1],
                    f64::from(s[2]) * widths[2],
                ]
            } else if s[0].abs() <= SHIFT_MAX && s[1].abs() <= SHIFT_MAX && s[2].abs() <= SHIFT_MAX
            {
                table.unwrap()[shift_slot(s[0], s[1], s[2])]
            } else {
                let fa = f64::from(s[0]);
                let fb = f64::from(s[1]);
                let fc = f64::from(s[2]);
                [
                    fa * h[0][0] + fb * h[1][0] + fc * h[2][0],
                    fa * h[0][1] + fb * h[1][1] + fc * h[2][1],
                    fa * h[0][2] + fb * h[1][2] + fc * h[2][2],
                ]
            };
            let dx = q[0] + t[0] - p[0];
            let dy = q[1] + t[1] - p[1];
            let dz = q[2] + t[2] - p[2];
            let d2 = (dx * dx + dy * dy) + dz * dz;
            if d2 < cut2 {
                out.i.push(ii[k]);
                out.j.push(jj[k]);
                out.shift.push(s);
                out.dist2.push(d2);
                if want_vector {
                    out.vector.push([dx, dy, dz]);
                }
            }
            k += 1;
        }
    }
    Ok(())
}

/// Mean position of each atom across the beads of a ring polymer.
///
/// `xyz` is bead-major: bead `b`, atom `a` is `xyz[b * atoms + a]`,
/// with `atoms = xyz.len() / beads`. Each bead is unwrapped onto bead 0
/// by the engine minimum image ([`Cell::fractional`] difference, wrapped
/// into `[-1/2, 1/2)`), the fractions are averaged, and the mean is
/// wrapped back into the primary cell. A bead spread wider than half a
/// box along one axis is not a single image; the average then follows
/// whichever image the wrap picked.
pub fn ring_centroid(cell: &Cell, beads: usize, xyz: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, Error> {
    if beads == 0 || xyz.is_empty() || xyz.len() % beads != 0 {
        return Err(Error::Empty);
    }
    let atoms = xyz.len() / beads;
    let mut out = Vec::with_capacity(atoms);
    let inv = 1.0 / beads as f64;
    for a in 0..atoms {
        let s0 = cell.fractional(xyz[a]);
        let mut acc = [0.0; 3];
        for b in 0..beads {
            let s = cell.fractional(xyz[b * atoms + a]);
            for k in 0..3 {
                let mut d = s[k] - s0[k];
                if d > 0.5 {
                    d -= 1.0;
                } else if d < -0.5 {
                    d += 1.0;
                }
                acc[k] += s0[k] + d;
            }
        }
        let mean = [
            wrap01(acc[0] * inv),
            wrap01(acc[1] * inv),
            wrap01(acc[2] * inv),
        ];
        out.push(cell.cartesian(mean));
    }
    Ok(out)
}

#[inline]
fn wrap01(mut s: f64) -> f64 {
    s -= s.floor();
    if s < 0.0 {
        s += 1.0;
    }
    if s >= 1.0 {
        s = 0.0;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_keeps_the_near_image() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let ii = [0, 0];
        let jj = [1, 2];
        let shift = [[-1, 0, 0], [0, 0, 0]];
        let mut out = FilterOut::default();
        filter_shifted(&cell, &xyz, &ii, &jj, &shift, 1.0, true, &mut out).unwrap();
        assert_eq!(out.i, vec![0]);
        assert_eq!(out.j, vec![1]);
        assert!((out.dist2[0] - 0.64).abs() < 1e-12);
        assert!((out.vector[0][0] + 0.8).abs() < 1e-12);
    }

    #[test]
    fn centroid_of_a_wrapped_pair_sits_on_the_boundary() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 1.0, 2.0], [9.8, 1.0, 2.0]];
        let c = ring_centroid(&cell, 2, &xyz).unwrap();
        assert!((c[0][0] - 0.0).abs() < 1e-12 || (c[0][0] - 10.0).abs() < 1e-9);
        assert!((c[0][1] - 1.0).abs() < 1e-12);
        assert!((c[0][2] - 2.0).abs() < 1e-12);
    }
}
