//! Minimum image with periodicity on some axes only: slabs and wires.
//!
//! [`PartialCell`] keeps the lattice vectors of the periodic axes and
//! takes no images along the open ones. The nearest image is searched
//! over the lattice those vectors span: one vector is a rounding along
//! it, two are a closest-vector problem in their plane, solved on a
//! Lagrange-reduced basis by rounding and the nine neighbouring
//! candidates. The component of a displacement normal to the periodic
//! vectors is never changed. With all three axes periodic the cell is a
//! [`Cell`] and [`PartialCell::displacement_euclidean`] is
//! [`Cell::displacement_euclidean`], bit for bit.
//!
//! As in [`Cell::displacement_euclidean`], the start is the engine wrap
//! `s - floor(s + 1/2)` along each periodic vector, and a shorter
//! candidate replaces it only when it is shorter by more than
//! `1e-12 (1 + |start|^2)`; a half-box tie keeps the engine vector.

use crate::kernel::{self, floor_fast};
use crate::{Cell, Error};

/// Periodic along the axes of a mask, open along the others.
///
/// ```
/// # use minimage::PartialCell;
/// # fn main() -> Result<(), minimage::Error> {
/// // A slab periodic in x and y with 10 A vectors, open in z.
/// let slab = PartialCell::from_ase(
///     [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]],
///     [true, true, false],
/// )?;
/// let d = slab.displacement_euclidean([0.5, 0.5, 0.0], [9.5, 0.5, 30.0]);
/// assert!((d[0] + 1.0).abs() < 1e-12 && (d[2] - 30.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialCell {
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// No periodic axis: the Cartesian difference.
    Open,
    /// One periodic vector `a`, with `a / |a|^2`.
    Wire { a: [f64; 3], dual: [f64; 3] },
    /// Two periodic vectors and their in-plane duals, with the
    /// Lagrange-reduced basis of the same lattice and its duals.
    Slab {
        basis: [[f64; 3]; 2],
        dual: [[f64; 3]; 2],
        reduced: [[f64; 3]; 2],
        reduced_dual: [[f64; 3]; 2],
    },
    /// Three periodic vectors.
    Full(Cell),
}

#[inline(always)]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn axpy(y: [f64; 3], alpha: f64, x: [f64; 3]) -> [f64; 3] {
    [
        y[0] + alpha * x[0],
        y[1] + alpha * x[1],
        y[2] + alpha * x[2],
    ]
}

/// In-plane duals of two independent vectors: `dual[i] . basis[j]` is
/// the Kronecker delta and each dual lies in the plane of the basis.
fn plane_duals(u: [f64; 3], v: [f64; 3]) -> Result<[[f64; 3]; 2], Error> {
    let uu = dot(u, u);
    let vv = dot(v, v);
    let uv = dot(u, v);
    let det = uu * vv - uv * uv;
    if !det.is_finite() || det <= 1e-24 * uu * vv {
        return Err(Error::BadBox);
    }
    let inv = 1.0 / det;
    let du = axpy(
        [vv * inv * u[0], vv * inv * u[1], vv * inv * u[2]],
        -uv * inv,
        v,
    );
    let dv = axpy(
        [uu * inv * v[0], uu * inv * v[1], uu * inv * v[2]],
        -uv * inv,
        u,
    );
    Ok([du, dv])
}

/// Lagrange (Gauss) reduction of a two-dimensional lattice basis.
fn lagrange_reduce(mut u: [f64; 3], mut v: [f64; 3]) -> [[f64; 3]; 2] {
    if dot(u, u) > dot(v, v) {
        std::mem::swap(&mut u, &mut v);
    }
    // At most a few dozen steps for any representable basis; the bound
    // only guards against a non-terminating loop on pathological input.
    for _ in 0..128 {
        let mu = kernel::round_away(dot(u, v) / dot(u, u));
        if mu != 0.0 {
            v = axpy(v, -mu, u);
        }
        if dot(v, v) >= dot(u, u) {
            break;
        }
        std::mem::swap(&mut u, &mut v);
    }
    [u, v]
}

fn finite3(v: [f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

impl PartialCell {
    /// Cell from ASE-style rows `(a, b, c)` and a periodicity mask.
    ///
    /// Rows of open axes are ignored and may be zero. The periodic rows
    /// must be finite and linearly independent. All three periodic is
    /// [`Cell::from_ase`].
    pub fn from_ase(rows: [[f64; 3]; 3], periodic: [bool; 3]) -> Result<Self, Error> {
        let vectors: Vec<[f64; 3]> = (0..3).filter(|&i| periodic[i]).map(|i| rows[i]).collect();
        if !vectors.iter().all(|&v| finite3(v)) {
            return Err(Error::BadBox);
        }
        let kind = match vectors.len() {
            0 => Kind::Open,
            1 => {
                let a = vectors[0];
                let aa = dot(a, a);
                if !aa.is_finite() || aa <= 0.0 {
                    return Err(Error::BadBox);
                }
                Kind::Wire {
                    a,
                    dual: [a[0] / aa, a[1] / aa, a[2] / aa],
                }
            }
            2 => {
                let basis = [vectors[0], vectors[1]];
                let dual = plane_duals(basis[0], basis[1])?;
                let reduced = lagrange_reduce(basis[0], basis[1]);
                let reduced_dual = plane_duals(reduced[0], reduced[1])?;
                Kind::Slab {
                    basis,
                    dual,
                    reduced,
                    reduced_dual,
                }
            }
            _ => Kind::Full(Cell::from_ase(rows)?),
        };
        Ok(Self { kind })
    }

    /// Number of periodic axes.
    pub fn periodic_count(&self) -> usize {
        match self.kind {
            Kind::Open => 0,
            Kind::Wire { .. } => 1,
            Kind::Slab { .. } => 2,
            Kind::Full(_) => 3,
        }
    }

    /// The fully periodic cell, when all three axes are periodic.
    pub fn full(&self) -> Option<&Cell> {
        match &self.kind {
            Kind::Full(cell) => Some(cell),
            _ => None,
        }
    }

    /// Displacement from `p` to `q` to the nearest image over the
    /// periodic lattice; open axes take no image.
    #[inline]
    pub fn displacement_euclidean(&self, p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        match &self.kind {
            Kind::Full(cell) => cell.displacement_euclidean(p, q),
            Kind::Open => [q[0] - p[0], q[1] - p[1], q[2] - p[2]],
            Kind::Wire { a, dual } => {
                let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                let s = dot(d, *dual);
                axpy(d, -floor_fast(s + 0.5), *a)
            }
            Kind::Slab {
                basis,
                dual,
                reduced,
                reduced_dual,
            } => {
                let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                let sa = dot(d, dual[0]);
                let sb = dot(d, dual[1]);
                let start = axpy(
                    axpy(d, -floor_fast(sa + 0.5), basis[0]),
                    -floor_fast(sb + 0.5),
                    basis[1],
                );
                slab_closest(start, reduced, reduced_dual)
            }
        }
    }

    /// Squared length of [`Self::displacement_euclidean`].
    #[inline]
    pub fn dist2_euclidean(&self, p: [f64; 3], q: [f64; 3]) -> f64 {
        kernel::n2(self.displacement_euclidean(p, q))
    }
}

/// Closest image of `start` over the plane lattice of `reduced`: round
/// the reduced coordinates, then try the nine neighbouring candidates.
/// For a Lagrange-reduced basis the nearest lattice vector to any point
/// lies among them.
#[inline(never)]
fn slab_closest(start: [f64; 3], reduced: &[[f64; 3]; 2], dual: &[[f64; 3]; 2]) -> [f64; 3] {
    let s2 = kernel::n2(start);
    let ru = kernel::round_away(dot(start, dual[0]));
    let rv = kernel::round_away(dot(start, dual[1]));
    let mut best = start;
    let mut b2 = s2;
    for i in -1..=1 {
        for j in -1..=1 {
            let cand = axpy(
                axpy(start, -(ru + f64::from(i)), reduced[0]),
                -(rv + f64::from(j)),
                reduced[1],
            );
            let c2 = kernel::n2(cand);
            if c2 < b2 {
                best = cand;
                b2 = c2;
            }
        }
    }
    if b2 + 1e-12 * (1.0 + s2) < s2 {
        best
    } else {
        start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force over a window of images of the periodic vectors.
    fn brute(rows: [[f64; 3]; 3], periodic: [bool; 3], p: [f64; 3], q: [f64; 3]) -> f64 {
        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let span = |k: usize| if periodic[k] { -60..=60 } else { 0..=0 };
        let mut best = f64::INFINITY;
        for i in span(0) {
            for j in span(1) {
                for k in span(2) {
                    let mut v = d;
                    for (n, row) in [(i, rows[0]), (j, rows[1]), (k, rows[2])] {
                        v = axpy(v, f64::from(n), row);
                    }
                    best = best.min(kernel::n2(v));
                }
            }
        }
        best
    }

    /// Deterministic points spread over a few cells.
    fn points(n: usize, scale: f64) -> Vec<[f64; 3]> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                [
                    scale * (2.0 * next() - 1.0),
                    scale * (2.0 * next() - 1.0),
                    scale * (2.0 * next() - 1.0),
                ]
            })
            .collect()
    }

    fn check(rows: [[f64; 3]; 3], periodic: [bool; 3]) {
        let cell = PartialCell::from_ase(rows, periodic).unwrap();
        let pts = points(400, 12.0);
        for w in pts.chunks(2) {
            let (p, q) = (w[0], w[1]);
            let d = cell.displacement_euclidean(p, q);
            let d2 = kernel::n2(d);
            let want = brute(rows, periodic, p, q);
            assert!(
                (d2 - want).abs() <= 1e-9 * (1.0 + want),
                "rows {rows:?} pbc {periodic:?}: {d2} vs {want}"
            );
            // Open axes keep the component normal to the periodic span.
            for k in 0..3 {
                if !periodic[k] && periodic.iter().filter(|&&x| x).count() == 2 {
                    let normal = {
                        let (u, v) = match k {
                            0 => (rows[1], rows[2]),
                            1 => (rows[0], rows[2]),
                            _ => (rows[0], rows[1]),
                        };
                        [
                            u[1] * v[2] - u[2] * v[1],
                            u[2] * v[0] - u[0] * v[2],
                            u[0] * v[1] - u[1] * v[0],
                        ]
                    };
                    let raw = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                    let nn = dot(normal, normal).sqrt();
                    assert!((dot(d, normal) - dot(raw, normal)).abs() <= 1e-9 * nn * 100.0);
                }
            }
        }
    }

    #[test]
    fn orthorhombic_slab_matches_brute_force() {
        check(
            [[7.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.0]],
            [true, true, false],
        );
        check(
            [[7.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]],
            [true, false, true],
        );
    }

    #[test]
    fn skewed_slabs_match_brute_force() {
        // Hexagonal, a strongly sheared pair, and a plane tilted off xy.
        check(
            [
                [8.0, 0.0, 0.0],
                [-4.0, 6.928_203_230_275_51, 0.0],
                [0.0, 0.0, 30.0],
            ],
            [true, true, false],
        );
        check(
            [[6.0, 0.0, 0.0], [17.5, 2.0, 0.0], [0.0, 0.0, 0.0]],
            [true, true, false],
        );
        check(
            [[5.0, 1.0, 2.0], [-1.0, 6.0, 1.5], [0.0, 0.0, 0.0]],
            [true, true, false],
        );
    }

    #[test]
    fn wires_match_brute_force() {
        check(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 6.5]],
            [false, false, true],
        );
        check(
            [[3.0, 4.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [true, false, false],
        );
    }

    #[test]
    fn half_box_ties_keep_the_engine_vector() {
        let slab = PartialCell::from_ase(
            [[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 0.0]],
            [true, true, false],
        )
        .unwrap();
        // +L/2 wraps to -L/2 and -L/2 stays, as in the orthorhombic wrap.
        let d = slab.displacement_euclidean([0.0; 3], [5.0, 4.0, 2.0]);
        assert_eq!(d, [-5.0, -4.0, 2.0]);
        let d = slab.displacement_euclidean([0.0; 3], [-5.0, -4.0, 2.0]);
        assert_eq!(d, [-5.0, -4.0, 2.0]);
        let wire =
            PartialCell::from_ase([[0.0; 3], [0.0; 3], [0.0, 0.0, 6.0]], [false, false, true])
                .unwrap();
        assert_eq!(
            wire.displacement_euclidean([0.0; 3], [1.0, 2.0, 3.0]),
            [1.0, 2.0, -3.0]
        );
        let ortho = Cell::ortho(10.0, 8.0, 6.0).unwrap();
        assert_eq!(
            ortho.displacement_euclidean([0.0; 3], [5.0, 4.0, 3.0]),
            [-5.0, -4.0, -3.0]
        );
    }

    #[test]
    fn all_periodic_is_the_cell_bit_for_bit() {
        for rows in [
            [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
            [[8.0, 0.0, 0.0], [3.0, 7.0, 0.0], [1.0, -2.0, 9.0]],
            [[6.0, 1.0, 0.5], [17.5, 2.0, 0.3], [0.2, 0.1, 5.0]],
        ] {
            let cell = Cell::from_ase(rows).unwrap();
            let all = PartialCell::from_ase(rows, [true; 3]).unwrap();
            assert_eq!(all.periodic_count(), 3);
            assert_eq!(all.full(), Some(&cell));
            for w in points(600, 40.0).chunks(2) {
                let a = all.displacement_euclidean(w[0], w[1]);
                let b = cell.displacement_euclidean(w[0], w[1]);
                assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn open_is_the_cartesian_difference() {
        let open = PartialCell::from_ase([[0.0; 3]; 3], [false; 3]).unwrap();
        assert_eq!(open.periodic_count(), 0);
        assert!(open.full().is_none());
        assert_eq!(
            open.displacement_euclidean([1.0, 2.0, 3.0], [-4.0, 9.0, 0.5]),
            [-5.0, 7.0, -2.5]
        );
        assert_eq!(open.dist2_euclidean([0.0; 3], [3.0, 4.0, 0.0]), 25.0);
    }

    #[test]
    fn bad_periodic_vectors_are_refused() {
        let zero = [[0.0; 3], [0.0, 8.0, 0.0], [0.0; 3]];
        assert_eq!(
            PartialCell::from_ase(zero, [true, true, false]),
            Err(Error::BadBox)
        );
        let parallel = [[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0; 3]];
        assert_eq!(
            PartialCell::from_ase(parallel, [true, true, false]),
            Err(Error::BadBox)
        );
        let nan = [[f64::NAN, 0.0, 0.0], [0.0; 3], [0.0; 3]];
        assert_eq!(
            PartialCell::from_ase(nan, [true, false, false]),
            Err(Error::BadBox)
        );
        assert_eq!(
            PartialCell::from_ase(zero, [true, false, false]),
            Err(Error::BadBox)
        );
    }
}
