//! Obtuse superbasis and the closest lattice point in three dimensions.
//!
//! Every lattice of dimension at most 3 has an obtuse superbasis
//! (Selling, *J. Math. Pures Appl.* 1874; Conway and Sloane, *Proc. R.
//! Soc. Lond. A* **436**, 55, 1992): four vectors that sum to zero and
//! whose pairwise dots are not positive. The Voronoi vectors are the
//! `{-1,0,1}` combinations of that superbasis. Nearest-integer rounding
//! in a Minkowski basis is not that certificate (Nguyen and Stehlé,
//! *ACM Trans. Algorithms* **5**, 46, 2009, reduces the basis; it does
//! not decide the closest image).
//!
//! The Delone step is the International Tables reduction, the same
//! transformation as spglib `delaunay_reduce_basis`: while a pair has
//! a positive dot, add that vector to the other two and flip it.
//! A near-parallel basis makes that one-vector step linear in the
//! reciprocal of the angle (a shear of `1e-4` is about a thousand
//! iterations). Lagrange's size reduction (1773; Gauss, 1801) subtracts
//! the rounded projection of each longer edge onto each shorter one
//! first. The lattice is unchanged. The Delone step then finishes in a
//! handful of iterations, which is the bound spglib's
//! `delaunay_reduce_basis` relies on for an already short basis
//! (Andrews, Bernstein, and Sauter, *Acta Cryst.* A **75**, 115, 2019,
//! cap their Selling loop at 1000).
//!
//! McKilliam, Grant, and Clarkson, *SIAM J. Discrete Math.* **28**, 1405
//! (2014): write the target in the superbasis, start at the component
//! floor, and at most three times add the `{0,1}^4` step that most
//! shortens the residual. Dimension 3 has sixteen such steps, so the
//! minimum cut of the paper is this enumeration. The series reaches a
//! closest lattice point.

use crate::kernel::{self, invert_columns};

/// Selling superbasis. `prepared` is false on an orthorhombic cell,
/// which never consults it: the per-axis wrap is already Euclidean.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obtuse {
    v: [[f64; 3]; 4],
    inv: [[f64; 3]; 3],
    idx: [u8; 3],
    pub ok: bool,
    pub prepared: bool,
}

impl PartialEq for Obtuse {
    fn eq(&self, other: &Self) -> bool {
        self.ok == other.ok
            && self.prepared == other.prepared
            && self.idx == other.idx
            && self.v == other.v
            && self.inv == other.inv
    }
}

impl Obtuse {
    pub(crate) const fn empty() -> Self {
        Self {
            v: [[0.0; 3]; 4],
            inv: [[0.0; 3]; 3],
            idx: [0, 1, 2],
            ok: false,
            prepared: false,
        }
    }
}

/// Delaunay-reduce columns `a, b, c` to an obtuse superbasis.
pub(crate) fn obtuse_superbasis(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Obtuse {
    // Shell fallback stays on the caller's basis. A Selling iterate
    // that stopped early is not that basis.
    let original = [
        a,
        b,
        c,
        kernel::scale(-1.0, kernel::add(kernel::add(a, b), c)),
    ];
    let reduced = lagrange_reduce([a, b, c]);
    let mut v = [
        reduced[0],
        reduced[1],
        reduced[2],
        kernel::scale(
            -1.0,
            kernel::add(kernel::add(reduced[0], reduced[1]), reduced[2]),
        ),
    ];
    let scale = (kernel::n2(a) + kernel::n2(b) + kernel::n2(c)).max(1.0);
    let tol = 1e-10 * scale;
    // 64 is past the handful of Delone steps left after size reduction.
    // 1000 matches the spglib / Andrews–Bernstein–Sauter cap if that
    // precondition did not shorten the basis.
    for budget in [64usize, 1000] {
        let done = selling_steps(&mut v, tol, budget);
        if done {
            return finish(v);
        }
    }
    Obtuse {
        v: original,
        inv: [[0.0; 3]; 3],
        idx: [0, 1, 2],
        ok: false,
        prepared: true,
    }
}

/// Subtract rounded projections onto shorter edges. Each update is
/// unimodular, so the lattice is the one the caller passed.
fn lagrange_reduce(mut v: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    for _ in 0..24 {
        for i in 1..3 {
            let mut j = i;
            while j > 0 && kernel::n2(v[j]) < kernel::n2(v[j - 1]) {
                v.swap(j, j - 1);
                j -= 1;
            }
        }
        let mut changed = false;
        for i in 1..3 {
            for j in (0..i).rev() {
                let lj = kernel::n2(v[j]);
                if lj < 1e-30 {
                    continue;
                }
                let q = (kernel::dot(v[i], v[j]) / lj).round();
                if q != 0.0 {
                    v[i] = kernel::add(v[i], kernel::scale(-q, v[j]));
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    v
}

/// One Delone step per positive pair. `true` when every pairwise dot
/// is at most `tol`.
fn selling_steps(v: &mut [[f64; 3]; 4], tol: f64, budget: usize) -> bool {
    for _ in 0..budget {
        let mut step_at = None;
        'pairs: for i in 0..3 {
            for j in (i + 1)..4 {
                if kernel::dot(v[i], v[j]) > tol {
                    step_at = Some((i, j));
                    break 'pairs;
                }
            }
        }
        let Some((i, j)) = step_at else {
            return true;
        };
        let step = v[i];
        for (k, slot) in v.iter_mut().enumerate() {
            if k != i && k != j {
                *slot = kernel::add(*slot, step);
            }
        }
        v[i] = kernel::scale(-1.0, step);
    }
    false
}

fn finish(v: [[f64; 3]; 4]) -> Obtuse {
    let mut best_abs = 0.0;
    let mut best: Option<([u8; 3], [[f64; 3]; 3])> = None;
    for drop_at in 0..4 {
        let mut idx = [0u8; 3];
        let mut n = 0;
        for t in 0..4 {
            if t != drop_at {
                idx[n] = t as u8;
                n += 1;
            }
        }
        let cols = [v[idx[0] as usize], v[idx[1] as usize], v[idx[2] as usize]];
        if let Some((inv, det)) = invert_columns(cols) {
            if det.abs() > best_abs {
                best_abs = det.abs();
                best = Some((idx, inv));
            }
        }
    }
    match best {
        Some((idx, inv)) if best_abs > 1e-18 => Obtuse {
            v,
            inv,
            idx,
            ok: true,
            prepared: true,
        },
        _ => Obtuse {
            v,
            inv: [[0.0; 3]; 3],
            idx: [0, 1, 2],
            ok: false,
            prepared: true,
        },
    }
}

/// Cartesian `y - v`, `v` a closest lattice point.
pub(crate) fn closest_displacement(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    if s.ok {
        mckilliam(s, y)
    } else {
        shell(s.v[0], s.v[1], s.v[2], y)
    }
}

fn mckilliam(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    let c = kernel::mul(s.inv, y);
    let mut z = [0.0; 4];
    z[s.idx[0] as usize] = c[0];
    z[s.idx[1] as usize] = c[1];
    z[s.idx[2] as usize] = c[2];
    let mut u = [z[0].floor(), z[1].floor(), z[2].floor(), z[3].floor()];
    let mut best_d = [0.0; 3];
    for _ in 0..3 {
        let mut best_mask = 0u8;
        let mut best_d2 = f64::INFINITY;
        best_d = [0.0; 3];
        for mask in 0..16u8 {
            let mut coeff = u;
            for (bit, c) in coeff.iter_mut().enumerate() {
                if mask & (1 << bit) != 0 {
                    *c += 1.0;
                }
            }
            let p = point(&s.v, coeff);
            let d = [y[0] - p[0], y[1] - p[1], y[2] - p[2]];
            let d2 = kernel::n2(d);
            if d2 < best_d2 {
                best_d2 = d2;
                best_mask = mask;
                best_d = d;
            }
        }
        if best_mask == 0 {
            break;
        }
        for (bit, u_bit) in u.iter_mut().enumerate() {
            if best_mask & (1 << bit) != 0 {
                *u_bit += 1.0;
            }
        }
    }
    best_d
}

fn point(v: &[[f64; 3]; 4], coeff: [f64; 4]) -> [f64; 3] {
    let mut p = [0.0; 3];
    for i in 0..4 {
        p[0] += coeff[i] * v[i][0];
        p[1] += coeff[i] * v[i][1];
        p[2] += coeff[i] * v[i][2];
    }
    p
}

/// Widening `{-r..r}^3` on the original basis. Used only when Selling
/// does not finish; a reduced superbasis never calls it.
fn shell(a: [f64; 3], b: [f64; 3], c: [f64; 3], y: [f64; 3]) -> [f64; 3] {
    let mut best = y;
    let mut best2 = kernel::n2(y);
    for r in 0i32..=16 {
        let lo = -r;
        let hi = r;
        for i in lo..=hi {
            for j in lo..=hi {
                for k in lo..=hi {
                    if r > 0 && i.abs() != r && j.abs() != r && k.abs() != r {
                        continue;
                    }
                    let p = [
                        f64::from(i) * a[0] + f64::from(j) * b[0] + f64::from(k) * c[0],
                        f64::from(i) * a[1] + f64::from(j) * b[1] + f64::from(k) * c[1],
                        f64::from(i) * a[2] + f64::from(j) * b[2] + f64::from(k) * c[2],
                    ];
                    let d = [y[0] - p[0], y[1] - p[1], y[2] - p[2]];
                    let d2 = kernel::n2(d);
                    if d2 < best2 {
                        best2 = d2;
                        best = d;
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(a: [f64; 3], b: [f64; 3], c: [f64; 3], y: [f64; 3]) -> f64 {
        shell(a, b, c, y).iter().map(|e| e * e).sum()
    }

    #[test]
    fn ortho_superbasis_is_obtuse() {
        let a = [10.0, 0.0, 0.0];
        let b = [0.0, 11.0, 0.0];
        let c = [0.0, 0.0, 12.0];
        let s = obtuse_superbasis(a, b, c);
        assert!(s.ok);
        let y = [25.0, -18.0, 3.0];
        let d = closest_displacement(&s, y);
        let d2 = kernel::n2(d);
        // Two cells along a and two back along b: residual (5, 4, 3).
        assert!((d2 - brute(a, b, c, y)).abs() < 1e-8);
        assert!((d2 - 50.0).abs() < 1e-8);
    }

    #[test]
    fn lattice_vector_is_zero() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.99, 0.01, 0.0];
        let c = [0.0, 0.0, 1.0];
        let s = obtuse_superbasis(a, b, c);
        assert!(s.ok, "Selling finishes on the skewed cell");
        let y = [0.02, -0.02, 0.0];
        let d = closest_displacement(&s, y);
        assert!(kernel::n2(d) < 1e-20, "2(a-b) is a lattice vector");
    }

    #[test]
    fn near_parallel_finishes() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0 - 1e-4, 1e-4, 0.0];
        let c = [0.2, -0.3, 1.5];
        let s = obtuse_superbasis(a, b, c);
        assert!(s.ok, "size reduction leaves a short Delone step");
        let y = [2e-4, -2e-4, 0.0];
        let d = closest_displacement(&s, y);
        assert!(kernel::n2(d) < 1e-18, "2(a-b) is a lattice vector");
        let probe = [3.5, -1.25, 4.0];
        let got = kernel::n2(closest_displacement(&s, probe));
        let oracle = brute(a, b, c, probe);
        assert!(got <= oracle + 1e-6 * (1.0 + oracle));
    }

    #[test]
    fn random_cells_match_shell() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f64 / f64::from(1u32 << 31);
            u * 6.0 - 3.0
        };
        for _ in 0..40 {
            let mut h = [[0.0; 3]; 3];
            for col in &mut h {
                *col = [next() + 4.0, next(), next()];
            }
            h[0][0] += 3.0;
            h[1][1] += 3.0;
            h[2][2] += 3.0;
            if invert_columns(h).is_none() {
                continue;
            }
            let s = obtuse_superbasis(h[0], h[1], h[2]);
            assert!(s.ok);
            let y = [next() * 8.0, next() * 8.0, next() * 8.0];
            let got = kernel::n2(closest_displacement(&s, y));
            let oracle = brute(h[0], h[1], h[2], y);
            assert!(
                got <= oracle + 1e-6 * (1.0 + oracle),
                "closest {got} longer than shell {oracle}"
            );
        }
    }
}
