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

use std::cell::RefCell;

use crate::kernel::{self, invert_columns};

/// Selling superbasis. `prepared` is false on an orthorhombic cell,
/// which never consults it: the per-axis wrap is already Euclidean.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obtuse {
    v: [[f64; 3]; 4],
    inv: [[f64; 3]; 3],
    /// Subset sums of `v`, structure of arrays. Slot 15 is unused.
    ox: [f64; 16],
    oy: [f64; 16],
    oz: [f64; 16],
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
            && self.ox == other.ox
            && self.oy == other.oy
            && self.oz == other.oz
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
        ox: [0.0; 16],
        oy: [0.0; 16],
        oz: [0.0; 16],
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
        Some((idx, inv)) if best_abs > 1e-18 => pack(Obtuse {
            v,
            inv,
            ox: [0.0; 16],
            oy: [0.0; 16],
            oz: [0.0; 16],
            idx,
            ok: true,
            prepared: true,
        }),
        _ => Obtuse {
            v,
            inv: [[0.0; 3]; 3],
            ox: [0.0; 16],
            oy: [0.0; 16],
            oz: [0.0; 16],
            idx: [0, 1, 2],
            ok: false,
            prepared: true,
        },
    }
}

thread_local! {
    /// Last Selling superbasis, keyed by the nine entries of H.
    /// Construction stays the inverse and the widths. The first
    /// Euclidean query for a lattice builds the superbasis; the next
    /// query on this thread reuses it.
    static CACHED_OBTUSE: RefCell<Option<([f64; 9], Obtuse)>> = const { RefCell::new(None) };
}

/// Closest displacement on the thread-cached superbasis of `h`.
pub(crate) fn cached_closest(h: [[f64; 3]; 3], y: [f64; 3]) -> [f64; 3] {
    let key = [
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    ];
    CACHED_OBTUSE.with(|slot| {
        {
            let borrow = slot.borrow();
            if let Some((cached, superbasis)) = borrow.as_ref() {
                if *cached == key {
                    return closest_displacement(superbasis, y);
                }
            }
        }
        let superbasis = obtuse_superbasis(h[0], h[1], h[2]);
        let displacement = closest_displacement(&superbasis, y);
        *slot.borrow_mut() = Some((key, superbasis));
        displacement
    })
}

/// Cartesian `y - v`, `v` a closest lattice point.
pub(crate) fn closest_displacement(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    if s.ok {
        mckilliam(s, y)
    } else {
        shell(s.v[0], s.v[1], s.v[2], y)
    }
}

/// Fifteen subset sums of the superbasis, structure of arrays.
/// Slot 15 is left enormous so a 16-wide score cannot select it:
/// the four vectors sum to zero, and mask 15 is the same point as mask 0.
fn subset_soa(v: &[[f64; 3]; 4]) -> ([f64; 16], [f64; 16], [f64; 16]) {
    let mut x = [0.0; 16];
    let mut y = [0.0; 16];
    let mut z = [0.0; 16];
    for mask in 1..15 {
        if mask & 1 != 0 {
            x[mask] += v[0][0];
            y[mask] += v[0][1];
            z[mask] += v[0][2];
        }
        if mask & 2 != 0 {
            x[mask] += v[1][0];
            y[mask] += v[1][1];
            z[mask] += v[1][2];
        }
        if mask & 4 != 0 {
            x[mask] += v[2][0];
            y[mask] += v[2][1];
            z[mask] += v[2][2];
        }
        if mask & 8 != 0 {
            x[mask] += v[3][0];
            y[mask] += v[3][1];
            z[mask] += v[3][2];
        }
    }
    x[15] = 1.0e300;
    y[15] = 1.0e300;
    z[15] = 1.0e300;
    (x, y, z)
}

fn pack(mut s: Obtuse) -> Obtuse {
    let (ox, oy, oz) = subset_soa(&s.v);
    s.ox = ox;
    s.oy = oy;
    s.oz = oz;
    s
}

fn mckilliam(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    let c = kernel::mul(s.inv, y);
    let mut z = [0.0; 4];
    z[s.idx[0] as usize] = c[0];
    z[s.idx[1] as usize] = c[1];
    z[s.idx[2] as usize] = c[2];
    let mut u = [z[0].floor(), z[1].floor(), z[2].floor(), z[3].floor()];
    let v = &s.v;
    let mut best_d = [0.0; 3];
    for _ in 0..3 {
        let mut base = [0.0; 3];
        for i in 0..4 {
            let ui = u[i];
            base[0] += ui * v[i][0];
            base[1] += ui * v[i][1];
            base[2] += ui * v[i][2];
        }
        let rx = y[0] - base[0];
        let ry = y[1] - base[1];
        let rz = y[2] - base[2];
        let (best_mask, _best_d2, disp) = score_steps(rx, ry, rz, &s.ox, &s.oy, &s.oz);
        best_d = disp;
        if best_mask == 0 {
            break;
        }
        if best_mask & 1 != 0 {
            u[0] += 1.0;
        }
        if best_mask & 2 != 0 {
            u[1] += 1.0;
        }
        if best_mask & 4 != 0 {
            u[2] += 1.0;
        }
        if best_mask & 8 != 0 {
            u[3] += 1.0;
        }
    }
    best_d
}

/// Strict `<`, so an equal length keeps the lower mask. Mask 0 is the
/// initial champion.
fn score_steps(
    rx: f64,
    ry: f64,
    rz: f64,
    ox: &[f64; 16],
    oy: &[f64; 16],
    oz: &[f64; 16],
) -> (u8, f64, [f64; 3]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected. The three tables have length 16.
            return unsafe { score_steps_avx(rx, ry, rz, ox, oy, oz) };
        }
    }
    score_steps_scalar(rx, ry, rz, ox, oy, oz)
}

fn score_steps_scalar(
    rx: f64,
    ry: f64,
    rz: f64,
    ox: &[f64; 16],
    oy: &[f64; 16],
    oz: &[f64; 16],
) -> (u8, f64, [f64; 3]) {
    let mut best_mask = 0u8;
    let mut best_d2 = rx * rx + ry * ry + rz * rz;
    let mut best_d = [rx, ry, rz];
    for mask in 1..15 {
        let dx = rx - ox[mask];
        let dy = ry - oy[mask];
        let dz = rz - oz[mask];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best_mask = mask as u8;
            best_d = [dx, dy, dz];
        }
    }
    (best_mask, best_d2, best_d)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn score_steps_avx(
    rx: f64,
    ry: f64,
    rz: f64,
    ox: &[f64; 16],
    oy: &[f64; 16],
    oz: &[f64; 16],
) -> (u8, f64, [f64; 3]) {
    use std::arch::x86_64::*;
    let vx = _mm256_set1_pd(rx);
    let vy = _mm256_set1_pd(ry);
    let vz = _mm256_set1_pd(rz);
    let mut best_d2v = _mm256_set1_pd(f64::INFINITY);
    let mut best_iv = _mm256_set1_pd(0.0);
    let mut mask = 0usize;
    while mask < 16 {
        let dx = _mm256_sub_pd(vx, _mm256_loadu_pd(ox.as_ptr().add(mask)));
        let dy = _mm256_sub_pd(vy, _mm256_loadu_pd(oy.as_ptr().add(mask)));
        let dz = _mm256_sub_pd(vz, _mm256_loadu_pd(oz.as_ptr().add(mask)));
        let d2 = _mm256_add_pd(
            _mm256_mul_pd(dx, dx),
            _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)),
        );
        let ids = _mm256_set_pd(
            (mask + 3) as f64,
            (mask + 2) as f64,
            (mask + 1) as f64,
            mask as f64,
        );
        let lt = _mm256_cmp_pd(d2, best_d2v, _CMP_LT_OQ);
        best_d2v = _mm256_blendv_pd(best_d2v, d2, lt);
        best_iv = _mm256_blendv_pd(best_iv, ids, lt);
        mask += 4;
    }
    let mut lane_d2 = [0.0; 4];
    let mut lane_i = [0.0; 4];
    _mm256_storeu_pd(lane_d2.as_mut_ptr(), best_d2v);
    _mm256_storeu_pd(lane_i.as_mut_ptr(), best_iv);
    let mut best_mask = 0u8;
    let mut best_d2 = rx * rx + ry * ry + rz * rz;
    for k in 0..4 {
        let id = lane_i[k] as u8;
        if id == 0 || id >= 15 {
            continue;
        }
        if lane_d2[k] < best_d2 {
            best_d2 = lane_d2[k];
            best_mask = id;
        }
    }
    let m = best_mask as usize;
    (best_mask, best_d2, [rx - ox[m], ry - oy[m], rz - oz[m]])
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
