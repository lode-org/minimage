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
//! The closest point starts from Babai's rounding (*Combinatorica* **6**,
//! 1, 1986) in three of the four superbasis vectors, then runs the
//! iterative slicer of Sommer, Feder, and Shalvi (*SIAM J. Discrete
//! Math.* **23**, 715, 2009): while some Voronoi-relevant vector `r`
//! has `2 |x · r| > |r|^2`, step `x` by `r` toward the origin. Every
//! Voronoi-relevant vector of a lattice with an obtuse superbasis is
//! one of the fourteen `±v_S`, `S` a nonempty proper subset of the four
//! (Conway and Sloane, 1992), seven up to sign. Each step shortens `x`,
//! so the walk stops, and where it stops `x` is inside the Voronoi cell:
//! the shortest vector of its coset. McKilliam, Grant, and Clarkson
//! (*SIAM J. Discrete Math.* **28**, 1405, 2014) search the same sixteen
//! subset sums; seven dot products and a reduced starting point replace
//! their three rounds of fifteen candidates.

use std::cell::RefCell;

use crate::kernel::{self, invert_columns};

/// Selling superbasis for one lattice. The far Euclidean query builds
/// it once per calling thread and keeps it in a cache keyed by H.
/// Constructors store the cell only. An orthorhombic wrap is already
/// Euclidean and never enters the cache.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obtuse {
    v: [[f64; 3]; 4],
    /// Voronoi-relevant vectors up to sign, by component: `v0, v1, v2,
    /// v3, v0 + v1, v0 + v2, v0 + v3`, then a zero pad. Every other
    /// `±v_S` is one of these negated.
    rel: [[f64; 8]; 3],
    /// `|r|^2 / 2` plus a rounding margin, the slicer's step threshold.
    /// The pad is infinite, so it never steps.
    half_n2: [f64; 8],
    /// `r_j · r_k`. A step by `r_j` moves every `x · r_k` by row `j`.
    gram: [[f64; 8]; 8],
    inv: [[f64; 3]; 3],
    idx: [u8; 3],
    pub ok: bool,
}

impl PartialEq for Obtuse {
    fn eq(&self, other: &Self) -> bool {
        self.ok == other.ok && self.idx == other.idx && self.v == other.v && self.inv == other.inv
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
    let scale = (kernel::n2(a) + kernel::n2(b) + kernel::n2(c)).max(1.0);
    let tol = 1e-10 * scale;
    if already_obtuse(&original, tol) {
        return finish(original);
    }
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
    // 64 is past the handful of Delone steps left after size reduction.
    // 1000 matches the spglib / Andrews–Bernstein–Sauter cap if that
    // precondition did not shorten the basis.
    for budget in [64usize, 1000] {
        let done = selling_steps(&mut v, tol, budget);
        if done {
            return finish(v);
        }
    }
    unfinished(original)
}

fn unfinished(v: [[f64; 3]; 4]) -> Obtuse {
    let (rel, half_n2, gram) = relevant_of(v);
    Obtuse {
        v,
        rel,
        half_n2,
        gram,
        inv: [[0.0; 3]; 3],
        idx: [0, 1, 2],
        ok: false,
    }
}

fn already_obtuse(v: &[[f64; 3]; 4], tol: f64) -> bool {
    for i in 0..3 {
        for j in (i + 1)..4 {
            if kernel::dot(v[i], v[j]) > tol {
                return false;
            }
        }
    }
    true
}

/// The seven `v_S` classes and their step thresholds. The margin is far
/// above the rounding of `x · r` for a reduced `x`, so a point on a
/// Voronoi face does not step back and forth.
type Relevant = ([[f64; 8]; 3], [f64; 8], [[f64; 8]; 8]);

fn relevant_of(v: [[f64; 3]; 4]) -> Relevant {
    let vecs = [
        v[0],
        v[1],
        v[2],
        v[3],
        kernel::add(v[0], v[1]),
        kernel::add(v[0], v[2]),
        kernel::add(v[0], v[3]),
        [0.0; 3],
    ];
    let mut rel = [[0.0; 8]; 3];
    let mut half_n2 = [f64::INFINITY; 8];
    let mut gram = [[0.0; 8]; 8];
    for (k, r) in vecs.iter().enumerate() {
        rel[0][k] = r[0];
        rel[1][k] = r[1];
        rel[2][k] = r[2];
        if k < 7 {
            half_n2[k] = 0.5 * kernel::n2(*r) * (1.0 + 1e-12);
        }
        for (j, q) in vecs.iter().enumerate() {
            gram[k][j] = kernel::dot(*r, *q);
        }
    }
    (rel, half_n2, gram)
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
                let q = kernel::round_away(kernel::dot(v[i], v[j]) / lj);
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
        Some((idx, inv)) if best_abs > 1e-18 => {
            let (rel, half_n2, gram) = relevant_of(v);
            Obtuse {
                v,
                rel,
                half_n2,
                gram,
                inv,
                idx,
                ok: true,
            }
        }
        _ => unfinished(v),
    }
}

/// Cartesian `y - v`, `v` a closest lattice point.
pub(crate) fn closest_displacement(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    if s.ok {
        slice(s, y)
    } else {
        shell(s.v[0], s.v[1], s.v[2], y)
    }
}

struct CachedBasis {
    h: [[f64; 3]; 3],
    s: Obtuse,
    /// The engine wrap of this `H` is a short start for the slicer.
    engine_start: bool,
}

/// Restricted with the tilts inside half an edge (GROMACS
/// `correct_box`): the engine parallelepiped then sits within a step or
/// two of the Voronoi cell, and the engine wrap replaces Babai's point.
fn engine_is_short(h: [[f64; 3]; 3]) -> bool {
    let [a, b, c] = h;
    let scale = (a[0].abs() + b[1].abs() + c[2].abs()).max(1.0);
    let tol = 1e-10 * scale;
    a[1].abs() <= tol
        && a[2].abs() <= tol
        && b[2].abs() <= tol
        && b[0].abs() <= 0.5 * a[0].abs() + tol
        && c[0].abs() <= 0.5 * a[0].abs() + tol
        && c[1].abs() <= 0.5 * b[1].abs() + tol
}

struct BasisCache {
    slots: [Option<CachedBasis>; 4],
    hand: usize,
}

thread_local! {
    static BASIS: RefCell<BasisCache> = const {
        RefCell::new(BasisCache {
            slots: [None, None, None, None],
            hand: 0,
        })
    };
}

/// Closest displacement on the Selling superbasis of `h`.
///
/// The first query for a lattice builds the superbasis. The next
/// three distinct lattices stay cached on this thread; a fifth
/// replaces the oldest. A hit compares the nine components of H.
///
/// `y` is the raw difference and `wrapped` its engine wrap; both are in
/// the same coset.
pub(crate) fn closest_for(h: [[f64; 3]; 3], y: [f64; 3], wrapped: [f64; 3]) -> [f64; 3] {
    BASIS.with(|cache| {
        let mut cache = cache.borrow_mut();
        #[allow(clippy::manual_flatten)]
        for slot in &cache.slots {
            if let Some(slot) = slot {
                if same_h(&slot.h, &h) {
                    return slot.closest(y, wrapped);
                }
            }
        }
        let slot = CachedBasis {
            h,
            s: obtuse_superbasis(h[0], h[1], h[2]),
            engine_start: engine_is_short(h),
        };
        let d = slot.closest(y, wrapped);
        let i = cache.hand;
        cache.hand = (i + 1) % cache.slots.len();
        cache.slots[i] = Some(slot);
        d
    })
}

impl CachedBasis {
    #[inline(always)]
    fn closest(&self, y: [f64; 3], wrapped: [f64; 3]) -> [f64; 3] {
        if self.engine_start && self.s.ok {
            descend(&self.s, wrapped)
        } else {
            closest_displacement(&self.s, y)
        }
    }
}

/// Nine components equal, compared without an early exit so the test
/// is a few packed compares.
#[inline(always)]
fn same_h(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
    let mut same = true;
    for (ca, cb) in a.iter().zip(b) {
        for (x, y) in ca.iter().zip(cb) {
            same &= x == y;
        }
    }
    same
}

/// Babai's point in the reduced basis, then slicer steps until no
/// relevant vector shortens the residual.
///
/// A reduced basis leaves zero or one step, about evenly, so a branch on
/// it would miss half the time. The first step always runs and is
/// scaled to zero when nothing is violated; the dot products after it
/// come from one Gram row. Only a second violation loops, and the cap is
/// far past what a reduced basis leaves.
fn slice(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    descend(s, babai(s, y))
}

/// Nearest-integer rounding in three of the superbasis vectors.
#[inline(always)]
fn babai(s: &Obtuse, y: [f64; 3]) -> [f64; 3] {
    let c = kernel::mul(s.inv, y);
    let b0 = s.v[s.idx[0] as usize];
    let b1 = s.v[s.idx[1] as usize];
    let b2 = s.v[s.idx[2] as usize];
    let z0 = kernel::round_away(c[0]);
    let z1 = kernel::round_away(c[1]);
    let z2 = kernel::round_away(c[2]);
    [
        y[0] - (z0 * b0[0] + z1 * b1[0] + z2 * b2[0]),
        y[1] - (z0 * b0[1] + z1 * b1[1] + z2 * b2[1]),
        y[2] - (z0 * b0[2] + z1 * b1[2] + z2 * b2[2]),
    ]
}

/// Slicer steps from `x` until no relevant vector shortens it.
#[inline(always)]
fn descend(s: &Obtuse, mut x: [f64; 3]) -> [f64; 3] {
    let mut t = [0.0; 8];
    for (k, slot) in t.iter_mut().enumerate() {
        *slot = x[0] * s.rel[0][k] + x[1] * s.rel[1][k] + x[2] * s.rel[2][k];
    }
    for _ in 0..64 {
        let (best, gain) = most_violated(&t, &s.half_n2);
        let f = if gain > 0.0 {
            if t[best] > 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        };
        x[0] -= f * s.rel[0][best];
        x[1] -= f * s.rel[1][best];
        x[2] -= f * s.rel[2][best];
        for (slot, g) in t.iter_mut().zip(s.gram[best]) {
            *slot -= f * g;
        }
        if !any_violated(&t, &s.half_n2) {
            break;
        }
    }
    x
}

/// Index and size of the largest `|t_k| - h_k`. A three-level tree of
/// selects, so the latency is three compares rather than six.
#[inline(always)]
fn most_violated(t: &[f64; 8], h: &[f64; 8]) -> (usize, f64) {
    let mut g = [0.0; 8];
    for ((slot, tk), hk) in g.iter_mut().zip(t).zip(h) {
        *slot = tk.abs() - hk;
    }
    let (i01, g01) = larger(0, g[0], 1, g[1]);
    let (i23, g23) = larger(2, g[2], 3, g[3]);
    let (i45, g45) = larger(4, g[4], 5, g[5]);
    let (i03, g03) = larger(i01, g01, i23, g23);
    let (i46, g46) = larger(i45, g45, 6, g[6]);
    larger(i03, g03, i46, g46)
}

/// Whether some `|t_k|` passes its threshold. Compares only, no index.
#[inline(always)]
fn any_violated(t: &[f64; 8], h: &[f64; 8]) -> bool {
    let mut any = false;
    for (tk, hk) in t.iter().zip(h) {
        any |= tk.abs() > *hk;
    }
    any
}

#[inline(always)]
fn larger(i: usize, gi: f64, j: usize, gj: f64) -> (usize, f64) {
    let take = usize::from(gj > gi).wrapping_neg();
    (i ^ ((i ^ j) & take), if gj > gi { gj } else { gi })
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
    fn slicer_matches_a_reduced_shell_on_faces_and_skews() {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for cell in 0..300 {
            let a = [3.0 + next(), next(), next()];
            let mut b = [next(), 3.0 + next(), next()];
            let c = [next(), next(), 3.0 + next()];
            if cell % 3 == 0 {
                let t = 1e-3 * (1.0 + next());
                b = [a[0] + t * next(), a[1] + t * next(), a[2] + t];
            }
            if invert_columns([a, b, c]).is_none() {
                continue;
            }
            let s = obtuse_superbasis(a, b, c);
            assert!(s.ok);
            let mut ys = Vec::new();
            for _ in 0..30 {
                ys.push([8.0 * next(), 8.0 * next(), 8.0 * next()]);
            }
            for k in 0..7 {
                let r = [s.rel[0][k], s.rel[1][k], s.rel[2][k]];
                for f in [0.5 - 1e-9, 0.5, 0.5 + 1e-9, -0.5] {
                    ys.push(kernel::scale(f, r));
                }
            }
            for y in ys {
                let got = kernel::n2(closest_displacement(&s, y));
                let v = s.v;
                let mut oracle = f64::INFINITY;
                let base = slice(&s, y);
                for i in -3i32..=3 {
                    for j in -3i32..=3 {
                        for k in -3i32..=3 {
                            let p = [
                                f64::from(i) * v[0][0]
                                    + f64::from(j) * v[1][0]
                                    + f64::from(k) * v[2][0],
                                f64::from(i) * v[0][1]
                                    + f64::from(j) * v[1][1]
                                    + f64::from(k) * v[2][1],
                                f64::from(i) * v[0][2]
                                    + f64::from(j) * v[1][2]
                                    + f64::from(k) * v[2][2],
                            ];
                            let d = [base[0] - p[0], base[1] - p[1], base[2] - p[2]];
                            oracle = oracle.min(kernel::n2(d));
                        }
                    }
                }
                assert!(
                    got <= oracle + 1e-9 * (1.0 + oracle),
                    "cell {cell}: slicer {got} longer than shell {oracle}"
                );
            }
        }
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
