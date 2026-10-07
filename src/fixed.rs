//! Fixed-point fractional coordinates.
//!
//! A position folded into the cell is three fractions in `[0, 1)`, each
//! stored on 64 bits as `round(s * 2^52) * 2^12`. The low 52 bits of
//! `s + 1` are that integer, exactly, so the conversion needs no
//! conversion instruction and no branch: a fraction that rounds up to
//! one lands on zero by itself. The engine wrap of a difference is
//! then exact integer arithmetic: `b - a` modulo `2^64`, read as signed,
//! is the wrapped fraction times `2^64` in `[-2^63, 2^63)`. Nothing is
//! rounded until the top 52 bits of that difference become a double,
//! and that conversion is exact. One product with `H` gives the
//! Cartesian vector, for every cell shape, with no `Hinv`, no rounding
//! step, and no branch. This is the exact integer split of the Ozaki
//! scheme (Ozaki, Ogita, Oishi, and Rump, *Numer. Algorithms* **59**, 95,
//! 2012) applied to the periodic wrap, and the fixed-point positions of
//! Anton (Shaw et al., *Commun. ACM* **51**, 91, 2008). The wrap is the
//! symmetric residue `a - m floor(a/m + 1/2)` of Ozaki Scheme II (Ozaki,
//! Uchino, and Imamura, arXiv:2504.08009, 2025) with the one modulus
//! `m = 2^64`; two's complement computes it, so no Chinese remainder
//! step is needed.
//!
//! A tie, a fraction exactly one half apart, wraps to `-1/2`. The
//! squared distance agrees with [`Cell::dist2`](crate::Cell::dist2) to a
//! few units in the last place.

use crate::kernel::{floor_fast, n2};

/// `2^-52`, one unit of the shifted difference.
pub(crate) const UNIT: f64 = 2.220_446_049_250_313e-16;

/// The 52 fraction bits of a double in `[1, 2)`.
const MANTISSA: u64 = (1 << 52) - 1;

/// A fraction `t` in `[0, 1]` on 64 fixed-point bits: the mantissa of
/// `t + 1`, which is `round(t * 2^52)`, shifted to the top. `t = 1`
/// gives `2.0`, whose mantissa is zero.
#[inline(always)]
pub(crate) fn to_fixed(t: f64) -> u64 {
    ((t + 1.0).to_bits() & MANTISSA) << 12
}

/// Cartesian to fractional for one cell: the inverse and the origin.
/// An orthorhombic cell keeps the reciprocals of its widths.
#[derive(Clone, Copy)]
pub(crate) struct Fold {
    pub(crate) inverse: Lattice,
    pub(crate) origin: [f64; 3],
}

impl Fold {
    /// `round(frac(s) * 2^52) * 2^12` per axis, `s` the fractional
    /// coordinates of `r`.
    #[inline(always)]
    pub(crate) fn fixed(&self, r: [f64; 3]) -> [u64; 3] {
        let d = [
            r[0] - self.origin[0],
            r[1] - self.origin[1],
            r[2] - self.origin[2],
        ];
        let s = match self.inverse {
            Lattice::Diagonal(inv) => [d[0] * inv[0], d[1] * inv[1], d[2] * inv[2]],
            Lattice::Full(m) => [
                m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
                m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
                m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
            ],
        };
        [
            to_fixed(s[0] - floor_fast(s[0])),
            to_fixed(s[1] - floor_fast(s[1])),
            to_fixed(s[2] - floor_fast(s[2])),
        ]
    }
}

/// `out[k] = fold.fixed(rs[k])`.
pub(crate) fn fixed_many(fold: &Fold, rs: &[[f64; 3]], out: &mut [[u64; 3]]) {
    #[cfg(target_arch = "x86_64")]
    {
        if rs.len() >= 4 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was detected. `rs` and `out` share one length.
            unsafe { avx2::fixed_many(fold, rs, out) };
            return;
        }
    }
    for (r, o) in rs.iter().zip(out.iter_mut()) {
        *o = fold.fixed(*r);
    }
}

/// The wrapped fraction `b - a` as a double, in units of `2^-52`.
///
/// The difference modulo `2^64` is the engine wrap. Its top 52 bits,
/// an arithmetic shift, fit a double exactly.
#[inline(always)]
pub(crate) fn wrapped(a: u64, b: u64) -> f64 {
    ((b.wrapping_sub(a) as i64) >> 12) as f64
}

/// `H` scaled by `2^-52`, so a wrapped difference maps straight to
/// Cartesian. An orthorhombic cell keeps its three widths.
#[derive(Clone, Copy)]
pub(crate) enum Lattice {
    Diagonal([f64; 3]),
    Full([[f64; 3]; 3]),
}

impl Lattice {
    pub(crate) fn diagonal(widths: [f64; 3]) -> Self {
        Lattice::Diagonal([widths[0] * UNIT, widths[1] * UNIT, widths[2] * UNIT])
    }

    pub(crate) fn full(h: [[f64; 3]; 3]) -> Self {
        let mut m = h;
        for col in &mut m {
            for e in col.iter_mut() {
                *e *= UNIT;
            }
        }
        Lattice::Full(m)
    }

    /// Cartesian displacement from `a` to `b`.
    #[inline(always)]
    pub(crate) fn displacement(&self, a: [u64; 3], b: [u64; 3]) -> [f64; 3] {
        let d = [
            wrapped(a[0], b[0]),
            wrapped(a[1], b[1]),
            wrapped(a[2], b[2]),
        ];
        match self {
            Lattice::Diagonal(w) => [w[0] * d[0], w[1] * d[1], w[2] * d[2]],
            Lattice::Full(m) => [
                m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
                m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
                m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
            ],
        }
    }

    #[inline(always)]
    pub(crate) fn dist2(&self, a: [u64; 3], b: [u64; 3]) -> f64 {
        n2(self.displacement(a, b))
    }
}

/// `out[k] = |b(qs[k]) - b(p)|^2` over fixed-point positions.
pub(crate) fn dist2_many(lat: Lattice, p: [u64; 3], qs: &[[u64; 3]], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if qs.len() >= 4 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was detected. `qs` and `out` share one length.
            unsafe { avx2::many(&lat, p, qs, out) };
            return;
        }
    }
    for (q, o) in qs.iter().zip(out.iter_mut()) {
        *o = lat.dist2(p, *q);
    }
}

/// `out[k] = |b(qs[k]) - b(ps[k])|^2` over fixed-point positions.
pub(crate) fn dist2_pairs(lat: Lattice, ps: &[[u64; 3]], qs: &[[u64; 3]], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if ps.len() >= 4 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was detected. The slices share one length.
            unsafe { avx2::pairs(&lat, ps, qs, out) };
            return;
        }
    }
    for ((p, q), o) in ps.iter().zip(qs).zip(out.iter_mut()) {
        *o = lat.dist2(*p, *q);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{Fold, Lattice};

    type V = __m256d;

    /// `super::wrapped` per lane. A logical shift leaves the signed top
    /// 52 bits offset by `2^52` when negative; xor with the bits of
    /// `1.5 * 2^52` and one subtraction recover them exactly.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn to_double(d: __m256i) -> V {
        let bits = _mm256_xor_si256(
            _mm256_srli_epi64(d, 12),
            _mm256_set1_epi64x(0x4338_0000_0000_0000),
        );
        _mm256_sub_pd(
            _mm256_castsi256_pd(bits),
            _mm256_set1_pd(6_755_399_441_055_744.0),
        )
    }

    /// Four packed rows of `[u64; 3]` differences as x, y, z doubles.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn transpose(a: __m256i, b: __m256i, c: __m256i) -> [V; 3] {
        let (a, b, c) = (to_double(a), to_double(b), to_double(c));
        let u = _mm256_blend_pd(a, b, 0b1100);
        let v = _mm256_permute2f128_pd(a, c, 0x21);
        let w = _mm256_blend_pd(b, c, 0b1100);
        [
            _mm256_shuffle_pd(u, v, 0b1010),
            _mm256_shuffle_pd(u, w, 0b0101),
            _mm256_shuffle_pd(v, w, 0b1010),
        ]
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dist2(lat: &Lattice, d: [V; 3]) -> V {
        let [x, y, z] = match lat {
            Lattice::Diagonal(w) => [
                _mm256_mul_pd(_mm256_set1_pd(w[0]), d[0]),
                _mm256_mul_pd(_mm256_set1_pd(w[1]), d[1]),
                _mm256_mul_pd(_mm256_set1_pd(w[2]), d[2]),
            ],
            Lattice::Full(m) => {
                let mut r = [_mm256_setzero_pd(); 3];
                for (i, slot) in r.iter_mut().enumerate() {
                    *slot = _mm256_add_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_set1_pd(m[0][i]), d[0]),
                            _mm256_mul_pd(_mm256_set1_pd(m[1][i]), d[1]),
                        ),
                        _mm256_mul_pd(_mm256_set1_pd(m[2][i]), d[2]),
                    );
                }
                r
            }
        };
        _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)),
            _mm256_mul_pd(z, z),
        )
    }

    /// `Fold::fixed` for four rows: transpose in, fold, floor, take the
    /// mantissa of `t + 1`, transpose the integers back out.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fixed_many(fold: &Fold, rs: &[[f64; 3]], out: &mut [[u64; 3]]) {
        let o = [
            _mm256_set1_pd(fold.origin[0]),
            _mm256_set1_pd(fold.origin[1]),
            _mm256_set1_pd(fold.origin[2]),
        ];
        let one = _mm256_set1_pd(1.0);
        let mask = _mm256_set1_epi64x(super::MANTISSA as i64);
        let n = rs.len();
        let mut i = 0;
        while i + 4 <= n {
            let src: *const f64 = rs.as_ptr().add(i).cast();
            let a = _mm256_loadu_pd(src);
            let b = _mm256_loadu_pd(src.add(4));
            let c = _mm256_loadu_pd(src.add(8));
            let u = _mm256_blend_pd(a, b, 0b1100);
            let v = _mm256_permute2f128_pd(a, c, 0x21);
            let w = _mm256_blend_pd(b, c, 0b1100);
            let d = [
                _mm256_sub_pd(_mm256_shuffle_pd(u, v, 0b1010), o[0]),
                _mm256_sub_pd(_mm256_shuffle_pd(u, w, 0b0101), o[1]),
                _mm256_sub_pd(_mm256_shuffle_pd(v, w, 0b1010), o[2]),
            ];
            let s = match fold.inverse {
                Lattice::Diagonal(inv) => [
                    _mm256_mul_pd(d[0], _mm256_set1_pd(inv[0])),
                    _mm256_mul_pd(d[1], _mm256_set1_pd(inv[1])),
                    _mm256_mul_pd(d[2], _mm256_set1_pd(inv[2])),
                ],
                Lattice::Full(m) => {
                    let mut r = [_mm256_setzero_pd(); 3];
                    for (k, slot) in r.iter_mut().enumerate() {
                        *slot = _mm256_add_pd(
                            _mm256_add_pd(
                                _mm256_mul_pd(_mm256_set1_pd(m[0][k]), d[0]),
                                _mm256_mul_pd(_mm256_set1_pd(m[1][k]), d[1]),
                            ),
                            _mm256_mul_pd(_mm256_set1_pd(m[2][k]), d[2]),
                        );
                    }
                    r
                }
            };
            let fx = |x: V| {
                let t = _mm256_add_pd(_mm256_sub_pd(x, _mm256_floor_pd(x)), one);
                _mm256_castsi256_pd(_mm256_slli_epi64(
                    _mm256_and_si256(_mm256_castpd_si256(t), mask),
                    12,
                ))
            };
            let (x, y, z) = (fx(s[0]), fx(s[1]), fx(s[2]));
            let u = _mm256_unpacklo_pd(x, y);
            let w = _mm256_unpackhi_pd(y, z);
            let v = _mm256_shuffle_pd(z, x, 0b1010);
            let dst: *mut f64 = out.as_mut_ptr().add(i).cast();
            _mm256_storeu_pd(dst, _mm256_permute2f128_pd(u, v, 0x20));
            _mm256_storeu_pd(dst.add(4), _mm256_blend_pd(w, u, 0b1100));
            _mm256_storeu_pd(dst.add(8), _mm256_permute2f128_pd(v, w, 0x31));
            i += 4;
        }
        for j in i..n {
            out[j] = fold.fixed(rs[j]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn many(lat: &Lattice, p: [u64; 3], qs: &[[u64; 3]], out: &mut [f64]) {
        let pa = _mm256_setr_epi64x(p[0] as i64, p[1] as i64, p[2] as i64, p[0] as i64);
        let pb = _mm256_setr_epi64x(p[1] as i64, p[2] as i64, p[0] as i64, p[1] as i64);
        let pc = _mm256_setr_epi64x(p[2] as i64, p[0] as i64, p[1] as i64, p[2] as i64);
        let n = qs.len();
        let mut i = 0;
        while i + 4 <= n {
            let src: *const __m256i = qs.as_ptr().add(i).cast();
            let d = transpose(
                _mm256_sub_epi64(_mm256_loadu_si256(src), pa),
                _mm256_sub_epi64(_mm256_loadu_si256(src.add(1)), pb),
                _mm256_sub_epi64(_mm256_loadu_si256(src.add(2)), pc),
            );
            _mm256_storeu_pd(out.as_mut_ptr().add(i), dist2(lat, d));
            i += 4;
        }
        for j in i..n {
            out[j] = lat.dist2(p, qs[j]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pairs(lat: &Lattice, ps: &[[u64; 3]], qs: &[[u64; 3]], out: &mut [f64]) {
        let n = ps.len();
        let mut i = 0;
        while i + 4 <= n {
            let q: *const __m256i = qs.as_ptr().add(i).cast();
            let p: *const __m256i = ps.as_ptr().add(i).cast();
            let d = transpose(
                _mm256_sub_epi64(_mm256_loadu_si256(q), _mm256_loadu_si256(p)),
                _mm256_sub_epi64(_mm256_loadu_si256(q.add(1)), _mm256_loadu_si256(p.add(1))),
                _mm256_sub_epi64(_mm256_loadu_si256(q.add(2)), _mm256_loadu_si256(p.add(2))),
            );
            _mm256_storeu_pd(out.as_mut_ptr().add(i), dist2(lat, d));
            i += 4;
        }
        for j in i..n {
            out[j] = lat.dist2(ps[j], qs[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Cell;

    fn cells() -> Vec<Cell> {
        vec![
            Cell::ortho(10.0, 11.0, 12.0).unwrap(),
            Cell::from_vectors(
                [10.0, 0.0, 0.0],
                [5.0, 8.660254037844386, 0.0],
                [0.0, 0.0, 10.0],
                [1.0, -2.0, 0.5],
            )
            .unwrap(),
            Cell::from_vectors(
                [9.0, 3.0, 2.0],
                [1.0, 9.0, -2.0],
                [-1.5, 2.0, 9.5],
                [0.0; 3],
            )
            .unwrap(),
            Cell::from_vectors(
                [-7.0, 0.0, 0.0],
                [2.5, 9.0, 0.0],
                [-1.0, 3.0, -11.0],
                [0.0; 3],
            )
            .unwrap(),
        ]
    }

    fn points(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let mut p = [0.0; 3];
                for e in &mut p {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *e = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 60.0;
                }
                p
            })
            .collect()
    }

    #[test]
    fn fixed_distance_matches_the_float_wrap() {
        for cell in cells() {
            let ps = points(500, 0x1f83_d9ab_fb41_bd6b);
            let qs = points(500, 0x5be0_cd19_137e_2179);
            for (p, q) in ps.iter().zip(&qs) {
                let (a, b) = (cell.fixed(*p), cell.fixed(*q));
                let want = cell.dist2(*p, *q);
                let got = cell.dist2_fixed(a, b);
                assert!((got - want).abs() <= 1e-13 * (1.0 + want), "{got} {want}");
                let d = cell.displacement_fixed(a, b);
                let e = cell.displacement(*p, *q);
                let dd = (d[0] - e[0]).powi(2) + (d[1] - e[1]).powi(2) + (d[2] - e[2]).powi(2);
                assert!(dd <= 1e-24 * (1.0 + want), "{d:?} {e:?}");
            }
        }
    }

    #[test]
    fn fixed_many_equals_the_per_position_call() {
        for cell in cells() {
            for n in [0usize, 1, 3, 4, 5, 64, 131] {
                let rs = points(n, 0x9b05_688c_2b3e_6c1f);
                let mut out = vec![[0u64; 3]; n];
                crate::fixed_many(&cell, &rs, &mut out).unwrap();
                for (r, got) in rs.iter().zip(&out) {
                    assert_eq!(*got, cell.fixed(*r));
                }
            }
        }
    }

    #[test]
    fn a_fraction_that_rounds_to_one_folds_to_zero() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let below = f64::from_bits(10.0f64.to_bits() - 1);
        assert_eq!(cell.fixed([-1e-300, 0.0, below]), [0, 0, 0]);
        assert_eq!(cell.fixed([5.0, 2.5, 10.0]), [1 << 63, 1 << 62, 0]);
    }

    #[test]
    fn fixed_tie_wraps_to_minus_half() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let a = cell.fixed([0.0, 0.0, 2.5]);
        let b = cell.fixed([5.0, 5.0, 2.5]);
        let d = cell.displacement_fixed(a, b);
        assert_eq!(d, [-5.0, -5.0, 0.0]);
        let d = cell.displacement_fixed(b, a);
        assert_eq!(d, [-5.0, -5.0, 0.0]);
    }

    #[test]
    fn fixed_batches_equal_the_per_pair_call() {
        for cell in cells() {
            for n in [0usize, 1, 3, 4, 7, 64, 129] {
                let ps: Vec<[u64; 3]> = points(n, 0x6a09_e667)
                    .iter()
                    .map(|r| cell.fixed(*r))
                    .collect();
                let qs: Vec<[u64; 3]> = points(n, 0xbb67_ae85)
                    .iter()
                    .map(|r| cell.fixed(*r))
                    .collect();
                let mut out = vec![0.0; n];
                crate::dist2_pairs_fixed(&cell, &ps, &qs, &mut out).unwrap();
                for k in 0..n {
                    assert_eq!(out[k].to_bits(), cell.dist2_fixed(ps[k], qs[k]).to_bits());
                }
                if n > 0 {
                    crate::dist2_many_fixed(&cell, ps[0], &qs, &mut out).unwrap();
                    for k in 0..n {
                        assert_eq!(out[k].to_bits(), cell.dist2_fixed(ps[0], qs[k]).to_bits());
                    }
                }
            }
        }
    }
}
