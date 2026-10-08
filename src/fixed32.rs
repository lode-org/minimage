//! Fixed-point fractions on 32 bits: half the bytes of a position.
//!
//! A batch past the second-level cache moves more bytes than it
//! computes on, and a 64-bit position is 24 of them. Each fraction here
//! is the 64-bit fraction of [`crate::fixed`] rounded to its top 32 bits,
//! 12 bytes a position. The wrap is still exact, the difference modulo
//! `2^32` read as signed, and converts to a double exactly; one product
//! with `H` scaled by `2^-32` gives the Cartesian vector. The cost is
//! the quantization: a fraction is within `2^-33` of the cell's, so a
//! displacement is within `2^-32 (|a| + |b| + |c|)` of the engine wrap,
//! about `7e-9` in a cell 10 long. A tie wraps to `-1/2`.

use crate::fixed::{to_fixed32, wrapped32, Fold, Lattice};
use crate::kernel::n2;

/// Squared distance between two 32-bit fixed-point positions.
#[inline(always)]
pub(crate) fn dist2(lat: &Lattice, a: [u32; 3], b: [u32; 3]) -> f64 {
    n2(lat.map([
        wrapped32(a[0], b[0]),
        wrapped32(a[1], b[1]),
        wrapped32(a[2], b[2]),
    ]))
}

/// `out[k] = round32(fold.fixed(rs[k]))`, through the 64-bit fold in
/// chunks that stay on the stack.
pub(crate) fn fixed32_many(fold: &Fold, rs: &[[f64; 3]], out: &mut [[u32; 3]]) {
    const CHUNK: usize = 256;
    let mut wide = [[0u64; 3]; CHUNK];
    for (r, o) in rs.chunks(CHUNK).zip(out.chunks_mut(CHUNK)) {
        let w = &mut wide[..r.len()];
        crate::fixed::fixed_many(fold, r, w);
        for (dst, src) in o.iter_mut().zip(w.iter()) {
            *dst = [to_fixed32(src[0]), to_fixed32(src[1]), to_fixed32(src[2])];
        }
    }
}

/// `out[k] = |b(qs[k]) - b(p)|^2` over 32-bit positions.
pub(crate) fn dist2_many(lat: Lattice, p: [u32; 3], qs: &[[u32; 3]], out: &mut [f64]) {
    #[cfg(all(target_arch = "x86_64", minimage_avx512))]
    {
        if avx512::detected() {
            // SAFETY: AVX-512F and DQ were detected. `qs` and `out` share
            // one length.
            unsafe { avx512::many(&lat, p, qs, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if qs.len() >= 8 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was detected. `qs` and `out` share one length.
            unsafe { avx2::many(&lat, p, qs, out) };
            return;
        }
    }
    for (q, o) in qs.iter().zip(out.iter_mut()) {
        *o = dist2(&lat, p, *q);
    }
}

/// `out[k] = |b(qs[k]) - b(ps[k])|^2` over 32-bit positions.
pub(crate) fn dist2_pairs(lat: Lattice, ps: &[[u32; 3]], qs: &[[u32; 3]], out: &mut [f64]) {
    #[cfg(all(target_arch = "x86_64", minimage_avx512))]
    {
        if avx512::detected() {
            // SAFETY: AVX-512F and DQ were detected. The slices share one
            // length.
            unsafe { avx512::pairs(&lat, ps, qs, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if ps.len() >= 8 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was detected. The slices share one length.
            unsafe { avx2::pairs(&lat, ps, qs, out) };
            return;
        }
    }
    for ((p, q), o) in ps.iter().zip(qs).zip(out.iter_mut()) {
        *o = dist2(&lat, *p, *q);
    }
}

/// Sixteen rows per pass: one vector of 32-bit differences per axis,
/// widened to doubles in two halves. The arithmetic is the scalar order.
#[cfg(all(target_arch = "x86_64", minimage_avx512))]
#[allow(clippy::incompatible_msrv)]
mod avx512 {
    use std::arch::x86_64::*;

    use crate::fixed::Lattice;

    type V = __m512d;

    pub(super) fn detected() -> bool {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
    }

    #[inline(always)]
    unsafe fn idx(i: [i32; 16]) -> __m512i {
        _mm512_loadu_si512(i.as_ptr().cast())
    }

    /// Sixteen `[x, y, z]` rows, three vectors in row order, as x, y, z:
    /// two two-source permutes per axis.
    #[inline(always)]
    unsafe fn transpose(a: __m512i, b: __m512i, c: __m512i) -> [__m512i; 3] {
        let axis = |lo: [i32; 16], hi: [i32; 16]| {
            _mm512_permutex2var_epi32(_mm512_permutex2var_epi32(a, idx(lo), b), idx(hi), c)
        };
        [
            axis(
                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29],
            ),
            axis(
                [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30],
            ),
            axis(
                [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31],
            ),
        ]
    }

    /// `super::dist2` for eight lanes of 32-bit differences.
    #[inline(always)]
    unsafe fn dist2(lat: &Lattice, d: [__m256i; 3]) -> V {
        let d = [
            _mm512_cvtepi32_pd(d[0]),
            _mm512_cvtepi32_pd(d[1]),
            _mm512_cvtepi32_pd(d[2]),
        ];
        let [x, y, z] = match lat {
            Lattice::Diagonal(w) => [
                _mm512_mul_pd(_mm512_set1_pd(w[0]), d[0]),
                _mm512_mul_pd(_mm512_set1_pd(w[1]), d[1]),
                _mm512_mul_pd(_mm512_set1_pd(w[2]), d[2]),
            ],
            Lattice::Full(m) => {
                let mut r = [_mm512_setzero_pd(); 3];
                for (i, slot) in r.iter_mut().enumerate() {
                    *slot = _mm512_add_pd(
                        _mm512_add_pd(
                            _mm512_mul_pd(_mm512_set1_pd(m[0][i]), d[0]),
                            _mm512_mul_pd(_mm512_set1_pd(m[1][i]), d[1]),
                        ),
                        _mm512_mul_pd(_mm512_set1_pd(m[2][i]), d[2]),
                    );
                }
                r
            }
        };
        _mm512_add_pd(
            _mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)),
            _mm512_mul_pd(z, z),
        )
    }

    /// Lane masks for `r <= 16` rows: `3 r` integers over three vectors.
    #[inline(always)]
    fn masks(r: usize) -> [__mmask16; 3] {
        let lanes = |k: usize| -> __mmask16 {
            let m = (3 * r).saturating_sub(16 * k).min(16);
            ((1u32 << m) - 1) as __mmask16
        };
        [lanes(0), lanes(1), lanes(2)]
    }

    /// Distances of sixteen rows of differences `a, b, c`, low eight and
    /// high eight.
    #[inline(always)]
    unsafe fn finish(lat: &Lattice, a: __m512i, b: __m512i, c: __m512i) -> (V, V) {
        let [x, y, z] = transpose(a, b, c);
        let lo = [
            _mm512_castsi512_si256(x),
            _mm512_castsi512_si256(y),
            _mm512_castsi512_si256(z),
        ];
        let hi = [
            _mm512_extracti64x4_epi64::<1>(x),
            _mm512_extracti64x4_epi64::<1>(y),
            _mm512_extracti64x4_epi64::<1>(z),
        ];
        (dist2(lat, lo), dist2(lat, hi))
    }

    /// The last `r < 16` results.
    #[inline(always)]
    unsafe fn store_tail(out: *mut f64, r: usize, (lo, hi): (V, V)) {
        _mm512_mask_storeu_pd(out, ((1u32 << r.min(8)) - 1) as __mmask8, lo);
        if r > 8 {
            _mm512_mask_storeu_pd(out.add(8), ((1u32 << (r - 8)) - 1) as __mmask8, hi);
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn pairs(lat: &Lattice, ps: &[[u32; 3]], qs: &[[u32; 3]], out: &mut [f64]) {
        let n = ps.len();
        let mut i = 0;
        while i + 16 <= n {
            let p: *const __m512i = ps.as_ptr().add(i).cast();
            let q: *const __m512i = qs.as_ptr().add(i).cast();
            let diff = |k: usize| {
                _mm512_sub_epi32(
                    _mm512_loadu_si512(q.add(k).cast()),
                    _mm512_loadu_si512(p.add(k).cast()),
                )
            };
            let (lo, hi) = finish(lat, diff(0), diff(1), diff(2));
            _mm512_storeu_pd(out.as_mut_ptr().add(i), lo);
            _mm512_storeu_pd(out.as_mut_ptr().add(i + 8), hi);
            i += 16;
        }
        if i < n {
            let r = n - i;
            let m = masks(r);
            let p: *const i32 = ps.as_ptr().add(i).cast();
            let q: *const i32 = qs.as_ptr().add(i).cast();
            let diff = |k: usize| {
                _mm512_sub_epi32(
                    _mm512_maskz_loadu_epi32(m[k], q.add(16 * k)),
                    _mm512_maskz_loadu_epi32(m[k], p.add(16 * k)),
                )
            };
            store_tail(
                out.as_mut_ptr().add(i),
                r,
                finish(lat, diff(0), diff(1), diff(2)),
            );
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn many(lat: &Lattice, p: [u32; 3], qs: &[[u32; 3]], out: &mut [f64]) {
        let [x, y, z] = [p[0] as i32, p[1] as i32, p[2] as i32];
        let pv = [
            _mm512_setr_epi32(x, y, z, x, y, z, x, y, z, x, y, z, x, y, z, x),
            _mm512_setr_epi32(y, z, x, y, z, x, y, z, x, y, z, x, y, z, x, y),
            _mm512_setr_epi32(z, x, y, z, x, y, z, x, y, z, x, y, z, x, y, z),
        ];
        let n = qs.len();
        let mut i = 0;
        while i + 16 <= n {
            let q: *const __m512i = qs.as_ptr().add(i).cast();
            let diff = |k: usize| _mm512_sub_epi32(_mm512_loadu_si512(q.add(k).cast()), pv[k]);
            let (lo, hi) = finish(lat, diff(0), diff(1), diff(2));
            _mm512_storeu_pd(out.as_mut_ptr().add(i), lo);
            _mm512_storeu_pd(out.as_mut_ptr().add(i + 8), hi);
            i += 16;
        }
        if i < n {
            let r = n - i;
            let m = masks(r);
            let q: *const i32 = qs.as_ptr().add(i).cast();
            let diff =
                |k: usize| _mm512_sub_epi32(_mm512_maskz_loadu_epi32(m[k], q.add(16 * k)), pv[k]);
            store_tail(
                out.as_mut_ptr().add(i),
                r,
                finish(lat, diff(0), diff(1), diff(2)),
            );
        }
    }
}

/// Eight rows per pass on AVX2: three permutes and two blends per axis,
/// then two halves of four doubles. The arithmetic is the scalar order.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use crate::fixed::Lattice;

    type V = __m256d;

    #[inline(always)]
    unsafe fn idx(i: [i32; 8]) -> __m256i {
        _mm256_loadu_si256(i.as_ptr().cast())
    }

    /// Eight `[x, y, z]` rows, three vectors in row order, as x, y, z.
    #[inline(always)]
    unsafe fn transpose(a: __m256i, b: __m256i, c: __m256i) -> [__m256i; 3] {
        let axis = |ia: [i32; 8], ib: [i32; 8], ic: [i32; 8], mb: i32, mc: i32| {
            let pa = _mm256_permutevar8x32_epi32(a, idx(ia));
            let pb = _mm256_permutevar8x32_epi32(b, idx(ib));
            let pc = _mm256_permutevar8x32_epi32(c, idx(ic));
            let ab = match mb {
                0b0011_1000 => _mm256_blend_epi32::<0b0011_1000>(pa, pb),
                0b0001_1000 => _mm256_blend_epi32::<0b0001_1000>(pa, pb),
                _ => _mm256_blend_epi32::<0b0001_1100>(pa, pb),
            };
            match mc {
                0b1100_0000 => _mm256_blend_epi32::<0b1100_0000>(ab, pc),
                _ => _mm256_blend_epi32::<0b1110_0000>(ab, pc),
            }
        };
        [
            axis(
                [0, 3, 6, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 4, 7, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 5],
                0b0011_1000,
                0b1100_0000,
            ),
            axis(
                [1, 4, 7, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 3, 6],
                0b0001_1000,
                0b1110_0000,
            ),
            axis(
                [2, 5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 6, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 4, 7],
                0b0001_1100,
                0b1110_0000,
            ),
        ]
    }

    /// `super::dist2` for four lanes of 32-bit differences.
    #[inline(always)]
    unsafe fn dist2(lat: &Lattice, d: [__m128i; 3]) -> V {
        let d = [
            _mm256_cvtepi32_pd(d[0]),
            _mm256_cvtepi32_pd(d[1]),
            _mm256_cvtepi32_pd(d[2]),
        ];
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

    #[inline(always)]
    unsafe fn finish(lat: &Lattice, a: __m256i, b: __m256i, c: __m256i, out: *mut f64) {
        let [x, y, z] = transpose(a, b, c);
        let lo = [
            _mm256_castsi256_si128(x),
            _mm256_castsi256_si128(y),
            _mm256_castsi256_si128(z),
        ];
        let hi = [
            _mm256_extracti128_si256::<1>(x),
            _mm256_extracti128_si256::<1>(y),
            _mm256_extracti128_si256::<1>(z),
        ];
        _mm256_storeu_pd(out, dist2(lat, lo));
        _mm256_storeu_pd(out.add(4), dist2(lat, hi));
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pairs(lat: &Lattice, ps: &[[u32; 3]], qs: &[[u32; 3]], out: &mut [f64]) {
        let n = ps.len();
        let mut i = 0;
        while i + 8 <= n {
            let p: *const __m256i = ps.as_ptr().add(i).cast();
            let q: *const __m256i = qs.as_ptr().add(i).cast();
            let diff = |k: usize| {
                _mm256_sub_epi32(_mm256_loadu_si256(q.add(k)), _mm256_loadu_si256(p.add(k)))
            };
            finish(lat, diff(0), diff(1), diff(2), out.as_mut_ptr().add(i));
            i += 8;
        }
        for j in i..n {
            out[j] = super::dist2(lat, ps[j], qs[j]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn many(lat: &Lattice, p: [u32; 3], qs: &[[u32; 3]], out: &mut [f64]) {
        let [x, y, z] = [p[0] as i32, p[1] as i32, p[2] as i32];
        let pv = [
            _mm256_setr_epi32(x, y, z, x, y, z, x, y),
            _mm256_setr_epi32(z, x, y, z, x, y, z, x),
            _mm256_setr_epi32(y, z, x, y, z, x, y, z),
        ];
        let n = qs.len();
        let mut i = 0;
        while i + 8 <= n {
            let q: *const __m256i = qs.as_ptr().add(i).cast();
            let diff = |k: usize| _mm256_sub_epi32(_mm256_loadu_si256(q.add(k)), pv[k]);
            finish(lat, diff(0), diff(1), diff(2), out.as_mut_ptr().add(i));
            i += 8;
        }
        for j in i..n {
            out[j] = super::dist2(lat, p, qs[j]);
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
        ]
    }

    fn points(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut r = [0.0; 3];
                for e in &mut r {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *e = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 40.0;
                }
                r
            })
            .collect()
    }

    #[test]
    fn batches_equal_the_per_pair_call() {
        for cell in cells() {
            for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 100] {
                let rs = points(n, 0x1f2e_3d4c_5b6a_7988);
                let ss = points(n, 0x0123_4567_89ab_cdef);
                let mut fp = vec![[0u32; 3]; n];
                let mut fq = vec![[0u32; 3]; n];
                crate::fixed32_many(&cell, &rs, &mut fp).unwrap();
                crate::fixed32_many(&cell, &ss, &mut fq).unwrap();
                for (r, f) in rs.iter().zip(&fp) {
                    assert_eq!(*f, cell.fixed32(*r));
                }
                let mut out = vec![0.0; n];
                crate::dist2_pairs_fixed32(&cell, &fp, &fq, &mut out).unwrap();
                for k in 0..n {
                    assert_eq!(out[k].to_bits(), cell.dist2_fixed32(fp[k], fq[k]).to_bits());
                }
                if n > 0 {
                    crate::dist2_many_fixed32(&cell, fp[0], &fq, &mut out).unwrap();
                    for k in 0..n {
                        assert_eq!(out[k].to_bits(), cell.dist2_fixed32(fp[0], fq[k]).to_bits());
                    }
                }
            }
        }
    }

    /// Each displacement is within `2^-32 (|a| + |b| + |c|)` of the
    /// engine wrap, plus the rounding of the product.
    #[test]
    fn distances_are_within_the_quantization_bound() {
        for cell in cells() {
            let h = cell.h();
            let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let eps = (len(h[0]) + len(h[1]) + len(h[2])) * crate::fixed::UNIT32;
            let rs = points(500, 0xdead_beef);
            let ss = points(500, 0xfeed_face);
            for (r, s) in rs.iter().zip(&ss) {
                let want = cell.dist2(*r, *s);
                let got = cell.dist2_fixed32(cell.fixed32(*r), cell.fixed32(*s));
                let bound = 2.0 * want.sqrt() * eps + eps * eps + 1e-13 * (1.0 + want);
                assert!((got - want).abs() <= bound, "{got} {want} {bound}");
            }
        }
    }

    /// The AVX2 kernels directly, so a processor with AVX-512, which the
    /// dispatch prefers, still checks them.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernels_equal_the_per_pair_call() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for cell in cells() {
            let lat = cell.fixed32_lattice();
            for n in [8usize, 9, 15, 16, 23, 64] {
                let mut fp = vec![[0u32; 3]; n];
                let mut fq = vec![[0u32; 3]; n];
                crate::fixed32_many(&cell, &points(n, 0x55aa_33cc), &mut fp).unwrap();
                crate::fixed32_many(&cell, &points(n, 0x0ff0_f00f), &mut fq).unwrap();
                let mut out = vec![0.0; n];
                // SAFETY: AVX2 was detected; the slices share one length.
                unsafe { super::avx2::pairs(&lat, &fp, &fq, &mut out) };
                for k in 0..n {
                    assert_eq!(out[k].to_bits(), super::dist2(&lat, fp[k], fq[k]).to_bits());
                }
                // SAFETY: as above.
                unsafe { super::avx2::many(&lat, fp[1], &fq, &mut out) };
                for k in 0..n {
                    assert_eq!(out[k].to_bits(), super::dist2(&lat, fp[1], fq[k]).to_bits());
                }
            }
        }
    }

    #[test]
    fn a_half_cell_apart_wraps_to_minus_half() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let a = cell.fixed32([0.0, 0.0, 0.0]);
        let b = cell.fixed32([5.0, 0.0, 0.0]);
        assert_eq!(b[0], 1 << 31);
        assert_eq!(cell.displacement_fixed32(a, b), [-5.0, 0.0, 0.0]);
        assert_eq!(cell.displacement_fixed32(b, a), [-5.0, 0.0, 0.0]);
    }
}
