//! Fused batch kernels over packed `[x, y, z]` rows.
//!
//! One pass per batch: four rows load as three vectors and transpose in
//! registers, then the difference, the wrap, and the square or the
//! store happen without a staging buffer. Each lane runs the per-pair
//! arithmetic in the per-pair order, so a batch squared distance equals
//! the per-pair call bit for bit, and a wrapped vector equals it up to
//! the sign of a zero (vector rounding keeps the sign of a tiny
//! negative, the scalar truncation does not). A group of four with any
//! pair a full box apart, and the tail, run the per-pair code itself.

use crate::kernel::{self, n2, Tri};

/// The wrap a cell needs, with the data that wrap reads.
#[derive(Clone, Copy)]
pub(crate) enum Frame {
    Ortho([f64; 3]),
    Tri(Tri),
    General([[f64; 3]; 3], [[f64; 3]; 3]),
}

impl Frame {
    #[inline(always)]
    fn wrap(&self, d: [f64; 3]) -> [f64; 3] {
        match self {
            Frame::Ortho(l) => kernel::ortho_wrap(*l, d),
            Frame::Tri(t) => t.wrap(d),
            Frame::General(h, hinv) => kernel::general_wrap(*h, *hinv, d),
        }
    }

    #[inline(always)]
    fn dist2(&self, d: [f64; 3]) -> f64 {
        match self {
            Frame::Ortho(l) => kernel::ortho_dist2(*l, d),
            _ => n2(self.wrap(d)),
        }
    }
}

#[inline(always)]
fn sub(q: [f64; 3], p: [f64; 3]) -> [f64; 3] {
    [q[0] - p[0], q[1] - p[1], q[2] - p[2]]
}

/// The per-pair code for rows `from..to`, out of line. Inlined into a
/// kernel, it keeps the rows that kernel loaded alive across the
/// far-image test, and they spill to the stack every iteration.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn pairs_from(
    frame: Frame,
    ps: &[[f64; 3]],
    qs: &[[f64; 3]],
    out: &mut [f64],
    from: usize,
    to: usize,
) {
    for j in from..to {
        out[j] = frame.dist2(sub(qs[j], ps[j]));
    }
}

/// [`pairs_from`] for one source.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn many_from(frame: Frame, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64], from: usize, to: usize) {
    for j in from..to {
        out[j] = frame.dist2(sub(qs[j], p));
    }
}

/// [`pairs_from`] for the wrap.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn wrap_from(frame: Frame, diffs: &[[f64; 3]], out: &mut [[f64; 3]], from: usize, to: usize) {
    for j in from..to {
        out[j] = frame.wrap(diffs[j]);
    }
}

/// A group with a row a full box away: rare, so cold.
#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn pairs_far(
    frame: Frame,
    ps: &[[f64; 3]],
    qs: &[[f64; 3]],
    out: &mut [f64],
    from: usize,
    to: usize,
) {
    pairs_from(frame, ps, qs, out, from, to);
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn many_far(frame: Frame, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64], from: usize, to: usize) {
    many_from(frame, p, qs, out, from, to);
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn wrap_far(frame: Frame, diffs: &[[f64; 3]], out: &mut [[f64; 3]], from: usize, to: usize) {
    wrap_from(frame, diffs, out, from, to);
}

/// `out[k] = |wrap(qs[k] - ps[k])|^2`.
pub(crate) fn dist2_pairs(frame: Frame, ps: &[[f64; 3]], qs: &[[f64; 3]], out: &mut [f64]) {
    #[cfg(all(target_arch = "x86_64", minimage_avx512))]
    {
        if avx512::detected() {
            // SAFETY: AVX-512F and DQ were detected. The slices share one
            // length.
            unsafe {
                match frame {
                    Frame::Ortho(l) => avx512::ortho_pairs(l, ps, qs, out),
                    Frame::Tri(t) => avx512::tri_pairs(&t, ps, qs, out),
                    Frame::General(h, hinv) => avx512::general_pairs(&h, &hinv, ps, qs, out),
                }
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if ps.len() >= 4 && std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected. The slices share one length.
            unsafe {
                match frame {
                    Frame::Ortho(l) => avx::ortho_pairs(l, ps, qs, out),
                    Frame::Tri(t) => avx::tri_pairs(&t, ps, qs, out),
                    Frame::General(h, hinv) => avx::general_pairs(&h, &hinv, ps, qs, out),
                }
            }
            return;
        }
    }
    for ((p, q), o) in ps.iter().zip(qs).zip(out.iter_mut()) {
        *o = frame.dist2(sub(*q, *p));
    }
}

/// `out[k] = |wrap(qs[k] - p)|^2`.
pub(crate) fn dist2_many(frame: Frame, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
    #[cfg(all(target_arch = "x86_64", minimage_avx512))]
    {
        if avx512::detected() {
            // SAFETY: AVX-512F and DQ were detected. `qs` and `out` share
            // one length.
            unsafe {
                match frame {
                    Frame::Ortho(l) => avx512::ortho_many(l, p, qs, out),
                    Frame::Tri(t) => avx512::tri_many(&t, p, qs, out),
                    Frame::General(h, hinv) => avx512::general_many(&h, &hinv, p, qs, out),
                }
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if qs.len() >= 4 && std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected. `qs` and `out` share one length.
            unsafe {
                match frame {
                    Frame::Ortho(l) => avx::ortho_many(l, p, qs, out),
                    Frame::Tri(t) => avx::tri_many(&t, p, qs, out),
                    Frame::General(h, hinv) => avx::general_many(&h, &hinv, p, qs, out),
                }
            }
            return;
        }
    }
    for (q, o) in qs.iter().zip(out.iter_mut()) {
        *o = frame.dist2(sub(*q, p));
    }
}

/// `out[k] = wrap(diffs[k])`.
pub(crate) fn wrap_many(frame: Frame, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
    // The orthorhombic wrap stores as much as it loads; three 64-byte
    // stores per eight rows split cache lines often enough that the
    // 32-byte AVX kernel is faster there.
    #[cfg(all(target_arch = "x86_64", minimage_avx512))]
    {
        if avx512::detected() {
            // SAFETY: AVX-512F and DQ were detected. `diffs` and `out`
            // share one length.
            match frame {
                Frame::Tri(t) => return unsafe { avx512::tri_wrap(&t, diffs, out) },
                Frame::General(h, hinv) => {
                    return unsafe { avx512::general_wrap(&h, &hinv, diffs, out) }
                }
                Frame::Ortho(_) => {}
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if diffs.len() >= 4 && std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected. `diffs` and `out` share one length.
            unsafe {
                match frame {
                    Frame::Ortho(l) => avx::ortho_wrap(l, diffs, out),
                    Frame::Tri(t) => avx::tri_wrap(&t, diffs, out),
                    Frame::General(h, hinv) => avx::general_wrap(&h, &hinv, diffs, out),
                }
            }
            return;
        }
    }
    for (d, o) in diffs.iter().zip(out.iter_mut()) {
        *o = frame.wrap(*d);
    }
}

/// The AVX kernels on eight lanes. A short group, the tail of a batch,
/// loads and stores under a lane mask and runs the same arithmetic, so a
/// batch of any length equals the per-pair calls.
#[cfg(all(target_arch = "x86_64", minimage_avx512))]
#[allow(clippy::incompatible_msrv)]
mod avx512 {
    use std::arch::x86_64::*;

    use super::{many_far, pairs_far, Frame};
    use crate::kernel::Tri;

    type V = __m512d;
    type M = __mmask8;

    pub(super) fn detected() -> bool {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
    }

    /// Lane masks for `r <= 8` rows: `3 r` doubles over three vectors,
    /// and `r` results.
    #[inline(always)]
    fn masks(r: usize) -> ([M; 3], M) {
        let lanes = |k: usize| -> M {
            let m = (3 * r).saturating_sub(8 * k).min(8);
            ((1u16 << m) - 1) as M
        };
        ([lanes(0), lanes(1), lanes(2)], ((1u16 << r) - 1) as M)
    }

    #[inline(always)]
    unsafe fn idx(i: [i64; 8]) -> __m512i {
        _mm512_setr_epi64(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7])
    }

    /// Eight packed rows, three vectors in row order, as x, y, z.
    #[inline(always)]
    unsafe fn transpose_in(a: V, b: V, c: V) -> [V; 3] {
        let x = _mm512_permutex2var_pd(a, idx([0, 3, 6, 9, 12, 15, 0, 0]), b);
        let y = _mm512_permutex2var_pd(a, idx([1, 4, 7, 10, 13, 0, 0, 0]), b);
        let z = _mm512_permutex2var_pd(a, idx([2, 5, 8, 11, 14, 0, 0, 0]), b);
        [
            _mm512_permutex2var_pd(x, idx([0, 1, 2, 3, 4, 5, 10, 13]), c),
            _mm512_permutex2var_pd(y, idx([0, 1, 2, 3, 4, 8, 11, 14]), c),
            _mm512_permutex2var_pd(z, idx([0, 1, 2, 3, 4, 9, 12, 15]), c),
        ]
    }

    /// Inverse of [`transpose_in`].
    #[inline(always)]
    unsafe fn transpose_out(x: V, y: V, z: V) -> [V; 3] {
        let a = _mm512_permutex2var_pd(x, idx([0, 8, 0, 1, 9, 0, 2, 10]), y);
        let b = _mm512_permutex2var_pd(x, idx([0, 3, 11, 0, 4, 12, 0, 5]), y);
        let c = _mm512_permutex2var_pd(x, idx([13, 0, 6, 14, 0, 7, 15, 0]), y);
        [
            _mm512_permutex2var_pd(a, idx([0, 1, 8, 3, 4, 9, 6, 7]), z),
            _mm512_permutex2var_pd(b, idx([10, 1, 2, 11, 4, 5, 12, 7]), z),
            _mm512_permutex2var_pd(c, idx([0, 13, 2, 3, 14, 5, 6, 15]), z),
        ]
    }

    /// Per-axis values in packed-row order for eight rows.
    #[inline(always)]
    unsafe fn splat(p: [f64; 3]) -> [V; 3] {
        let [x, y, z] = p;
        [
            _mm512_setr_pd(x, y, z, x, y, z, x, y),
            _mm512_setr_pd(z, x, y, z, x, y, z, x),
            _mm512_setr_pd(y, z, x, y, z, x, y, z),
        ]
    }

    /// Up to eight packed rows, zero past `r`.
    #[inline(always)]
    unsafe fn load(rows: *const f64, m: &[M; 3]) -> [V; 3] {
        [
            _mm512_maskz_loadu_pd(m[0], rows),
            _mm512_maskz_loadu_pd(m[1], rows.add(8)),
            _mm512_maskz_loadu_pd(m[2], rows.add(16)),
        ]
    }

    #[inline(always)]
    unsafe fn store(rows: *mut f64, m: &[M; 3], v: [V; 3]) {
        _mm512_mask_storeu_pd(rows, m[0], v[0]);
        _mm512_mask_storeu_pd(rows.add(8), m[1], v[1]);
        _mm512_mask_storeu_pd(rows.add(16), m[2], v[2]);
    }

    /// `qs[k] - ps[k]` on packed rows, zero past `r`, then transposed.
    #[inline(always)]
    unsafe fn diff(qs: *const f64, ps: *const f64, m: &[M; 3]) -> [V; 3] {
        let q = load(qs, m);
        let p = load(ps, m);
        transpose_in(
            _mm512_sub_pd(q[0], p[0]),
            _mm512_sub_pd(q[1], p[1]),
            _mm512_sub_pd(q[2], p[2]),
        )
    }

    /// `qs[k] - p`, zero past `r`.
    #[inline(always)]
    unsafe fn diff_from(qs: *const f64, p: &[V; 3], m: &[M; 3]) -> [V; 3] {
        let q = load(qs, m);
        transpose_in(
            _mm512_maskz_sub_pd(m[0], q[0], p[0]),
            _mm512_maskz_sub_pd(m[1], q[1], p[1]),
            _mm512_maskz_sub_pd(m[2], q[2], p[2]),
        )
    }

    /// `kernel::round_away` per lane.
    #[inline(always)]
    unsafe fn round_away(x: V) -> V {
        let signed = _mm512_or_pd(
            _mm512_and_pd(x, _mm512_set1_pd(-0.0)),
            _mm512_set1_pd(0.499_999_999_999_999_94),
        );
        _mm512_roundscale_pd::<0x0B>(_mm512_add_pd(x, signed))
    }

    #[inline(always)]
    unsafe fn norm(w: [V; 3]) -> V {
        _mm512_add_pd(
            _mm512_add_pd(_mm512_mul_pd(w[0], w[0]), _mm512_mul_pd(w[1], w[1])),
            _mm512_mul_pd(w[2], w[2]),
        )
    }

    /// `kernel::ortho_dist2` per lane, or `None` when a lane is a full
    /// box away.
    #[inline(always)]
    unsafe fn ortho_dist2(l: &[V; 3], d: [V; 3]) -> Option<V> {
        let sign = _mm512_set1_pd(-0.0);
        let a = [
            _mm512_andnot_pd(sign, d[0]),
            _mm512_andnot_pd(sign, d[1]),
            _mm512_andnot_pd(sign, d[2]),
        ];
        let b = [
            _mm512_sub_pd(l[0], a[0]),
            _mm512_sub_pd(l[1], a[1]),
            _mm512_sub_pd(l[2], a[2]),
        ];
        let any = _mm512_or_pd(_mm512_or_pd(b[0], b[1]), b[2]);
        if _mm512_movepi64_mask(_mm512_castpd_si512(any)) != 0 {
            return None;
        }
        Some(norm([
            _mm512_min_pd(b[0], a[0]),
            _mm512_min_pd(b[1], a[1]),
            _mm512_min_pd(b[2], a[2]),
        ]))
    }

    /// `Tri::wrap` per lane.
    #[inline(always)]
    unsafe fn tri_wrap8(t: &Tri, d: [V; 3]) -> [V; 3] {
        let s1 = _mm512_set1_pd;
        let (xy, xz, yz) = (s1(t.xy), s1(t.xz), s1(t.yz));
        let sz = _mm512_mul_pd(d[2], s1(t.inv_lz));
        let sy = _mm512_mul_pd(_mm512_sub_pd(d[1], _mm512_mul_pd(yz, sz)), s1(t.inv_ly));
        let sx = _mm512_mul_pd(
            _mm512_sub_pd(
                _mm512_sub_pd(d[0], _mm512_mul_pd(xy, sy)),
                _mm512_mul_pd(xz, sz),
            ),
            s1(t.inv_lx),
        );
        let sx = _mm512_sub_pd(sx, round_away(sx));
        let sy = _mm512_sub_pd(sy, round_away(sy));
        let sz = _mm512_sub_pd(sz, round_away(sz));
        [
            _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(s1(t.lx), sx), _mm512_mul_pd(xy, sy)),
                _mm512_mul_pd(xz, sz),
            ),
            _mm512_add_pd(_mm512_mul_pd(s1(t.ly), sy), _mm512_mul_pd(yz, sz)),
            _mm512_mul_pd(s1(t.lz), sz),
        ]
    }

    /// `kernel::mul(m, v)` per lane, column-major `m`.
    #[inline(always)]
    unsafe fn mul(m: &[[f64; 3]; 3], v: [V; 3]) -> [V; 3] {
        let mut out = [_mm512_setzero_pd(); 3];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = _mm512_add_pd(
                _mm512_add_pd(
                    _mm512_mul_pd(_mm512_set1_pd(m[0][i]), v[0]),
                    _mm512_mul_pd(_mm512_set1_pd(m[1][i]), v[1]),
                ),
                _mm512_mul_pd(_mm512_set1_pd(m[2][i]), v[2]),
            );
        }
        out
    }

    /// `kernel::general_wrap` per lane.
    #[inline(always)]
    unsafe fn general_wrap8(h: &[[f64; 3]; 3], hinv: &[[f64; 3]; 3], d: [V; 3]) -> [V; 3] {
        let s = mul(hinv, d);
        mul(
            h,
            [
                _mm512_sub_pd(s[0], round_away(s[0])),
                _mm512_sub_pd(s[1], round_away(s[1])),
                _mm512_sub_pd(s[2], round_away(s[2])),
            ],
        )
    }

    /// Groups of eight rows, then one masked group of `r < 8`.
    #[inline(always)]
    unsafe fn groups(n: usize, mut f: impl FnMut(usize, usize, &[M; 3], M)) {
        let full = masks(8);
        let mut i = 0;
        while i + 8 <= n {
            f(i, 8, &full.0, full.1);
            i += 8;
        }
        if i < n {
            let (m, mo) = masks(n - i);
            f(i, n - i, &m, mo);
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn ortho_pairs(
        l: [f64; 3],
        ps: &[[f64; 3]],
        qs: &[[f64; 3]],
        out: &mut [f64],
    ) {
        let lv = [
            _mm512_set1_pd(l[0]),
            _mm512_set1_pd(l[1]),
            _mm512_set1_pd(l[2]),
        ];
        let frame = Frame::Ortho(l);
        groups(ps.len(), |i, r, m, mo| {
            let d = diff(qs.as_ptr().add(i).cast(), ps.as_ptr().add(i).cast(), m);
            match ortho_dist2(&lv, d) {
                Some(r2) => _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mo, r2),
                None => pairs_far(frame, ps, qs, out, i, i + r),
            }
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn ortho_many(l: [f64; 3], p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
        let lv = [
            _mm512_set1_pd(l[0]),
            _mm512_set1_pd(l[1]),
            _mm512_set1_pd(l[2]),
        ];
        let frame = Frame::Ortho(l);
        let pv = splat(p);
        groups(qs.len(), |i, r, m, mo| {
            let d = diff_from(qs.as_ptr().add(i).cast(), &pv, m);
            match ortho_dist2(&lv, d) {
                Some(r2) => _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mo, r2),
                None => many_far(frame, p, qs, out, i, i + r),
            }
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn tri_pairs(t: &Tri, ps: &[[f64; 3]], qs: &[[f64; 3]], out: &mut [f64]) {
        groups(ps.len(), |i, _, m, mo| {
            let d = diff(qs.as_ptr().add(i).cast(), ps.as_ptr().add(i).cast(), m);
            _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mo, norm(tri_wrap8(t, d)));
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn tri_many(t: &Tri, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
        let pv = splat(p);
        groups(qs.len(), |i, _, m, mo| {
            let d = diff_from(qs.as_ptr().add(i).cast(), &pv, m);
            _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mo, norm(tri_wrap8(t, d)));
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn tri_wrap(t: &Tri, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
        groups(diffs.len(), |i, _, m, _| {
            let v = load(diffs.as_ptr().add(i).cast(), m);
            let w = tri_wrap8(t, transpose_in(v[0], v[1], v[2]));
            store(
                out.as_mut_ptr().add(i).cast(),
                m,
                transpose_out(w[0], w[1], w[2]),
            );
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn general_pairs(
        h: &[[f64; 3]; 3],
        hinv: &[[f64; 3]; 3],
        ps: &[[f64; 3]],
        qs: &[[f64; 3]],
        out: &mut [f64],
    ) {
        groups(ps.len(), |i, _, m, mo| {
            let d = diff(qs.as_ptr().add(i).cast(), ps.as_ptr().add(i).cast(), m);
            _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mo, norm(general_wrap8(h, hinv, d)));
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn general_many(
        h: &[[f64; 3]; 3],
        hinv: &[[f64; 3]; 3],
        p: [f64; 3],
        qs: &[[f64; 3]],
        out: &mut [f64],
    ) {
        let pv = splat(p);
        groups(qs.len(), |i, _, m, mo| {
            let d = diff_from(qs.as_ptr().add(i).cast(), &pv, m);
            _mm512_mask_storeu_pd(out.as_mut_ptr().add(i), mo, norm(general_wrap8(h, hinv, d)));
        });
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) unsafe fn general_wrap(
        h: &[[f64; 3]; 3],
        hinv: &[[f64; 3]; 3],
        diffs: &[[f64; 3]],
        out: &mut [[f64; 3]],
    ) {
        groups(diffs.len(), |i, _, m, _| {
            let v = load(diffs.as_ptr().add(i).cast(), m);
            let w = general_wrap8(h, hinv, transpose_in(v[0], v[1], v[2]));
            store(
                out.as_mut_ptr().add(i).cast(),
                m,
                transpose_out(w[0], w[1], w[2]),
            );
        });
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    use super::{many_far, many_from, pairs_far, pairs_from, sub, wrap_far, wrap_from, Frame};
    use crate::kernel::Tri;

    type V = __m256d;

    /// Rows `k..k+4` of a packed `[x, y, z]` slice as x, y, z vectors.
    ///
    /// # Safety
    ///
    /// AVX is available and `rows` holds twelve readable doubles.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn load4(rows: *const f64) -> (V, V, V) {
        // x0 y0 z0 x1, y1 z1 x2 y2, z2 x3 y3 z3.
        let [x, y, z] = transpose_in(
            _mm256_loadu_pd(rows),
            _mm256_loadu_pd(rows.add(4)),
            _mm256_loadu_pd(rows.add(8)),
        );
        (x, y, z)
    }

    /// Inverse of [`load4`].
    ///
    /// # Safety
    ///
    /// AVX is available and `rows` holds twelve writable doubles.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn store4(rows: *mut f64, x: V, y: V, z: V) {
        let u = _mm256_unpacklo_pd(x, y);
        let w = _mm256_unpackhi_pd(y, z);
        let v = _mm256_shuffle_pd(z, x, 0b1010);
        _mm256_storeu_pd(rows, _mm256_permute2f128_pd(u, v, 0x20));
        _mm256_storeu_pd(rows.add(4), _mm256_blend_pd(w, u, 0b1100));
        _mm256_storeu_pd(rows.add(8), _mm256_permute2f128_pd(v, w, 0x31));
    }

    /// `kernel::round_away` per lane: add the largest double below one
    /// half with the sign of `x`, then truncate.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn round_away(x: V) -> V {
        let signed = _mm256_or_pd(
            _mm256_and_pd(x, _mm256_set1_pd(-0.0)),
            _mm256_set1_pd(0.499_999_999_999_999_94),
        );
        _mm256_round_pd(_mm256_add_pd(x, signed), 0x0B)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn norm(x: V, y: V, z: V) -> V {
        _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)),
            _mm256_mul_pd(z, z),
        )
    }

    /// Orthorhombic lanes: lengths and the sign mask.
    #[derive(Clone, Copy)]
    struct Ortho {
        l: [V; 3],
        sign: V,
    }

    impl Ortho {
        #[target_feature(enable = "avx")]
        #[inline]
        unsafe fn new(l: [f64; 3]) -> Self {
            Self {
                l: [
                    _mm256_set1_pd(l[0]),
                    _mm256_set1_pd(l[1]),
                    _mm256_set1_pd(l[2]),
                ],
                sign: _mm256_set1_pd(-0.0),
            }
        }
    }

    /// `kernel::ortho_dist2` per lane, or `None` when a lane is a full
    /// box away and needs the floor form.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn ortho_dist2(k: &Ortho, d: [V; 3]) -> Option<V> {
        let a = [
            _mm256_andnot_pd(k.sign, d[0]),
            _mm256_andnot_pd(k.sign, d[1]),
            _mm256_andnot_pd(k.sign, d[2]),
        ];
        let b = [
            _mm256_sub_pd(k.l[0], a[0]),
            _mm256_sub_pd(k.l[1], a[1]),
            _mm256_sub_pd(k.l[2], a[2]),
        ];
        // A lane is past one image exactly when `L - |d|` is negative,
        // so the sign bits replace three compares.
        if _mm256_movemask_pd(_mm256_or_pd(_mm256_or_pd(b[0], b[1]), b[2])) != 0 {
            return None;
        }
        // min(L - |d|, |d|) returns |d| unless L - |d| is smaller, as the
        // scalar select does.
        let w = [
            _mm256_min_pd(b[0], a[0]),
            _mm256_min_pd(b[1], a[1]),
            _mm256_min_pd(b[2], a[2]),
        ];
        Some(norm(w[0], w[1], w[2]))
    }

    /// Restricted triclinic lanes, broadcast once per batch.
    #[derive(Clone, Copy)]
    struct Lamda {
        l: [V; 3],
        xy: V,
        xz: V,
        yz: V,
        inv: [V; 3],
    }

    impl Lamda {
        #[target_feature(enable = "avx")]
        #[inline]
        unsafe fn new(t: &Tri) -> Self {
            Self {
                l: [
                    _mm256_set1_pd(t.lx),
                    _mm256_set1_pd(t.ly),
                    _mm256_set1_pd(t.lz),
                ],
                xy: _mm256_set1_pd(t.xy),
                xz: _mm256_set1_pd(t.xz),
                yz: _mm256_set1_pd(t.yz),
                inv: [
                    _mm256_set1_pd(t.inv_lx),
                    _mm256_set1_pd(t.inv_ly),
                    _mm256_set1_pd(t.inv_lz),
                ],
            }
        }
    }

    /// `Tri::wrap` per lane, same operations in the same order.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn tri_wrap4(k: &Lamda, d: [V; 3]) -> [V; 3] {
        let sz = _mm256_mul_pd(d[2], k.inv[2]);
        let sy = _mm256_mul_pd(_mm256_sub_pd(d[1], _mm256_mul_pd(k.yz, sz)), k.inv[1]);
        let sx = _mm256_mul_pd(
            _mm256_sub_pd(
                _mm256_sub_pd(d[0], _mm256_mul_pd(k.xy, sy)),
                _mm256_mul_pd(k.xz, sz),
            ),
            k.inv[0],
        );
        let sx = _mm256_sub_pd(sx, round_away(sx));
        let sy = _mm256_sub_pd(sy, round_away(sy));
        let sz = _mm256_sub_pd(sz, round_away(sz));
        [
            _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(k.l[0], sx), _mm256_mul_pd(k.xy, sy)),
                _mm256_mul_pd(k.xz, sz),
            ),
            _mm256_add_pd(_mm256_mul_pd(k.l[1], sy), _mm256_mul_pd(k.yz, sz)),
            _mm256_mul_pd(k.l[2], sz),
        ]
    }

    /// `kernel::mul(m, v)` per lane, column-major `m`.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn mul(m: &[[f64; 3]; 3], v: [V; 3]) -> [V; 3] {
        let mut out = [_mm256_setzero_pd(); 3];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(_mm256_broadcast_sd(&m[0][i]), v[0]),
                    _mm256_mul_pd(_mm256_broadcast_sd(&m[1][i]), v[1]),
                ),
                _mm256_mul_pd(_mm256_broadcast_sd(&m[2][i]), v[2]),
            );
        }
        out
    }

    /// `kernel::general_wrap` per lane.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn general_wrap4(h: &[[f64; 3]; 3], hinv: &[[f64; 3]; 3], d: [V; 3]) -> [V; 3] {
        let s = mul(hinv, d);
        let s = [
            _mm256_sub_pd(s[0], round_away(s[0])),
            _mm256_sub_pd(s[1], round_away(s[1])),
            _mm256_sub_pd(s[2], round_away(s[2])),
        ];
        mul(h, s)
    }

    /// Transpose three packed vectors (four rows) into x, y, z.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn transpose_in(a: V, b: V, c: V) -> [V; 3] {
        let u = _mm256_blend_pd(a, b, 0b1100);
        let v = _mm256_permute2f128_pd(a, c, 0x21);
        let w = _mm256_blend_pd(b, c, 0b1100);
        [
            _mm256_shuffle_pd(u, v, 0b1010),
            _mm256_shuffle_pd(u, w, 0b0101),
            _mm256_shuffle_pd(v, w, 0b1010),
        ]
    }

    /// `qs[k] - ps[k]` for four rows. The subtraction is per element, so
    /// it runs on the packed rows and only the difference is transposed.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn diff4(qs: *const f64, ps: *const f64) -> [V; 3] {
        transpose_in(
            _mm256_sub_pd(_mm256_loadu_pd(qs), _mm256_loadu_pd(ps)),
            _mm256_sub_pd(_mm256_loadu_pd(qs.add(4)), _mm256_loadu_pd(ps.add(4))),
            _mm256_sub_pd(_mm256_loadu_pd(qs.add(8)), _mm256_loadu_pd(ps.add(8))),
        )
    }

    /// `qs[k] - p` for four rows, with `p` laid out as packed rows.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn diff4_from(qs: *const f64, p: &[V; 3]) -> [V; 3] {
        transpose_in(
            _mm256_sub_pd(_mm256_loadu_pd(qs), p[0]),
            _mm256_sub_pd(_mm256_loadu_pd(qs.add(4)), p[1]),
            _mm256_sub_pd(_mm256_loadu_pd(qs.add(8)), p[2]),
        )
    }

    /// Per-axis values repeated in packed-row order: `[x y z x]`,
    /// `[y z x y]`, `[z x y z]`, the layout of four rows in three loads.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn splat(p: [f64; 3]) -> [V; 3] {
        [
            _mm256_setr_pd(p[0], p[1], p[2], p[0]),
            _mm256_setr_pd(p[1], p[2], p[0], p[1]),
            _mm256_setr_pd(p[2], p[0], p[1], p[2]),
        ]
    }

    /// `kernel::wrap_one` on packed rows directly: each element meets
    /// its own axis length, so the rows never transpose. `None` when an
    /// element is a full box away.
    #[target_feature(enable = "avx")]
    #[inline]
    unsafe fn ortho_wrap_packed(l: &[V; 3], half: &[V; 3], d: [V; 3]) -> Option<[V; 3]> {
        let sign = _mm256_set1_pd(-0.0);
        let mut far = _mm256_setzero_pd();
        let mut w = d;
        for k in 0..3 {
            let mag = _mm256_andnot_pd(sign, d[k]);
            far = _mm256_or_pd(far, _mm256_cmp_pd(mag, l[k], _CMP_GE_OQ));
            let neg_half = _mm256_xor_pd(half[k], sign);
            let up = _mm256_cmp_pd(d[k], neg_half, _CMP_LT_OQ);
            let x = _mm256_add_pd(d[k], _mm256_and_pd(up, l[k]));
            let down = _mm256_cmp_pd(x, half[k], _CMP_GE_OQ);
            w[k] = _mm256_sub_pd(x, _mm256_and_pd(down, l[k]));
        }
        if _mm256_movemask_pd(far) != 0 {
            return None;
        }
        Some(w)
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn ortho_pairs(
        l: [f64; 3],
        ps: &[[f64; 3]],
        qs: &[[f64; 3]],
        out: &mut [f64],
    ) {
        let k = Ortho::new(l);
        let frame = Frame::Ortho(l);
        let n = ps.len();
        let mut i = 0;
        while i + 4 <= n {
            let d = diff4(qs.as_ptr().add(i).cast(), ps.as_ptr().add(i).cast());
            match ortho_dist2(&k, d) {
                Some(r2) => _mm256_storeu_pd(out.as_mut_ptr().add(i), r2),
                None => pairs_far(frame, ps, qs, out, i, i + 4),
            }
            i += 4;
        }
        pairs_from(frame, ps, qs, out, i, n);
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn ortho_many(l: [f64; 3], p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
        let k = Ortho::new(l);
        let frame = Frame::Ortho(l);
        let pv = splat(p);
        let n = qs.len();
        let mut i = 0;
        while i + 4 <= n {
            let d = diff4_from(qs.as_ptr().add(i).cast(), &pv);
            match ortho_dist2(&k, d) {
                Some(r2) => _mm256_storeu_pd(out.as_mut_ptr().add(i), r2),
                None => many_far(frame, p, qs, out, i, i + 4),
            }
            i += 4;
        }
        many_from(frame, p, qs, out, i, n);
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn ortho_wrap(l: [f64; 3], diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
        let lv = splat(l);
        let hv = splat([0.5 * l[0], 0.5 * l[1], 0.5 * l[2]]);
        let frame = Frame::Ortho(l);
        let n = diffs.len();
        let mut i = 0;
        while i + 4 <= n {
            let src: *const f64 = diffs.as_ptr().add(i).cast();
            let d = [
                _mm256_loadu_pd(src),
                _mm256_loadu_pd(src.add(4)),
                _mm256_loadu_pd(src.add(8)),
            ];
            match ortho_wrap_packed(&lv, &hv, d) {
                Some(w) => {
                    let dst: *mut f64 = out.as_mut_ptr().add(i).cast();
                    _mm256_storeu_pd(dst, w[0]);
                    _mm256_storeu_pd(dst.add(4), w[1]);
                    _mm256_storeu_pd(dst.add(8), w[2]);
                }
                None => wrap_far(frame, diffs, out, i, i + 4),
            }
            i += 4;
        }
        wrap_from(frame, diffs, out, i, n);
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn tri_pairs(t: &Tri, ps: &[[f64; 3]], qs: &[[f64; 3]], out: &mut [f64]) {
        let k = Lamda::new(t);
        let n = ps.len();
        let mut i = 0;
        while i + 4 <= n {
            let d = diff4(qs.as_ptr().add(i).cast(), ps.as_ptr().add(i).cast());
            let w = tri_wrap4(&k, d);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), norm(w[0], w[1], w[2]));
            i += 4;
        }
        let frame = Frame::Tri(*t);
        for j in i..n {
            out[j] = frame.dist2(sub(qs[j], ps[j]));
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn tri_many(t: &Tri, p: [f64; 3], qs: &[[f64; 3]], out: &mut [f64]) {
        let k = Lamda::new(t);
        let pv = splat(p);
        let n = qs.len();
        let mut i = 0;
        while i + 4 <= n {
            let d = diff4_from(qs.as_ptr().add(i).cast(), &pv);
            let w = tri_wrap4(&k, d);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), norm(w[0], w[1], w[2]));
            i += 4;
        }
        let frame = Frame::Tri(*t);
        for j in i..n {
            out[j] = frame.dist2(sub(qs[j], p));
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn tri_wrap(t: &Tri, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
        let k = Lamda::new(t);
        let n = diffs.len();
        let mut i = 0;
        while i + 4 <= n {
            let (x, y, z) = load4(diffs.as_ptr().add(i).cast());
            let w = tri_wrap4(&k, [x, y, z]);
            store4(out.as_mut_ptr().add(i).cast(), w[0], w[1], w[2]);
            i += 4;
        }
        let frame = Frame::Tri(*t);
        for j in i..n {
            out[j] = frame.wrap(diffs[j]);
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn general_pairs(
        h: &[[f64; 3]; 3],
        hinv: &[[f64; 3]; 3],
        ps: &[[f64; 3]],
        qs: &[[f64; 3]],
        out: &mut [f64],
    ) {
        let n = ps.len();
        let mut i = 0;
        while i + 4 <= n {
            let d = diff4(qs.as_ptr().add(i).cast(), ps.as_ptr().add(i).cast());
            let w = general_wrap4(h, hinv, d);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), norm(w[0], w[1], w[2]));
            i += 4;
        }
        let frame = Frame::General(*h, *hinv);
        for j in i..n {
            out[j] = frame.dist2(sub(qs[j], ps[j]));
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn general_many(
        h: &[[f64; 3]; 3],
        hinv: &[[f64; 3]; 3],
        p: [f64; 3],
        qs: &[[f64; 3]],
        out: &mut [f64],
    ) {
        let pv = splat(p);
        let n = qs.len();
        let mut i = 0;
        while i + 4 <= n {
            let d = diff4_from(qs.as_ptr().add(i).cast(), &pv);
            let w = general_wrap4(h, hinv, d);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), norm(w[0], w[1], w[2]));
            i += 4;
        }
        let frame = Frame::General(*h, *hinv);
        for j in i..n {
            out[j] = frame.dist2(sub(qs[j], p));
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn general_wrap(
        h: &[[f64; 3]; 3],
        hinv: &[[f64; 3]; 3],
        diffs: &[[f64; 3]],
        out: &mut [[f64; 3]],
    ) {
        let n = diffs.len();
        let mut i = 0;
        while i + 4 <= n {
            let (x, y, z) = load4(diffs.as_ptr().add(i).cast());
            let w = general_wrap4(h, hinv, [x, y, z]);
            store4(out.as_mut_ptr().add(i).cast(), w[0], w[1], w[2]);
            i += 4;
        }
        let frame = Frame::General(*h, *hinv);
        for j in i..n {
            out[j] = frame.wrap(diffs[j]);
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
            Cell::from_vectors(
                [9.0, 3.0, 2.0],
                [1.0, 9.0, -2.0],
                [-1.5, 2.0, 9.5],
                [0.0; 3],
            )
            .unwrap(),
        ]
    }

    /// Batch entries equal the per-pair calls bit for bit, including
    /// half-box ties and separations several boxes long.
    #[test]
    fn batches_equal_the_per_pair_calls() {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut unit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for cell in cells() {
            for n in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 37, 203] {
                let span = if n == 37 { 60.0 } else { 12.0 };
                let mut ps = Vec::new();
                let mut qs = Vec::new();
                for k in 0..n {
                    let p = [unit() * 10.0, unit() * 10.0, unit() * 10.0];
                    let mut q = [
                        (unit() - 0.5) * span,
                        (unit() - 0.5) * span,
                        (unit() - 0.5) * span,
                    ];
                    if k % 7 == 3 {
                        let half = cell.cartesian([0.5, -0.5, 0.5]);
                        q = [p[0] + half[0], p[1] + half[1], p[2] + half[2]];
                    }
                    ps.push(p);
                    qs.push(q);
                }
                let mut out = vec![0.0; n];
                crate::dist2_pairs(&cell, &ps, &qs, &mut out).unwrap();
                for k in 0..n {
                    assert_eq!(out[k].to_bits(), cell.dist2(ps[k], qs[k]).to_bits());
                }
                if n > 0 {
                    crate::dist2_many(&cell, ps[0], &qs, &mut out).unwrap();
                    for k in 0..n {
                        assert_eq!(out[k].to_bits(), cell.dist2(ps[0], qs[k]).to_bits());
                    }
                }
                let diffs: Vec<[f64; 3]> = ps
                    .iter()
                    .zip(&qs)
                    .map(|(p, q)| [q[0] - p[0], q[1] - p[1], q[2] - p[2]])
                    .collect();
                let mut wout = vec![[0.0; 3]; n];
                crate::wrap_many(&cell, &diffs, &mut wout).unwrap();
                for k in 0..n {
                    let one = cell.displacement(ps[k], qs[k]);
                    for a in 0..3 {
                        assert!(wout[k][a] == one[a], "{:?} {:?}", wout[k], one);
                    }
                }
            }
        }
    }
}
