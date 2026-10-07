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

/// `out[k] = |wrap(qs[k] - ps[k])|^2`.
pub(crate) fn dist2_pairs(frame: Frame, ps: &[[f64; 3]], qs: &[[f64; 3]], out: &mut [f64]) {
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

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    use super::{sub, Frame};
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
                None => {
                    for j in i..i + 4 {
                        out[j] = frame.dist2(sub(qs[j], ps[j]));
                    }
                }
            }
            i += 4;
        }
        for j in i..n {
            out[j] = frame.dist2(sub(qs[j], ps[j]));
        }
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
                None => {
                    for j in i..i + 4 {
                        out[j] = frame.dist2(sub(qs[j], p));
                    }
                }
            }
            i += 4;
        }
        for j in i..n {
            out[j] = frame.dist2(sub(qs[j], p));
        }
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
                None => {
                    for j in i..i + 4 {
                        out[j] = frame.wrap(diffs[j]);
                    }
                }
            }
            i += 4;
        }
        for j in i..n {
            out[j] = frame.wrap(diffs[j]);
        }
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
            for n in [0usize, 1, 3, 4, 5, 8, 37, 203] {
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
