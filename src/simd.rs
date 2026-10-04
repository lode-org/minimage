//! SoA orthorhombic squared distances.
//!
//! The arithmetic is Highway `BatchPeriodicDistSq`: one reciprocal per
//! axis, `abs`, then `dr -= L * round(dr / L)`. AVX issues four doubles
//! when the CPU has it. The tail and every other target stay on the
//! scalar kernel. Squared length matches the signed wrap; the sign at
//! a tie does not.

use crate::kernel::wrap_half;

#[cfg(target_arch = "x86_64")]
const ROUND_NEAREST: i32 = 0x08;

#[inline]
fn ortho_diff_one(d: [f64; 3], len: [f64; 3], recip: [f64; 3]) -> f64 {
    let mut acc = 0.0;
    for a in 0..3 {
        let mut x = d[a].abs();
        x -= len[a] * (x * recip[a]).round();
        acc += x * x;
    }
    acc
}

pub(crate) fn dist2_ortho_diffs(
    dx: &[f64],
    dy: &[f64],
    dz: &[f64],
    bx: f64,
    by: f64,
    bz: f64,
    out: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let n = dx.len();
        if n >= 4 && std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected. The three inputs and `out` share length `n`.
            unsafe { ortho_avx(dx, dy, dz, bx, by, bz, out) };
            return;
        }
    }
    dist2_ortho_diffs_scalar(dx, dy, dz, bx, by, bz, out);
}

pub(crate) fn dist2_ortho_diffs_scalar(
    dx: &[f64],
    dy: &[f64],
    dz: &[f64],
    bx: f64,
    by: f64,
    bz: f64,
    out: &mut [f64],
) {
    let len = [bx, by, bz];
    let recip = [1.0 / bx, 1.0 / by, 1.0 / bz];
    for (((x, y), z), slot) in dx.iter().zip(dy).zip(dz).zip(out.iter_mut()) {
        *slot = ortho_diff_one([*x, *y, *z], len, recip);
    }
}

/// Signed AoS wrap. AVX transposes four rows, applies
/// `floor(d/L + 1/2)`, and writes them back. Short lists stay scalar.
pub(crate) fn wrap_many_ortho(l: [f64; 3], diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
    #[cfg(target_arch = "x86_64")]
    {
        let n = diffs.len();
        if n >= 4 && std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected.
            unsafe { wrap_avx(l, diffs, out) };
            return;
        }
    }
    wrap_many_ortho_scalar(l, diffs, out);
}

fn wrap_many_ortho_scalar(l: [f64; 3], diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (d, o) in diffs.iter().zip(out.iter_mut()) {
        *o = [
            wrap_half(d[0], l[0]),
            wrap_half(d[1], l[1]),
            wrap_half(d[2], l[2]),
        ];
    }
}

/// `|q + shift - p|^2` for a bin of occupants. Rapaport's pair: the
/// lattice shift is already applied, so this is a subtract and a dot.
pub(crate) fn dist2_shifted_many(p: [f64; 3], qs: &[[f64; 3]], shift: [f64; 3], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        let n = qs.len();
        if n >= 4 && std::is_x86_feature_detected!("avx") {
            // SAFETY: `avx` was detected.
            unsafe { shifted_avx(p, qs, shift, out) };
            return;
        }
    }
    shifted_scalar(p, qs, shift, out);
}

/// `|positions[indices[k]] + shift - p|^2`. An index past `positions` panics.
pub(crate) fn dist2_shifted_indexed(
    p: [f64; 3],
    positions: &[[f64; 3]],
    indices: &[usize],
    shift: [f64; 3],
    out: &mut [f64],
) {
    let n = indices.len();
    #[cfg(target_arch = "x86_64")]
    {
        let flat_len = positions.len().saturating_mul(3);
        if n >= 8 && flat_len <= i32::MAX as usize && std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was detected, and every index is checked inside.
            unsafe { shifted_gather_avx2(p, positions, indices, shift, out) };
            return;
        }
    }
    shifted_indexed_scalar(p, positions, indices, shift, out);
}

fn shifted_indexed_scalar(
    p: [f64; 3],
    positions: &[[f64; 3]],
    indices: &[usize],
    shift: [f64; 3],
    out: &mut [f64],
) {
    for (idx, slot) in indices.iter().zip(out.iter_mut()) {
        let q = positions[*idx];
        let dx = q[0] + shift[0] - p[0];
        let dy = q[1] + shift[1] - p[1];
        let dz = q[2] + shift[2] - p[2];
        *slot = dx * dx + dy * dy + dz * dz;
    }
}

fn shifted_scalar(p: [f64; 3], qs: &[[f64; 3]], shift: [f64; 3], out: &mut [f64]) {
    for (q, o) in qs.iter().zip(out.iter_mut()) {
        let dx = q[0] + shift[0] - p[0];
        let dy = q[1] + shift[1] - p[1];
        let dz = q[2] + shift[2] - p[2];
        *o = dx * dx + dy * dy + dz * dz;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn ortho_avx(
    dx: &[f64],
    dy: &[f64],
    dz: &[f64],
    bx: f64,
    by: f64,
    bz: f64,
    out: &mut [f64],
) {
    use std::arch::x86_64::*;
    let n = dx.len();
    let vbx = _mm256_set1_pd(bx);
    let vby = _mm256_set1_pd(by);
    let vbz = _mm256_set1_pd(bz);
    let vrbx = _mm256_set1_pd(1.0 / bx);
    let vrby = _mm256_set1_pd(1.0 / by);
    let vrbz = _mm256_set1_pd(1.0 / bz);
    let sign = _mm256_set1_pd(-0.0);
    let mut i = 0usize;
    while i + 4 <= n {
        let x = _mm256_andnot_pd(sign, _mm256_loadu_pd(dx.as_ptr().add(i)));
        let y = _mm256_andnot_pd(sign, _mm256_loadu_pd(dy.as_ptr().add(i)));
        let z = _mm256_andnot_pd(sign, _mm256_loadu_pd(dz.as_ptr().add(i)));
        let x = _mm256_sub_pd(
            x,
            _mm256_mul_pd(vbx, _mm256_round_pd(_mm256_mul_pd(x, vrbx), ROUND_NEAREST)),
        );
        let y = _mm256_sub_pd(
            y,
            _mm256_mul_pd(vby, _mm256_round_pd(_mm256_mul_pd(y, vrby), ROUND_NEAREST)),
        );
        let z = _mm256_sub_pd(
            z,
            _mm256_mul_pd(vbz, _mm256_round_pd(_mm256_mul_pd(z, vrbz), ROUND_NEAREST)),
        );
        let r2 = _mm256_add_pd(
            _mm256_mul_pd(x, x),
            _mm256_add_pd(_mm256_mul_pd(y, y), _mm256_mul_pd(z, z)),
        );
        _mm256_storeu_pd(out.as_mut_ptr().add(i), r2);
        i += 4;
    }
    if i < n {
        dist2_ortho_diffs_scalar(&dx[i..], &dy[i..], &dz[i..], bx, by, bz, &mut out[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn wrap_avx(l: [f64; 3], diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
    use std::arch::x86_64::*;
    let n = diffs.len();
    let inv = [
        _mm256_set1_pd(1.0 / l[0]),
        _mm256_set1_pd(1.0 / l[1]),
        _mm256_set1_pd(1.0 / l[2]),
    ];
    let len = [
        _mm256_set1_pd(l[0]),
        _mm256_set1_pd(l[1]),
        _mm256_set1_pd(l[2]),
    ];
    let half = _mm256_set1_pd(0.5);
    let mut i = 0usize;
    while i + 4 <= n {
        let mut cols = [[0.0; 4]; 3];
        for k in 0..4 {
            cols[0][k] = diffs[i + k][0];
            cols[1][k] = diffs[i + k][1];
            cols[2][k] = diffs[i + k][2];
        }
        let mut wrapped = [[0.0; 4]; 3];
        for a in 0..3 {
            let v = _mm256_loadu_pd(cols[a].as_ptr());
            let q = _mm256_floor_pd(_mm256_add_pd(_mm256_mul_pd(v, inv[a]), half));
            let w = _mm256_sub_pd(v, _mm256_mul_pd(len[a], q));
            _mm256_storeu_pd(wrapped[a].as_mut_ptr(), w);
        }
        for k in 0..4 {
            out[i + k] = [wrapped[0][k], wrapped[1][k], wrapped[2][k]];
        }
        i += 4;
    }
    wrap_many_ortho_scalar(l, &diffs[i..], &mut out[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn shifted_avx(p: [f64; 3], qs: &[[f64; 3]], shift: [f64; 3], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = qs.len();
    let ox = _mm256_set1_pd(p[0] - shift[0]);
    let oy = _mm256_set1_pd(p[1] - shift[1]);
    let oz = _mm256_set1_pd(p[2] - shift[2]);
    let mut i = 0usize;
    while i + 4 <= n {
        let mut x = [0.0; 4];
        let mut y = [0.0; 4];
        let mut z = [0.0; 4];
        for k in 0..4 {
            x[k] = qs[i + k][0];
            y[k] = qs[i + k][1];
            z[k] = qs[i + k][2];
        }
        let dx = _mm256_sub_pd(_mm256_loadu_pd(x.as_ptr()), ox);
        let dy = _mm256_sub_pd(_mm256_loadu_pd(y.as_ptr()), oy);
        let dz = _mm256_sub_pd(_mm256_loadu_pd(z.as_ptr()), oz);
        let r2 = _mm256_add_pd(
            _mm256_mul_pd(dx, dx),
            _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)),
        );
        _mm256_storeu_pd(out.as_mut_ptr().add(i), r2);
        i += 4;
    }
    shifted_scalar(p, &qs[i..], shift, &mut out[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shifted_gather_avx2(
    p: [f64; 3],
    positions: &[[f64; 3]],
    indices: &[usize],
    shift: [f64; 3],
    out: &mut [f64],
) {
    use std::arch::x86_64::*;
    let n = indices.len();
    let base = positions.as_ptr() as *const f64;
    let ox = _mm256_set1_pd(p[0] - shift[0]);
    let oy = _mm256_set1_pd(p[1] - shift[1]);
    let oz = _mm256_set1_pd(p[2] - shift[2]);
    let npos = positions.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let i0 = indices[i];
        let i1 = indices[i + 1];
        let i2 = indices[i + 2];
        let i3 = indices[i + 3];
        if i0 >= npos || i1 >= npos || i2 >= npos || i3 >= npos {
            shifted_indexed_scalar(p, positions, &indices[i..], shift, &mut out[i..]);
            return;
        }
        let vindex = _mm_set_epi32(
            (i3 * 3) as i32,
            (i2 * 3) as i32,
            (i1 * 3) as i32,
            (i0 * 3) as i32,
        );
        let x = _mm256_i32gather_pd(base, vindex, 8);
        let y = _mm256_i32gather_pd(base.add(1), vindex, 8);
        let z = _mm256_i32gather_pd(base.add(2), vindex, 8);
        let dx = _mm256_sub_pd(x, ox);
        let dy = _mm256_sub_pd(y, oy);
        let dz = _mm256_sub_pd(z, oz);
        let r2 = _mm256_add_pd(
            _mm256_mul_pd(dx, dx),
            _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)),
        );
        _mm256_storeu_pd(out.as_mut_ptr().add(i), r2);
        i += 4;
    }
    if i < n {
        shifted_indexed_scalar(p, positions, &indices[i..], shift, &mut out[i..]);
    }
}
