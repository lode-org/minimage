//! Runtime SIMD path, and the SoA orthorhombic squared distances.
//!
//! [`select`] picks one [`Path`] per batch call. `MINIMAGE_SIMD` forces
//! `scalar`, `avx`, `avx2`, `avx512`, or `neon`. An unknown name is
//! ignored. A name this CPU cannot run selects the scalar path. Unset,
//! the widest usable path is used: AVX-512F+DQ, then AVX2 with FMA,
//! then AVX, then NEON, then scalar.
//!
//! AVX2 is taken only when FMA is present too. FMA is a selection gate:
//! the kernels do not enable it, and they multiply and add separately.
//! NEON is baseline on aarch64 (`cfg!(target_feature = "neon")`).
//!
//! The orthorhombic arithmetic is Highway `BatchPeriodicDistSq`: one
//! reciprocal per axis, `abs`, then `dr -= L * round(dr / L)`. The tail
//! of a wide kernel runs the scalar loop, in the same element order.

use std::cell::Cell;
use std::sync::OnceLock;

/// Width of a batch kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// One row at a time.
    Scalar,
    /// Four f64 lanes. Requires AVX.
    Avx,
    /// Four f64 lanes, integer kernels. Requires AVX2 and FMA.
    Avx2,
    /// Eight f64 lanes. Requires AVX-512F and AVX-512DQ.
    Avx512,
    /// Two f64 lanes. Requires NEON.
    Neon,
}

impl Path {
    /// Stable name for logs and [`simd_path`].
    pub(crate) fn name(self) -> &'static str {
        match self {
            Path::Scalar => "scalar",
            Path::Avx => "avx",
            Path::Avx2 => "avx2",
            Path::Avx512 => "avx512",
            Path::Neon => "neon",
        }
    }

    /// This build contains the kernel and this CPU can run it.
    pub(crate) fn usable(self) -> bool {
        match self {
            Path::Scalar => true,
            Path::Avx => avx_ready(),
            Path::Avx2 => avx2_ready(),
            Path::Avx512 => avx512_ready(),
            Path::Neon => neon_ready(),
        }
    }
}

thread_local! {
    // `const { ... }` is newer than this crate's 1.70 floor.
    #[allow(clippy::missing_const_for_thread_local)]
    static FORCE: Cell<Option<Path>> = Cell::new(None);
}

/// Path for the next batch.
///
/// A test override wins, then `MINIMAGE_SIMD`, then [`native`].
/// The environment variable is read once. Call this on the thread that
/// owns the batch, before any worker is spawned, and not inside a lane
/// loop.
pub(crate) fn select() -> Path {
    if let Some(path) = FORCE.with(|slot| slot.get()) {
        return if path.usable() { path } else { Path::Scalar };
    }
    if let Some(path) = env_path() {
        return path;
    }
    native()
}

/// Widest path this CPU can run. Ignores `MINIMAGE_SIMD`.
pub(crate) fn native() -> Path {
    static CACHED: OnceLock<Path> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if avx512_ready() {
            Path::Avx512
        } else if avx2_ready() {
            Path::Avx2
        } else if avx_ready() {
            Path::Avx
        } else if neon_ready() {
            Path::Neon
        } else {
            Path::Scalar
        }
    })
}

fn env_path() -> Option<Path> {
    static CACHED: OnceLock<Option<Path>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("MINIMAGE_SIMD").ok()?;
        let path = parse(raw.trim())?;
        Some(if path.usable() { path } else { Path::Scalar })
    })
}

pub(crate) fn parse(name: &str) -> Option<Path> {
    match name {
        "scalar" | "0" => Some(Path::Scalar),
        "avx" => Some(Path::Avx),
        "avx2" | "avx2fma" => Some(Path::Avx2),
        "avx512" | "avx512f" => Some(Path::Avx512),
        "neon" => Some(Path::Neon),
        _ => None,
    }
}

fn avx_ready() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx2_ready() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx512_ready() -> bool {
    #[cfg(all(target_arch = "x86_64", minimage_avx512))]
    {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
    }
    #[cfg(not(all(target_arch = "x86_64", minimage_avx512)))]
    {
        false
    }
}

fn neon_ready() -> bool {
    // NEON is baseline on aarch64. A runtime check would not change the answer.
    cfg!(target_feature = "neon")
}

/// Name of the SIMD path the next batch will use.
///
/// One of `scalar`, `avx`, `avx2`, `avx512`, `neon`.
/// `MINIMAGE_SIMD` overrides the choice. An unknown name is ignored.
/// A name this CPU cannot run selects `scalar`.
pub fn simd_path() -> &'static str {
    select().name()
}

/// Restores the previous test override, including on panic.
#[cfg(test)]
pub(crate) struct ForceGuard {
    prev: Option<Path>,
}

/// Force [`select`] on this thread until the guard drops.
///
/// Workers do not see this thread's override. A batch that fans out
/// must read [`select`] on the calling thread and pass the path down.
#[cfg(test)]
pub(crate) fn force(path: Path) -> ForceGuard {
    let prev = FORCE.with(|slot| slot.replace(Some(path)));
    ForceGuard { prev }
}

#[cfg(test)]
impl Drop for ForceGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        FORCE.with(|slot| slot.set(prev));
    }
}

/// Scalar plus every wide path this CPU can run.
#[cfg(test)]
pub(crate) fn paths_to_compare() -> Vec<Path> {
    let mut out = vec![Path::Scalar];
    for path in [Path::Avx, Path::Avx2, Path::Avx512, Path::Neon] {
        if path.usable() {
            out.push(path);
        }
    }
    out
}

#[inline]
fn ortho_diff_one(d: [f64; 3], len: [f64; 3], recip: [f64; 3]) -> f64 {
    let mut acc = 0.0;
    for a in 0..3 {
        let mut x = d[a].abs();
        x -= len[a] * crate::kernel::round_away(x * recip[a]);
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
    let path = select();
    #[cfg(target_arch = "x86_64")]
    if matches!(path, Path::Avx | Path::Avx2 | Path::Avx512) {
        // SAFETY: AVX was detected. The three inputs and `out` share length `n`.
        unsafe { ortho_avx(dx, dy, dz, bx, by, bz, out) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if path == Path::Neon {
        // SAFETY: NEON is baseline on this target. The slices share one length.
        unsafe { neon::ortho(dx, dy, dz, bx, by, bz, out) };
        return;
    }
    let _ = path;
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

/// `|q + shift - p|^2` for a bin of occupants. Rapaport's pair: the
/// lattice shift is already applied, so this is a subtract and a dot.
pub(crate) fn dist2_shifted_many(p: [f64; 3], qs: &[[f64; 3]], shift: [f64; 3], out: &mut [f64]) {
    let path = select();
    #[cfg(target_arch = "x86_64")]
    if matches!(path, Path::Avx | Path::Avx2 | Path::Avx512) {
        // SAFETY: AVX was detected.
        unsafe { shifted_avx(p, qs, shift, out) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if path == Path::Neon {
        // SAFETY: NEON is baseline on this target.
        unsafe { neon::shifted(p, qs, shift, out) };
        return;
    }
    let _ = path;
    shifted_scalar(p, qs, shift, out);
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
    let half = _mm256_set1_pd(0.499_999_999_999_999_94);
    let mut i = 0usize;
    while i + 4 <= n {
        let x = _mm256_andnot_pd(sign, _mm256_loadu_pd(dx.as_ptr().add(i)));
        let y = _mm256_andnot_pd(sign, _mm256_loadu_pd(dy.as_ptr().add(i)));
        let z = _mm256_andnot_pd(sign, _mm256_loadu_pd(dz.as_ptr().add(i)));
        let round = |v: __m256d| {
            let signed = _mm256_or_pd(_mm256_and_pd(v, sign), half);
            _mm256_round_pd(_mm256_add_pd(v, signed), 0x0B)
        };
        let x = _mm256_sub_pd(x, _mm256_mul_pd(vbx, round(_mm256_mul_pd(x, vrbx))));
        let y = _mm256_sub_pd(y, _mm256_mul_pd(vby, round(_mm256_mul_pd(y, vrby))));
        let z = _mm256_sub_pd(z, _mm256_mul_pd(vbz, round(_mm256_mul_pd(z, vrbz))));
        let x2 = _mm256_mul_pd(x, x);
        let y2 = _mm256_mul_pd(y, y);
        let z2 = _mm256_mul_pd(z, z);
        let r2 = _mm256_add_pd(_mm256_add_pd(x2, y2), z2);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), r2);
        i += 4;
    }
    if i < n {
        dist2_ortho_diffs_scalar(&dx[i..], &dy[i..], &dz[i..], bx, by, bz, &mut out[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn shifted_avx(p: [f64; 3], qs: &[[f64; 3]], shift: [f64; 3], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = qs.len();
    let px = _mm256_set1_pd(p[0]);
    let py = _mm256_set1_pd(p[1]);
    let pz = _mm256_set1_pd(p[2]);
    let sx = _mm256_set1_pd(shift[0]);
    let sy = _mm256_set1_pd(shift[1]);
    let sz = _mm256_set1_pd(shift[2]);
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
        let dx = _mm256_sub_pd(_mm256_add_pd(_mm256_loadu_pd(x.as_ptr()), sx), px);
        let dy = _mm256_sub_pd(_mm256_add_pd(_mm256_loadu_pd(y.as_ptr()), sy), py);
        let dz = _mm256_sub_pd(_mm256_add_pd(_mm256_loadu_pd(z.as_ptr()), sz), pz);
        let r2 = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
            _mm256_mul_pd(dz, dz),
        );
        _mm256_storeu_pd(out.as_mut_ptr().add(i), r2);
        i += 4;
    }
    shifted_scalar(p, &qs[i..], shift, &mut out[i..]);
}

/// Two lanes, the same arithmetic as the scalar loops.
///
/// [`ortho`] runs two independent pairs. After `abs` the quotient is
/// non-negative, so each pair rounds with `trunc(q + (1/2 - eps))`
/// and sums `(x^2 + y^2) + z^2`. A lane at `2^52` or a NaN sends that
/// group through the scalar loop, which is what `kernel::round_away` does.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::{dist2_ortho_diffs_scalar, shifted_scalar};

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn norm(x: float64x2_t, y: float64x2_t, z: float64x2_t) -> float64x2_t {
        let x2 = vmulq_f64(x, x);
        let y2 = vmulq_f64(y, y);
        let z2 = vmulq_f64(z, z);
        vaddq_f64(vaddq_f64(x2, y2), z2)
    }

    /// `trunc(q + (1/2 - eps))` for a non-negative quotient below `2^52`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn trunc_pos(q: float64x2_t, half: float64x2_t) -> float64x2_t {
        vcvtq_f64_s64(vcvtq_s64_f64(vaddq_f64(q, half)))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn in_fast(q: float64x2_t, lim: float64x2_t) -> uint64x2_t {
        vcltq_f64(q, lim)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn lanes_fast(m: uint64x2_t) -> bool {
        vgetq_lane_u64(m, 0) == u64::MAX && vgetq_lane_u64(m, 1) == u64::MAX
    }

    /// One pair: absolute quotient, fast rounding, square in scalar order.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn pair(
        x: float64x2_t,
        y: float64x2_t,
        z: float64x2_t,
        len: [float64x2_t; 3],
        recip: [float64x2_t; 3],
        half: float64x2_t,
        lim: float64x2_t,
    ) -> (float64x2_t, uint64x2_t) {
        let x = vabsq_f64(x);
        let y = vabsq_f64(y);
        let z = vabsq_f64(z);
        let qx = vmulq_f64(x, recip[0]);
        let qy = vmulq_f64(y, recip[1]);
        let qz = vmulq_f64(z, recip[2]);
        let ok = vandq_u64(
            vandq_u64(in_fast(qx, lim), in_fast(qy, lim)),
            in_fast(qz, lim),
        );
        let x = vsubq_f64(x, vmulq_f64(len[0], trunc_pos(qx, half)));
        let y = vsubq_f64(y, vmulq_f64(len[1], trunc_pos(qy, half)));
        let z = vsubq_f64(z, vmulq_f64(len[2], trunc_pos(qz, half)));
        (norm(x, y, z), ok)
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn ortho(
        dx: &[f64],
        dy: &[f64],
        dz: &[f64],
        bx: f64,
        by: f64,
        bz: f64,
        out: &mut [f64],
    ) {
        let n = dx.len();
        let len = [vdupq_n_f64(bx), vdupq_n_f64(by), vdupq_n_f64(bz)];
        let recip = [
            vdupq_n_f64(1.0 / bx),
            vdupq_n_f64(1.0 / by),
            vdupq_n_f64(1.0 / bz),
        ];
        let half = vdupq_n_f64(0.499_999_999_999_999_94);
        let lim = vdupq_n_f64(crate::kernel::INTEGRAL);
        let mut i = 0usize;
        while i + 4 <= n {
            let (r0, ok0) = pair(
                vld1q_f64(dx.as_ptr().add(i)),
                vld1q_f64(dy.as_ptr().add(i)),
                vld1q_f64(dz.as_ptr().add(i)),
                len,
                recip,
                half,
                lim,
            );
            let (r1, ok1) = pair(
                vld1q_f64(dx.as_ptr().add(i + 2)),
                vld1q_f64(dy.as_ptr().add(i + 2)),
                vld1q_f64(dz.as_ptr().add(i + 2)),
                len,
                recip,
                half,
                lim,
            );
            if lanes_fast(ok0) && lanes_fast(ok1) {
                vst1q_f64(out.as_mut_ptr().add(i), r0);
                vst1q_f64(out.as_mut_ptr().add(i + 2), r1);
            } else {
                dist2_ortho_diffs_scalar(
                    &dx[i..i + 4],
                    &dy[i..i + 4],
                    &dz[i..i + 4],
                    bx,
                    by,
                    bz,
                    &mut out[i..i + 4],
                );
            }
            i += 4;
        }
        if i + 2 <= n {
            let (r0, ok0) = pair(
                vld1q_f64(dx.as_ptr().add(i)),
                vld1q_f64(dy.as_ptr().add(i)),
                vld1q_f64(dz.as_ptr().add(i)),
                len,
                recip,
                half,
                lim,
            );
            if lanes_fast(ok0) {
                vst1q_f64(out.as_mut_ptr().add(i), r0);
            } else {
                dist2_ortho_diffs_scalar(
                    &dx[i..i + 2],
                    &dy[i..i + 2],
                    &dz[i..i + 2],
                    bx,
                    by,
                    bz,
                    &mut out[i..i + 2],
                );
            }
            i += 2;
        }
        if i < n {
            dist2_ortho_diffs_scalar(&dx[i..], &dy[i..], &dz[i..], bx, by, bz, &mut out[i..]);
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn shifted(p: [f64; 3], qs: &[[f64; 3]], shift: [f64; 3], out: &mut [f64]) {
        let n = qs.len();
        let px = vdupq_n_f64(p[0]);
        let py = vdupq_n_f64(p[1]);
        let pz = vdupq_n_f64(p[2]);
        let sx = vdupq_n_f64(shift[0]);
        let sy = vdupq_n_f64(shift[1]);
        let sz = vdupq_n_f64(shift[2]);
        let mut i = 0usize;
        while i + 2 <= n {
            let mut x = [0.0; 2];
            let mut y = [0.0; 2];
            let mut z = [0.0; 2];
            for k in 0..2 {
                x[k] = qs[i + k][0];
                y[k] = qs[i + k][1];
                z[k] = qs[i + k][2];
            }
            let dx = vsubq_f64(vaddq_f64(vld1q_f64(x.as_ptr()), sx), px);
            let dy = vsubq_f64(vaddq_f64(vld1q_f64(y.as_ptr()), sy), py);
            let dz = vsubq_f64(vaddq_f64(vld1q_f64(z.as_ptr()), sz), pz);
            vst1q_f64(out.as_mut_ptr().add(i), norm(dx, dy, dz));
            i += 2;
        }
        shifted_scalar(p, &qs[i..], shift, &mut out[i..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_names() {
        assert_eq!(parse("scalar"), Some(Path::Scalar));
        assert_eq!(parse("0"), Some(Path::Scalar));
        assert_eq!(parse("avx"), Some(Path::Avx));
        assert_eq!(parse("avx2"), Some(Path::Avx2));
        assert_eq!(parse("avx2fma"), Some(Path::Avx2));
        assert_eq!(parse("avx512"), Some(Path::Avx512));
        assert_eq!(parse("avx512f"), Some(Path::Avx512));
        assert_eq!(parse("neon"), Some(Path::Neon));
        assert_eq!(parse("sve"), None);
        assert_eq!(parse(""), None);
        assert_eq!(parse("AVX2"), None);
    }

    #[test]
    fn native_path_is_usable() {
        let path = native();
        assert!(path.usable());
        assert_eq!(path.name(), simd_path());
    }

    #[test]
    fn force_overrides_this_thread_only() {
        let prev = select();
        {
            let _guard = force(Path::Scalar);
            assert_eq!(select(), Path::Scalar);
        }
        assert_eq!(select(), prev);
        let _unused = force(Path::Neon);
        if Path::Neon.usable() {
            assert_eq!(select(), Path::Neon);
        } else {
            assert_eq!(select(), Path::Scalar);
        }
    }
}
