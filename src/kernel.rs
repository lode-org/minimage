//! Engine minimum image: orthorhombic nearest integer, restricted
//! triangular lamda, and a general `H` wrap.
//!
//! Orthorhombic form, Allen and Tildesley and Frenkel and Smit:
//! `d - L * floor(d / L + 1/2)`. One floor covers every image. The
//! tie keeps `-L/2` and maps `+L/2` onto `-L/2` (dump `relDist`).
//!
//! Restricted triclinic form, LAMMPS `Domain::minimum_image`, HOOMD
//! `BoxDim::minImage`, GROMACS `pbc_dx`: `a` along x, `b` in the xy
//! plane. The lamda step is the upper-triangular inverse, then
//! nearest integer, then `H` (Tuckerman; Wassenaar). Nearest integer
//! here is `round`, half away from zero, matching the general `H`
//! path and eOn.
//!
//! A general orientation is two 3x3 products with the stored inverse.

/// Positive orthorhombic lengths when `H` is diagonal, else `None`.
pub(crate) fn ortho_lengths(h: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    if !is_axis_aligned(h) {
        return None;
    }
    let l = [h[0][0].abs(), h[1][1].abs(), h[2][2].abs()];
    if l[0] > 0.0 && l[1] > 0.0 && l[2] > 0.0 {
        Some(l)
    } else {
        None
    }
}

#[inline]
pub(crate) fn is_axis_aligned(h: [[f64; 3]; 3]) -> bool {
    // Exact zeros are the orthorhombic constructors.
    if h[0][1] == 0.0
        && h[0][2] == 0.0
        && h[1][0] == 0.0
        && h[1][2] == 0.0
        && h[2][0] == 0.0
        && h[2][1] == 0.0
    {
        return true;
    }
    // A real shear fails here. The tolerance is only for a diagonal
    // that roundoff left a few ulps off zero.
    let scale = (h[0][0].abs() + h[1][1].abs() + h[2][2].abs()).max(1.0);
    let tol = 1e-12 * scale;
    h[0][1].abs() <= tol
        && h[0][2].abs() <= tol
        && h[1][0].abs() <= tol
        && h[1][2].abs() <= tol
        && h[2][0].abs() <= tol
        && h[2][1].abs() <= tol
}

/// Orthorhombic signed wrap into `[-L/2, L/2)`.
///
/// `d - L * floor(d / L + 1/2)`. Every image, not one subtraction.
/// Keeps `-L/2` and maps `+L/2` onto `-L/2`.
#[inline(always)]
pub(crate) fn wrap_half(d: f64, length: f64) -> f64 {
    let half = 0.5 * length;
    if d < half {
        if d >= -half {
            return d;
        }
        let w = d + length;
        if w >= -half {
            return w;
        }
    } else {
        let w = d - length;
        if w < half {
            return w;
        }
    }
    d - length * (d / length + 0.5).floor()
}

/// `d - L * floor(d * (1/L) + 1/2)`. The reciprocal is the cell's.
///
/// A separation inside one neighbouring image is two comparisons and
/// one add or subtract, the same shape as the single-image wrap.
/// Anything past that image uses the floor, so a later image still
/// lands in `[-L/2, L/2)`.
#[inline]
pub(crate) fn wrap_half_recip(d: f64, length: f64, recip: f64) -> f64 {
    let half = 0.5 * length;
    if d < half {
        if d >= -half {
            return d;
        }
        let w = d + length;
        if w >= -half {
            return w;
        }
    } else {
        let w = d - length;
        if w < half {
            return w;
        }
    }
    d - length * (d * recip + 0.5).floor()
}

#[inline]
pub(crate) fn ortho_wrap_recip(l: [f64; 3], recip: [f64; 3], dp: [f64; 3]) -> [f64; 3] {
    [
        wrap_half_recip(dp[0], l[0], recip[0]),
        wrap_half_recip(dp[1], l[1], recip[1]),
        wrap_half_recip(dp[2], l[2], recip[2]),
    ]
}

#[inline]
pub(crate) fn general_wrap(h: [[f64; 3]; 3], hinv: [[f64; 3]; 3], dp: [f64; 3]) -> [f64; 3] {
    let mut ds = mul(hinv, dp);
    ds[0] -= ds[0].round();
    ds[1] -= ds[1].round();
    ds[2] -= ds[2].round();
    mul(h, ds)
}

/// Half away from zero, the same value as `f64::round`, using `floorsd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub(crate) unsafe fn nearest_sse(x: f64) -> f64 {
    use std::arch::x86_64::{_mm_cvtsd_f64, _mm_floor_sd, _mm_set_sd};
    let floored = _mm_floor_sd(_mm_set_sd(0.0), _mm_set_sd(x.abs() + 0.5));
    _mm_cvtsd_f64(floored).copysign(x)
}

/// Restricted wrap whose rounding is `floorsd` rather than libm `round`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub(crate) unsafe fn restricted_wrap_sse(
    h: [[f64; 3]; 3],
    hinv: [[f64; 3]; 3],
    dp: [f64; 3],
) -> [f64; 3] {
    let mut sz = hinv[2][2] * dp[2];
    let mut sy = hinv[1][1] * dp[1] + hinv[2][1] * dp[2];
    let mut sx = hinv[0][0] * dp[0] + hinv[1][0] * dp[1] + hinv[2][0] * dp[2];
    sx -= nearest_sse(sx);
    sy -= nearest_sse(sy);
    sz -= nearest_sse(sz);
    [
        h[0][0] * sx + h[1][0] * sy + h[2][0] * sz,
        h[1][1] * sy + h[2][1] * sz,
        h[2][2] * sz,
    ]
}

/// Restricted frame: the inverse and H are triangular, so the wrap
/// skips the zero products of the general 3x3.
#[inline(always)]
pub(crate) fn restricted_wrap(h: [[f64; 3]; 3], hinv: [[f64; 3]; 3], dp: [f64; 3]) -> [f64; 3] {
    let mut sz = hinv[2][2] * dp[2];
    let mut sy = hinv[1][1] * dp[1] + hinv[2][1] * dp[2];
    let mut sx = hinv[0][0] * dp[0] + hinv[1][0] * dp[1] + hinv[2][0] * dp[2];
    sx -= sx.round();
    sy -= sy.round();
    sz -= sz.round();
    [
        h[0][0] * sx + h[1][0] * sy + h[2][0] * sz,
        h[1][1] * sy + h[2][1] * sz,
        h[2][2] * sz,
    ]
}

#[inline]
pub(crate) fn mul(h: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        h[0][0] * v[0] + h[1][0] * v[1] + h[2][0] * v[2],
        h[0][1] * v[0] + h[1][1] * v[1] + h[2][1] * v[2],
        h[0][2] * v[0] + h[1][2] * v[1] + h[2][2] * v[2],
    ]
}

#[inline]
pub(crate) fn invert_columns(h: [[f64; 3]; 3]) -> Option<([[f64; 3]; 3], f64)> {
    let a = h[0];
    let b = h[1];
    let c = h[2];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let invdet = 1.0 / det;
    let inv = [
        [
            (b[1] * c[2] - b[2] * c[1]) * invdet,
            (a[2] * c[1] - a[1] * c[2]) * invdet,
            (a[1] * b[2] - a[2] * b[1]) * invdet,
        ],
        [
            (b[2] * c[0] - b[0] * c[2]) * invdet,
            (a[0] * c[2] - a[2] * c[0]) * invdet,
            (a[2] * b[0] - a[0] * b[2]) * invdet,
        ],
        [
            (b[0] * c[1] - b[1] * c[0]) * invdet,
            (a[1] * c[0] - a[0] * c[1]) * invdet,
            (a[0] * b[1] - a[1] * b[0]) * invdet,
        ],
    ];
    Some((inv, det))
}

#[inline]
pub(crate) fn n2(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

#[inline]
pub(crate) fn norm(v: [f64; 3]) -> f64 {
    n2(v).sqrt()
}

#[inline]
pub(crate) fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

#[inline]
pub(crate) fn add(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
}

#[inline]
pub(crate) fn scale(s: f64, v: [f64; 3]) -> [f64; 3] {
    [s * v[0], s * v[1], s * v[2]]
}

#[inline]
pub(crate) fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recip_wrap_matches_the_division_form() {
        let l = [10.0, 11.0, 12.0];
        let samples = [
            [0.0, 0.0, 0.0],
            [5.0, -5.5, 6.0],
            [15.0, -18.0, 25.0],
            [-15.0, 16.5, -30.0],
            [6.0, 12.0, -6.0],
        ];
        for dp in samples {
            let recip = [1.0 / l[0], 1.0 / l[1], 1.0 / l[2]];
            let direct = ortho_wrap_recip(l, recip, dp);
            let via = [
                wrap_half(dp[0], l[0]),
                wrap_half(dp[1], l[1]),
                wrap_half(dp[2], l[2]),
            ];
            for a in 0..3 {
                let floor = dp[a] - l[a] * (dp[a] / l[a] + 0.5).floor();
                assert!((via[a] - floor).abs() < 1e-12);
                assert!((direct[a] - floor).abs() < 1e-12);
            }
        }
    }
}
