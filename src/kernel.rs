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

/// Restricted-triclinic edges. `lx, ly, lz` are the diagonal entries
/// and may be negative. `xy, xz, yz` are the LAMMPS tilt factors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tri {
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub xy: f64,
    pub xz: f64,
    pub yz: f64,
    pub inv_lx: f64,
    pub inv_ly: f64,
    pub inv_lz: f64,
}

impl Tri {
    /// Wrapped `dp` in Cartesian coordinates.
    #[inline(always)]
    pub(crate) fn wrap(self, dp: [f64; 3]) -> [f64; 3] {
        let mut sz = dp[2] * self.inv_lz;
        let mut sy = (dp[1] - self.yz * sz) * self.inv_ly;
        let mut sx = (dp[0] - self.xy * sy - self.xz * sz) * self.inv_lx;
        sx -= round_away(sx);
        sy -= round_away(sy);
        sz -= round_away(sz);
        [
            self.lx * sx + self.xy * sy + self.xz * sz,
            self.ly * sy + self.yz * sz,
            self.lz * sz,
        ]
    }
}

/// Below this magnitude a double may have a fractional part.
const INTEGRAL: f64 = 4503599627370496.0;

/// `f64::round`, half away from zero, without the libm call.
///
/// Baseline x86-64 has no rounding instruction, so `f64::round` is a
/// call. Adding the largest double below one half, with the sign of
/// `x`, and truncating gives the same integer for every double; LLVM
/// expands `round` the same way when SSE4.1 is present. Truncation is
/// `cvttsd2si`, which is baseline. Other targets have the instruction.
#[inline(always)]
pub(crate) fn round_away(x: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if x.abs() < INTEGRAL {
            let t = x + 0.499_999_999_999_999_94_f64.copysign(x);
            // SAFETY: |t| < 2^52 + 1 is inside the i64 range.
            return unsafe { t.to_int_unchecked::<i64>() } as f64;
        }
        x
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        x.round()
    }
}

/// `f64::floor` without the libm call on baseline x86-64.
#[inline(always)]
pub(crate) fn floor_fast(x: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if x.abs() < INTEGRAL {
            // SAFETY: |x| < 2^52 is inside the i64 range.
            let t = unsafe { x.to_int_unchecked::<i64>() } as f64;
            return if t > x { t - 1.0 } else { t };
        }
        x
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        x.floor()
    }
}

/// Orthorhombic signed wrap into `[-L/2, L/2)`.
///
/// `d - L * floor(d / L + 1/2)`. Every image, not one subtraction.
/// Keeps `-L/2` and maps `+L/2` onto `-L/2`.
#[inline(always)]
pub(crate) fn wrap_half(d: f64, length: f64) -> f64 {
    if d.abs() >= length {
        return wrap_half_far(d, length);
    }
    wrap_one(d, length)
}

/// One neighbouring image. Exact for `|d| < L`, which is every pair of
/// positions folded into the box. Two selects, no branch.
#[inline(always)]
pub(crate) fn wrap_one(d: f64, length: f64) -> f64 {
    let half = 0.5 * length;
    let w = if d < -half { d + length } else { d };
    if w >= half {
        w - length
    } else {
        w
    }
}

/// Orthorhombic wrap of one difference. A pair of folded positions is
/// inside one box on every axis, so the three selects run without a
/// branch; one combined test sends anything longer to the floor form.
#[inline(always)]
pub(crate) fn ortho_wrap(l: [f64; 3], dp: [f64; 3]) -> [f64; 3] {
    let far = (dp[0].abs() >= l[0]) | (dp[1].abs() >= l[1]) | (dp[2].abs() >= l[2]);
    if far {
        return ortho_wrap_far(l[0], l[1], l[2], dp[0], dp[1], dp[2]);
    }
    [
        wrap_one(dp[0], l[0]),
        wrap_one(dp[1], l[1]),
        wrap_one(dp[2], l[2]),
    ]
}

/// Squared orthorhombic minimum image. For `|d| < L` the wrapped
/// length on an axis is `min(|d|, L - |d|)`: `L - |d|` is exactly
/// `|d - L|`, and at `|d| = L/2` both are `L/2`, so the square is the
/// signed wrap's square bit for bit with a third of the selects.
#[inline(always)]
pub(crate) fn ortho_dist2(l: [f64; 3], dp: [f64; 3]) -> f64 {
    let ax = dp[0].abs();
    let ay = dp[1].abs();
    let az = dp[2].abs();
    if (ax >= l[0]) | (ay >= l[1]) | (az >= l[2]) {
        return n2(ortho_wrap_far(l[0], l[1], l[2], dp[0], dp[1], dp[2]));
    }
    let bx = l[0] - ax;
    let by = l[1] - ay;
    let bz = l[2] - az;
    let wx = if bx < ax { bx } else { ax };
    let wy = if by < ay { by } else { ay };
    let wz = if bz < az { bz } else { az };
    wx * wx + wy * wy + wz * wz
}

#[cold]
#[inline(never)]
fn ortho_wrap_far(lx: f64, ly: f64, lz: f64, dx: f64, dy: f64, dz: f64) -> [f64; 3] {
    [wrap_half(dx, lx), wrap_half(dy, ly), wrap_half(dz, lz)]
}

/// Images past one neighbouring cell. The floor estimate can miss by
/// one image at a tie; the two exact selects put it back.
#[cold]
#[inline(never)]
fn wrap_half_far(d: f64, length: f64) -> f64 {
    let w = d - length * floor_fast(d / length + 0.5);
    let half = 0.5 * length;
    let w = if w < -half { w + length } else { w };
    if w >= half {
        w - length
    } else {
        w
    }
}

#[inline(always)]
pub(crate) fn general_wrap(h: [[f64; 3]; 3], hinv: [[f64; 3]; 3], dp: [f64; 3]) -> [f64; 3] {
    let mut ds = mul(hinv, dp);
    ds[0] -= round_away(ds[0]);
    ds[1] -= round_away(ds[1]);
    ds[2] -= round_away(ds[2]);
    mul(h, ds)
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
    fn wrap_matches_the_division_form() {
        let l = [10.0, 11.0, 12.0];
        let samples = [
            [0.0, 0.0, 0.0],
            [5.0, -5.5, 6.0],
            [15.0, -18.0, 25.0],
            [-15.0, 16.5, -30.0],
            [6.0, 12.0, -6.0],
        ];
        for dp in samples {
            let via = ortho_wrap(l, dp);
            for a in 0..3 {
                let floor = dp[a] - l[a] * (dp[a] / l[a] + 0.5).floor();
                assert!((wrap_half(dp[a], l[a]) - floor).abs() < 1e-12);
                assert!((via[a] - floor).abs() < 1e-12);
            }
        }
    }

    /// Doubles near every half integer, near 2^52, signed zeros,
    /// subnormals, and random bit patterns.
    fn awkward_doubles() -> Vec<f64> {
        let mut xs = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            0.499_999_999_999_999_94,
            -0.499_999_999_999_999_94,
            1.5,
            2.5,
            -2.5,
            4_503_599_627_370_495.5,
            -4_503_599_627_370_495.5,
            4_503_599_627_370_496.0,
            9_007_199_254_740_993.0,
            1e300,
            -1e300,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE * 0.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for k in -40i32..=40 {
            let h = f64::from(k) + 0.5;
            let mut lo = h;
            let mut hi = h;
            for _ in 0..3 {
                lo = f64::from_bits(lo.to_bits() - 1);
                hi = f64::from_bits(hi.to_bits() + 1);
                xs.extend([lo, hi, -lo, -hi]);
            }
            xs.extend([h, -h]);
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f64::from_bits(state);
            if x.is_finite() {
                xs.push(x);
                xs.push(x * 1e-300);
            }
            xs.push((state >> 11) as f64 / (1u64 << 40) as f64 - 4096.0);
        }
        xs
    }

    #[test]
    fn round_and_floor_match_libm() {
        for x in awkward_doubles() {
            assert_eq!(round_away(x), x.round(), "round {x:e}");
            assert_eq!(floor_fast(x), x.floor(), "floor {x:e}");
        }
        assert!(round_away(f64::NAN).is_nan());
        assert!(floor_fast(f64::NAN).is_nan());
    }

    #[test]
    fn ortho_square_is_the_signed_wrap_square() {
        let l = [10.0, 11.0, 12.0];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut unit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut cases = vec![
            [5.0, -5.5, 6.0],
            [-5.0, 5.5, -6.0],
            [10.0, -11.0, 12.0],
            [15.0, -16.5, 18.0],
            [-15.0, 16.5, -18.0],
            [9.999_999_999_999_998, 0.0, 0.0],
        ];
        for _ in 0..4000 {
            cases.push([
                (unit() - 0.5) * 6.0 * l[0],
                (unit() - 0.5) * 2.0 * l[1],
                (unit() - 0.5) * 2.0 * l[2],
            ]);
        }
        for dp in cases {
            let signed = ortho_wrap(l, dp);
            for a in 0..3 {
                assert!(signed[a] >= -0.5 * l[a] && signed[a] < 0.5 * l[a], "{dp:?}");
            }
            assert_eq!(ortho_dist2(l, dp).to_bits(), n2(signed).to_bits(), "{dp:?}");
        }
    }
}
