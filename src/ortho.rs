/// Lengths below `2^-1000` are folded after scaling by `2^64`, because
/// `0.5 * length` is inexact for a subnormal length with an odd last bit.
const SMALL_LENGTH: f64 = 9.332_636_185_032_189e-302;
const SMALL_SCALE: f64 = 18_446_744_073_709_551_616.0;

/// Wrap one orthorhombic component into the closed interval
/// `[-length / 2, length / 2]`.
///
/// The result is the nearest periodic image `delta - n * length` for an
/// integer `n`, with no rounding error. A finite `delta` within two
/// lengths, and a length from `2^-1000` through half the largest finite
/// value, is reduced by exact additions and subtractions. Every other
/// input uses the remainder operator, which is exact, then at most one
/// exact fold. A length below `2^-1000` is scaled by `2^64` before that
/// fold so a subnormal half stays exact. Values just inside `+-length / 2`
/// are kept, and multiples far larger than `length` (such as `1e308`
/// against `0.5`) still give a finite, in-range result.
///
/// # Ties
///
/// When the remainder equals `length / 2` exactly in binary, the result
/// keeps the sign of `delta`: `5.0` and `15.0` with length `10.0` give
/// `5.0`, while `-5.0` and `-15.0` give `-5.0`. A tie is judged on the
/// stored `f64` values, so a decimal tie such as `0.45` against `0.3` is
/// not a tie and returns its nearest image, here `-0.14999999999999997`.
/// The sign of a zero result follows `delta`.
///
/// # Invalid input
///
/// Returns `NaN` when `delta` is not finite, or `length` is not finite or
/// not strictly positive. The function never panics.
///
/// ```
/// use minimage::wrap_ortho_half_strict;
///
/// assert_eq!(wrap_ortho_half_strict(6.0, 10.0), -4.0);
/// assert_eq!(wrap_ortho_half_strict(15.0, 10.0), 5.0);
/// assert_eq!(wrap_ortho_half_strict(-15.0, 10.0), -5.0);
/// assert!(wrap_ortho_half_strict(1.0, 0.0).is_nan());
/// ```
#[inline]
pub fn wrap_ortho_half_strict(delta: f64, length: f64) -> f64 {
    // `2 * length` and `length / 2` are exact for every length in this window.
    const MAX_FAST_LENGTH: f64 = f64::from_bits(0x7fdfffffffffffff);
    if length >= SMALL_LENGTH && length <= MAX_FAST_LENGTH && delta.abs() <= length + length {
        let down = mask_length(delta > length, length);
        let up = mask_length(delta < -length, length);
        let mut wrapped = (delta - down) + up;
        let half = 0.5 * length;
        let fold_down = mask_length(wrapped > half, length);
        let fold_up = mask_length(wrapped < -half, length);
        wrapped = (wrapped - fold_down) + fold_up;
        if wrapped == 0.0 {
            wrapped = 0.0_f64.copysign(delta);
        }
        return wrapped;
    }
    wrap_ortho_half_strict_slow(delta, length)
}

/// Fold `|rem| < length` into `[-length / 2, length / 2]`. Each branch
/// subtracts values within a factor of two of each other, so it is exact
/// by Sterbenz's lemma.
#[inline]
fn fold_remainder(rem: f64, length: f64) -> f64 {
    let half = 0.5 * length;
    if rem > half {
        rem - length
    } else if rem < -half {
        rem + length
    } else {
        rem
    }
}

#[inline(always)]
fn mask_length(cond: bool, length: f64) -> f64 {
    f64::from_bits(u64::from(cond).wrapping_neg() & length.to_bits())
}

#[inline(never)]
fn wrap_ortho_half_strict_slow(delta: f64, length: f64) -> f64 {
    if !delta.is_finite() || !length.is_finite() || length <= 0.0 {
        return f64::NAN;
    }
    let rem = delta % length;
    if length < SMALL_LENGTH {
        fold_remainder(rem * SMALL_SCALE, length * SMALL_SCALE) / SMALL_SCALE
    } else {
        fold_remainder(rem, length)
    }
}

#[cfg(test)]
mod tests {
    use super::wrap_ortho_half_strict;

    const SUB: f64 = 5e-324;

    fn same(delta: f64, length: f64, expected: f64) {
        let got = wrap_ortho_half_strict(delta, length);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "wrap({delta:?}, {length:?}) = {got:?}, expected {expected:?}"
        );
    }

    fn in_range(delta: f64, length: f64) {
        let got = wrap_ortho_half_strict(delta, length);
        assert!(
            got.is_finite() && got.abs() <= 0.5 * length,
            "wrap({delta:?}, {length:?}) = {got:?} is outside the interval"
        );
    }

    #[test]
    fn exact_positive_ties_keep_positive_half() {
        same(5.0, 10.0, 5.0);
        same(15.0, 10.0, 5.0);
        same(1.5, 3.0, 1.5);
        same(4.5, 3.0, 1.5);
        same(0.5, 1.0, 0.5);
    }

    #[test]
    fn exact_negative_ties_keep_negative_half() {
        same(-5.0, 10.0, -5.0);
        same(-15.0, 10.0, -5.0);
        same(-1.5, 3.0, -1.5);
        same(-4.5, 3.0, -1.5);
        same(-0.5, 1.0, -0.5);
    }

    #[test]
    fn non_ties_wrap_to_nearest_image() {
        same(4.0, 10.0, 4.0);
        same(6.0, 10.0, -4.0);
        same(-4.0, 10.0, -4.0);
        same(-6.0, 10.0, 4.0);
    }

    #[test]
    fn zero_keeps_its_sign() {
        same(0.0, 10.0, 0.0);
        same(-0.0, 10.0, -0.0);
        same(20.0, 10.0, 0.0);
        same(-20.0, 10.0, -0.0);
    }

    #[test]
    fn values_one_ulp_inside_the_boundary_stay_put() {
        same(1.4999999999999998, 3.0, 1.4999999999999998);
        same(-1.4999999999999998, 3.0, -1.4999999999999998);
        same(0.49999999999999994, 1.0, 0.49999999999999994);
        same(-0.49999999999999994, 1.0, -0.49999999999999994);
        same(0.5000000000000001, 1.0, -0.4999999999999999);
        same(-0.5000000000000001, 1.0, 0.4999999999999999);
    }

    #[test]
    fn decimal_ties_land_inside_the_interval() {
        same(0.45, 0.3, -0.14999999999999997);
        same(-0.75, 0.3, 0.14999999999999997);
        same(-63.45000000000001, 2.7, 1.3499999999999943);
        for (delta, length) in [(0.45, 0.3), (-0.75, 0.3), (-63.45000000000001, 2.7)] {
            in_range(delta, length);
        }
    }

    #[test]
    fn large_multiples_reduce_exactly() {
        same(1e16, 3.0, 1.0);
        same(1e17, 0.3, -0.09925658458281139);
        same(1e308, 0.5, 0.0);
        same(-1e200, 1e-200, -4.18199169208316e-201);
        same(f64::MAX, f64::MAX, 0.0);
        same(1e308, 1.7e308, -6.999999999999999e307);
        same(-f64::MAX, 1e308, 2.023_068_651_376_843_1e307);
        for (delta, length) in [
            (1e16, 3.0),
            (1e17, 0.3),
            (1e308, 0.5),
            (-1e200, 1e-200),
            (f64::MAX, 3.0),
            (-f64::MAX, 0.7),
        ] {
            in_range(delta, length);
        }
    }

    #[test]
    fn subnormal_lengths_fold_exactly() {
        same(SUB, 3.0 * SUB, SUB);
        same(2.0 * SUB, 3.0 * SUB, -SUB);
        same(-2.0 * SUB, 3.0 * SUB, SUB);
        same(1.0, 3.0 * SUB, SUB);
        in_range(1.0, 3.0 * SUB);
        in_range(-1e300, 7.0 * SUB);
    }

    #[test]
    fn non_finite_delta_gives_nan() {
        assert!(wrap_ortho_half_strict(f64::NAN, 10.0).is_nan());
        assert!(wrap_ortho_half_strict(f64::INFINITY, 10.0).is_nan());
        assert!(wrap_ortho_half_strict(f64::NEG_INFINITY, 10.0).is_nan());
    }

    #[test]
    fn invalid_length_gives_nan() {
        for length in [
            0.0,
            -0.0,
            -10.0,
            -SUB,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(
                wrap_ortho_half_strict(6.0, length).is_nan(),
                "length {length:?}"
            );
        }
        assert!(wrap_ortho_half_strict(f64::NAN, -1.0).is_nan());
    }
}
