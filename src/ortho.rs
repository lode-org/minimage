/// Wrap one orthorhombic component into the strict half-open interval
/// `[-length / 2, length / 2)`.
///
/// Exact positive half-box ties map to `-length / 2`; exact negative ties
/// remain `-length / 2`.
///
/// # Panics
///
/// Panics when `delta` is not finite or `length` is not finite and strictly
/// positive.
#[inline]
pub fn wrap_ortho_half_strict(delta: f64, length: f64) -> f64 {
    assert!(delta.is_finite(), "delta must be finite");
    assert!(
        length.is_finite() && length > 0.0,
        "length must be finite and positive"
    );
    delta - length * (delta / length + 0.5).floor()
}

#[cfg(test)]
mod tests {
    use super::wrap_ortho_half_strict;

    #[test]
    fn exact_half_box_ties_choose_negative_half() {
        assert_eq!(wrap_ortho_half_strict(5.0, 10.0), -5.0);
        assert_eq!(wrap_ortho_half_strict(15.0, 10.0), -5.0);
    }

    #[test]
    fn exact_negative_ties_keep_negative_half() {
        assert_eq!(wrap_ortho_half_strict(-5.0, 10.0), -5.0);
        assert_eq!(wrap_ortho_half_strict(-15.0, 10.0), -5.0);
    }

    #[test]
    fn non_ties_wrap_to_nearest_image() {
        assert_eq!(wrap_ortho_half_strict(4.0, 10.0), 4.0);
        assert_eq!(wrap_ortho_half_strict(6.0, 10.0), -4.0);
        assert_eq!(wrap_ortho_half_strict(-4.0, 10.0), -4.0);
        assert_eq!(wrap_ortho_half_strict(-6.0, 10.0), 4.0);
    }
}
