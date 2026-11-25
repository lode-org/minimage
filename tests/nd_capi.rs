#![cfg(feature = "capi")]

use std::ffi::CStr;

use minimage::{
    minimage_last_error, minimage_nd_aabb_min_dist2, minimage_nd_axis_min_dist2, minimage_nd_dist2,
    minimage_nd_dist2_many, minimage_nd_distance, minimage_nd_wrap_difference,
    minimage_nd_wrap_point,
};

#[test]
fn nd_capi_scalar_batch_and_box_paths() {
    let periods = [360.0, 0.0, 10.0];
    let point = [-10.0, 2.0, 10.0];
    let mut wrapped = [0.0; 3];
    assert_eq!(
        unsafe {
            minimage_nd_wrap_point(periods.as_ptr(), 3, point.as_ptr(), wrapped.as_mut_ptr())
        },
        0
    );
    assert_eq!(wrapped, [350.0, 2.0, 0.0]);

    let boundary = [-1e-20, -7.0, 20.0];
    assert_eq!(
        unsafe {
            minimage_nd_wrap_point(periods.as_ptr(), 3, boundary.as_ptr(), wrapped.as_mut_ptr())
        },
        0
    );
    assert_eq!(wrapped, [0.0, -7.0, 0.0]);

    let difference = [180.0, -4.0, 5.0];
    assert_eq!(
        unsafe {
            minimage_nd_wrap_difference(
                periods.as_ptr(),
                3,
                difference.as_ptr(),
                wrapped.as_mut_ptr(),
            )
        },
        0
    );
    assert_eq!(wrapped, [-180.0, -4.0, -5.0]);

    let p = [350.0, 1.0, 9.0];
    let q = [10.0, 4.0, 1.0];
    let mut out = 0.0;
    assert_eq!(
        unsafe { minimage_nd_dist2(periods.as_ptr(), 3, p.as_ptr(), q.as_ptr(), &mut out,) },
        0
    );
    assert_eq!(out, 413.0);
    assert_eq!(
        unsafe { minimage_nd_distance(periods.as_ptr(), 3, p.as_ptr(), q.as_ptr(), &mut out,) },
        0
    );
    assert_eq!(out, 413.0_f64.sqrt());

    let points = [10.0, 4.0, 1.0, 180.0, 1.0, 9.0];
    let mut distances = [0.0; 2];
    assert_eq!(
        unsafe {
            minimage_nd_dist2_many(
                periods.as_ptr(),
                3,
                p.as_ptr(),
                points.as_ptr(),
                2,
                distances.as_mut_ptr(),
            )
        },
        0
    );
    assert_eq!(distances, [413.0, 28_900.0]);

    assert_eq!(
        unsafe { minimage_nd_axis_min_dist2(periods.as_ptr(), 3, 0, 350.0, 0.0, 10.0, &mut out) },
        0
    );
    assert_eq!(out, 100.0);

    let lower = [0.0, 2.0, 0.0];
    let upper = [10.0, 3.0, 2.0];
    assert_eq!(
        unsafe {
            minimage_nd_aabb_min_dist2(
                periods.as_ptr(),
                3,
                p.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                &mut out,
            )
        },
        0
    );
    assert_eq!(out, 102.0);
}

#[test]
fn nd_capi_rejects_zero_dimensions_and_bad_periods() {
    let mut out = 0.0;
    assert_eq!(
        unsafe { minimage_nd_dist2([].as_ptr(), 0, [].as_ptr(), [].as_ptr(), &mut out,) },
        1
    );
    let error = minimage_last_error();
    assert!(!error.is_null());
    // SAFETY: `minimage_last_error` returns a valid thread-local C string
    // until the next C API call on this thread.
    assert_eq!(unsafe { CStr::from_ptr(error) }.to_bytes(), b"empty input");

    let periods = [f64::NAN];
    let point = [0.0];
    assert_eq!(
        unsafe {
            minimage_nd_dist2(
                periods.as_ptr(),
                1,
                point.as_ptr(),
                point.as_ptr(),
                &mut out,
            )
        },
        1
    );
    assert!(!minimage_last_error().is_null());
}

#[test]
fn nd_capi_extreme_and_negative_coordinates_stay_finite() {
    let periods = [10.0, 0.0, 7.0];
    let p = [-f64::MAX, -4.0, f64::MAX];
    let q = [f64::MAX, -1.0, -f64::MAX];
    let mut dist2 = 0.0;
    assert_eq!(
        unsafe { minimage_nd_dist2(periods.as_ptr(), 3, p.as_ptr(), q.as_ptr(), &mut dist2) },
        0
    );
    assert!(dist2.is_finite() && dist2 >= 9.0);

    let mut distance = 0.0;
    assert_eq!(
        unsafe { minimage_nd_distance(periods.as_ptr(), 3, p.as_ptr(), q.as_ptr(), &mut distance) },
        0
    );
    assert_eq!(distance, dist2.sqrt());

    let points = [q[0], q[1], q[2], p[0], p[1], p[2]];
    let mut many = [0.0; 2];
    assert_eq!(
        unsafe {
            minimage_nd_dist2_many(
                periods.as_ptr(),
                3,
                p.as_ptr(),
                points.as_ptr(),
                2,
                many.as_mut_ptr(),
            )
        },
        0
    );
    assert_eq!(many, [dist2, 0.0]);

    let mut axis_bound = 0.0;
    assert_eq!(
        unsafe {
            minimage_nd_axis_min_dist2(periods.as_ptr(), 3, 0, f64::MAX, -1.0, 1.0, &mut axis_bound)
        },
        0
    );
    assert_eq!(axis_bound, 1.0);

    let lower = [-1.0, -2.0, -1.0];
    let upper = [1.0, 0.0, 1.0];
    let mut box_bound = 0.0;
    assert_eq!(
        unsafe {
            minimage_nd_aabb_min_dist2(
                periods.as_ptr(),
                3,
                p.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                &mut box_bound,
            )
        },
        0
    );
    assert!(box_bound.is_finite() && box_bound >= 4.0);
}

#[test]
fn nd_capi_huge_image_counts_keep_a_representable_separation() {
    let periods = [9.664_251_935_223_536e-9];
    let (a, b) = (35_602_323.555_269_06, 35_602_323.555_269_07);
    let expected = -2.213_671_338_299_708_4e-9_f64;
    let mut out = 0.0;
    assert_eq!(
        unsafe { minimage_nd_dist2(periods.as_ptr(), 1, &a, &b, &mut out) },
        0
    );
    assert_eq!(out, expected * expected);
    assert_eq!(
        unsafe { minimage_nd_axis_min_dist2(periods.as_ptr(), 1, 0, b, a, a, &mut out) },
        0
    );
    assert_eq!(out, expected * expected);
    assert_eq!(
        unsafe { minimage_nd_aabb_min_dist2(periods.as_ptr(), 1, &b, &a, &a, &mut out) },
        0
    );
    assert_eq!(out, expected * expected);
}

#[test]
fn nd_capi_distances_use_the_exact_minimum_image() {
    let micro = [1e-6];
    let far = 4.525_188_817_411_374e-16_f64;
    let (from, to) = (-10.0, 0.0);
    let mut out = 0.0;
    assert_eq!(
        unsafe { minimage_nd_dist2(micro.as_ptr(), 1, &from, &to, &mut out) },
        0
    );
    assert_eq!(out, far * far);
    assert_eq!(
        unsafe { minimage_nd_axis_min_dist2(micro.as_ptr(), 1, 0, to, from, from, &mut out) },
        0
    );
    assert_eq!(out, far * far);

    let ten = [10.0];
    let two_54 = 18_014_398_509_481_984.0;
    let edge = two_54 + 4.0;
    assert_eq!(
        unsafe { minimage_nd_dist2(ten.as_ptr(), 1, &two_54, &(two_54 - 4.0), &mut out) },
        0
    );
    assert_eq!(out, 16.0);
    assert_eq!(
        unsafe { minimage_nd_aabb_min_dist2(ten.as_ptr(), 1, &two_54, &edge, &edge, &mut out) },
        0
    );
    assert_eq!(out, 16.0);
}
