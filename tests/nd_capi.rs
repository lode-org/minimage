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
