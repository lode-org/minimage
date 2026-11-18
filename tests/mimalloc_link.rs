#![cfg(all(feature = "capi", not(feature = "capi-compat-names")))]

use std::ffi::{c_char, CStr};

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

extern "C" {
    fn minimage_version() -> *const c_char;
}

#[test]
fn capi_links_and_runs_with_mimalloc() {
    let values = Vec::from([1_u64, 2, 3, 4]);
    assert_eq!(values.iter().sum::<u64>(), 10);

    let cell = minimage::Cell::ortho(10.0, 10.0, 10.0).expect("valid cell");
    let distance = cell.dist2([0.2, 0.0, 0.0], [9.4, 0.0, 0.0]);
    assert!((distance - 0.64).abs() < f64::EPSILON * 16.0);

    let version = unsafe { CStr::from_ptr(minimage_version()) };
    assert!(!version.to_bytes().is_empty());
}
