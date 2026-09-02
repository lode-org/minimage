#![cfg(feature = "capi")]
//! C ABI buffer contract. Feature `capi` is on by default.

use std::os::raw::c_int;

use minimage::{mi_cell, Cell};

extern "C" {
    fn mi_dist2(simbox: *const mi_cell, p: *const f64, q: *const f64, out: *mut f64) -> c_int;
    fn mi_reduce_pairs(pairs: *const c_int, n: usize, out: *mut c_int, out_n: *mut usize) -> c_int;
    fn mi_cell_from_lammps_bounds(
        xspan: f64,
        yspan: f64,
        zspan: f64,
        xy: f64,
        xz: f64,
        yz: f64,
        xlo_b: f64,
        ylo_b: f64,
        zlo_b: f64,
        out: *mut mi_cell,
    ) -> c_int;
    fn mi_version() -> *const std::os::raw::c_char;
}

#[test]
fn ortho_dist2_matches_rust() {
    let rust = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let raw = mi_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 10.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let p = [0.2, 0.0, 0.0];
    let q = [9.4, 0.0, 0.0];
    let mut got = -1.0;
    let status = unsafe { mi_dist2(&raw, p.as_ptr(), q.as_ptr(), &mut got) };
    assert_eq!(status, 0);
    assert!((got - rust.dist2(p, q)).abs() < 1e-15);
    assert!((got - 0.64).abs() < 1e-12);
}

#[test]
fn lammps_bounds_abi() {
    let mut raw = mi_cell {
        ax: 0.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 0.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 0.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let status = unsafe {
        mi_cell_from_lammps_bounds(
            15.0,
            8.660254037844386,
            10.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            &mut raw,
        )
    };
    assert_eq!(status, 0);
    assert!((raw.ax - 10.0).abs() < 1e-12);
    assert!((raw.bx - 5.0).abs() < 1e-12);
}

#[test]
fn reduce_pairs_abi() {
    let pairs: [c_int; 8] = [0, 1, 0, 0, 0, 1, 2, 3];
    let mut out = [0; 8];
    let mut n = 0usize;
    let status = unsafe { mi_reduce_pairs(pairs.as_ptr(), 4, out.as_mut_ptr(), &mut n) };
    assert_eq!(status, 0);
    assert_eq!(n, 2);
    assert_eq!(&out[..4], &[0, 1, 2, 3]);
}

#[test]
fn version_is_nonempty() {
    let p = unsafe { mi_version() };
    assert!(!p.is_null());
}
