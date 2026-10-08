#![cfg(feature = "capi")]
//! C ABI buffer contract. Feature `capi` is on by default.

use std::os::raw::c_int;

use minimage::{mi_cell, Cell};

extern "C" {
    fn mi_dist2(simbox: *const mi_cell, p: *const f64, q: *const f64, out: *mut f64) -> c_int;
    fn mi_displacement_euclidean(
        simbox: *const mi_cell,
        p: *const f64,
        q: *const f64,
        dr: *mut f64,
    ) -> c_int;
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
    fn mi_last_error() -> *const std::os::raw::c_char;
    fn mi_dist2_pairs(
        simbox: *const mi_cell,
        ps: *const f64,
        qs: *const f64,
        n: usize,
        out: *mut f64,
    ) -> c_int;
}

extern "C" {
    fn mi_dist2_euclidean_many(
        simbox: *const mi_cell,
        p: *const f64,
        qs: *const f64,
        n: usize,
        out: *mut f64,
    ) -> c_int;
    fn mi_dist2_euclidean_pairs(
        simbox: *const mi_cell,
        ps: *const f64,
        qs: *const f64,
        n: usize,
        out: *mut f64,
    ) -> c_int;
}

#[test]
fn euclidean_batches_abi_match_rust() {
    let raw = hex_raw();
    let cell = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [5.0, 8.660254037844386, 0.0],
        [0.0, 0.0, 10.0],
        [0.0; 3],
    )
    .unwrap();
    let ps: Vec<[f64; 3]> = (0..37)
        .map(|k| {
            [
                (k * 7 % 10) as f64,
                (k * 3 % 9) as f64,
                (k % 5) as f64 * 2.0,
            ]
        })
        .collect();
    let qs: Vec<[f64; 3]> = (0..37)
        .map(|k| {
            [
                (k * 13 % 17) as f64 - 4.0,
                (k * 5 % 11) as f64 * 1.5,
                (k % 7) as f64 - 9.0,
            ]
        })
        .collect();
    let mut out = vec![0.0; ps.len()];
    let rc = unsafe {
        mi_dist2_euclidean_pairs(
            &raw,
            ps.as_ptr().cast(),
            qs.as_ptr().cast(),
            ps.len(),
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    for k in 0..ps.len() {
        assert_eq!(
            out[k].to_bits(),
            cell.dist2_euclidean(ps[k], qs[k]).to_bits()
        );
    }
    let rc = unsafe {
        mi_dist2_euclidean_many(
            &raw,
            ps[3].as_ptr(),
            qs.as_ptr().cast(),
            qs.len(),
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    for k in 0..qs.len() {
        assert_eq!(
            out[k].to_bits(),
            cell.dist2_euclidean(ps[3], qs[k]).to_bits()
        );
    }
    assert_ne!(
        unsafe {
            mi_dist2_euclidean_pairs(
                &raw,
                std::ptr::null(),
                qs.as_ptr().cast(),
                1,
                out.as_mut_ptr(),
            )
        },
        0
    );
}

extern "C" {
    fn mi_fixed_many(simbox: *const mi_cell, rs: *const f64, n: usize, out: *mut u64) -> c_int;
    fn mi_dist2_fixed(simbox: *const mi_cell, a: *const u64, b: *const u64, out: *mut f64)
        -> c_int;
    fn mi_dist2_pairs_fixed(
        simbox: *const mi_cell,
        ps: *const u64,
        qs: *const u64,
        n: usize,
        out: *mut f64,
    ) -> c_int;
}

#[test]
fn fixed_abi_matches_rust() {
    let raw = hex_raw();
    let cell = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [5.0, 8.660254037844386, 0.0],
        [0.0, 0.0, 10.0],
        [0.0; 3],
    )
    .unwrap();
    let rs = [
        [0.5, 0.25, 9.75],
        [9.5, 8.0, 0.5],
        [-3.0, 14.0, 22.0],
        [4.0, 4.0, 4.0],
        [1.0, 2.0, 3.0],
    ];
    let mut fx = [[0u64; 3]; 5];
    let status = unsafe { mi_fixed_many(&raw, rs.as_ptr().cast(), 5, fx.as_mut_ptr().cast()) };
    assert_eq!(status, 0);
    for (r, f) in rs.iter().zip(&fx) {
        assert_eq!(*f, cell.fixed(*r));
    }
    let mut d2 = 0.0;
    let status = unsafe { mi_dist2_fixed(&raw, fx[0].as_ptr(), fx[1].as_ptr(), &mut d2) };
    assert_eq!(status, 0);
    assert_eq!(d2.to_bits(), cell.dist2_fixed(fx[0], fx[1]).to_bits());
    let ps = [fx[0], fx[1], fx[2], fx[3], fx[4]];
    let qs = [fx[4], fx[3], fx[2], fx[1], fx[0]];
    let mut out = [0.0; 5];
    let status = unsafe {
        mi_dist2_pairs_fixed(
            &raw,
            ps.as_ptr().cast(),
            qs.as_ptr().cast(),
            5,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(status, 0);
    for k in 0..5 {
        assert_eq!(out[k].to_bits(), cell.dist2_fixed(ps[k], qs[k]).to_bits());
        let float = cell.dist2(rs[k], rs[4 - k]);
        assert!((out[k] - float).abs() <= 1e-12 * (1.0 + float));
    }
}

fn hex_raw() -> mi_cell {
    mi_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 5.0,
        by: 8.660254037844386,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    }
}

#[test]
fn error_slot_clears_on_the_next_success_and_cells_alternate() {
    let hex = hex_raw();
    let mut ortho = hex_raw();
    ortho.bx = 0.0;
    let bad = mi_cell {
        cz: 0.0,
        ..hex_raw()
    };
    let p = [0.5, 0.25, 9.75];
    let q = [9.5, 8.0, 0.5];
    let mut got = 0.0;
    assert_eq!(
        unsafe { mi_dist2(&bad, p.as_ptr(), q.as_ptr(), &mut got) },
        1
    );
    assert!(!unsafe { mi_last_error() }.is_null());
    assert_eq!(
        unsafe { mi_dist2(&hex, std::ptr::null(), q.as_ptr(), &mut got) },
        1
    );
    assert!(!unsafe { mi_last_error() }.is_null());
    for _ in 0..3 {
        for (raw, cell) in [
            (
                hex,
                Cell::from_vectors(
                    [10.0, 0.0, 0.0],
                    [5.0, 8.660254037844386, 0.0],
                    [0.0, 0.0, 10.0],
                    [0.0; 3],
                ),
            ),
            (ortho, Cell::ortho(10.0, 8.660254037844386, 10.0)),
        ] {
            let cell = cell.unwrap();
            assert_eq!(
                unsafe { mi_dist2(&raw, p.as_ptr(), q.as_ptr(), &mut got) },
                0
            );
            assert!(unsafe { mi_last_error() }.is_null());
            assert_eq!(got.to_bits(), cell.dist2(p, q).to_bits());
            let ps = [p, q, p, q, p];
            let qs = [q, p, p, q, [20.0, -20.0, 35.0]];
            let mut out = [0.0; 5];
            let status = unsafe {
                mi_dist2_pairs(
                    &raw,
                    ps.as_ptr().cast(),
                    qs.as_ptr().cast(),
                    5,
                    out.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0);
            for k in 0..5 {
                assert_eq!(out[k].to_bits(), cell.dist2(ps[k], qs[k]).to_bits());
            }
        }
    }
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

#[test]
fn hex_body_diagonal_euclidean_abi_beats_fractional() {
    let rust = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [5.0, 8.660254037844386, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap();
    let raw = mi_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 5.0,
        by: 8.660254037844386,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let p = [0.0, 0.0, 0.0];
    let q = rust.cartesian([0.49, 0.49, 0.49]);
    let mut dr = [0.0; 3];
    let status =
        unsafe { mi_displacement_euclidean(&raw, p.as_ptr(), q.as_ptr(), dr.as_mut_ptr()) };
    assert_eq!(status, 0);
    let want = rust.displacement_euclidean(p, q);
    assert!((dr[0] - want[0]).abs() < 1e-15);
    assert!((dr[1] - want[1]).abs() < 1e-15);
    assert!((dr[2] - want[2]).abs() < 1e-15);
    let frac = rust.displacement(p, q);
    let e2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
    let f2 = frac[0] * frac[0] + frac[1] * frac[1] + frac[2] * frac[2];
    assert!(e2 + 1e-8 < f2);
}
