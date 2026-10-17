#![cfg(feature = "capi")]
//! Per-pair cost of the Euclidean nearest image, through Rust and the C ABI.
//!
//! `cargo test --release --test bench_euclidean -- --ignored --nocapture`

use std::hint::black_box;
use std::os::raw::c_int;
use std::time::Instant;

use minimage::{mi_cell, mi_pbc_cell, Cell, PartialCell};

extern "C" {
    fn mi_cell_from_ase(rows: *const f64, origin: *const f64, out: *mut mi_cell) -> c_int;
    fn mi_displacement_euclidean(
        simbox: *const mi_cell,
        p: *const f64,
        q: *const f64,
        dr: *mut f64,
    ) -> c_int;
    fn mi_pbc_cell_from_ase(
        rows: *const f64,
        pbc: *const c_int,
        origin: *const f64,
        out: *mut mi_pbc_cell,
    ) -> c_int;
    fn mi_displacement_euclidean_pbc(
        simbox: *const mi_pbc_cell,
        p: *const f64,
        q: *const f64,
        dr: *mut f64,
    ) -> c_int;
}

/// Deterministic points spread over a few cells.
fn points(n: usize, scale: f64) -> Vec<[f64; 3]> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            [
                scale * (2.0 * next() - 1.0),
                scale * (2.0 * next() - 1.0),
                scale * (2.0 * next() - 1.0),
            ]
        })
        .collect()
}

/// Best of seven passes over the pairs, in nanoseconds per pair.
fn time(name: &str, pts: &[[f64; 3]], f: &dyn Fn([f64; 3], [f64; 3]) -> [f64; 3]) {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        let mut acc = 0.0;
        for w in pts.chunks(2) {
            acc += f(black_box(w[0]), black_box(w[1]))[0];
        }
        black_box(acc);
        best = best.min(t.elapsed().as_secs_f64());
    }
    let ns = 1e9 * best / (pts.len() / 2) as f64;
    println!("bench path={name} ns_per_pair={ns:.3}");
}

#[test]
#[ignore]
fn bench_euclidean() {
    let rows = [8.0, 0.0, 0.0, 3.0, 7.0, 0.0, 1.0, -2.0, 9.0];
    let nested = [[8.0, 0.0, 0.0], [3.0, 7.0, 0.0], [1.0, -2.0, 9.0]];
    let cell = Cell::from_ase(nested).unwrap();
    let mut raw = std::mem::MaybeUninit::<mi_cell>::zeroed();
    assert_eq!(
        unsafe { mi_cell_from_ase(rows.as_ptr(), std::ptr::null(), raw.as_mut_ptr()) },
        0
    );
    let raw = unsafe { raw.assume_init() };
    let pts = points(1 << 18, 30.0);
    time("cell", &pts, &|p, q| cell.displacement_euclidean(p, q));
    time("c_cell", &pts, &|p, q| {
        let mut dr = [0.0; 3];
        unsafe { mi_displacement_euclidean(&raw, p.as_ptr(), q.as_ptr(), dr.as_mut_ptr()) };
        dr
    });
    let all = PartialCell::from_ase(nested, [true; 3]).unwrap();
    let slab = PartialCell::from_ase(nested, [true, true, false]).unwrap();
    let wire = PartialCell::from_ase(nested, [true, false, false]).unwrap();
    time("partial_all", &pts, &|p, q| {
        all.displacement_euclidean(p, q)
    });
    time("partial_slab", &pts, &|p, q| {
        slab.displacement_euclidean(p, q)
    });
    time("partial_wire", &pts, &|p, q| {
        wire.displacement_euclidean(p, q)
    });
    let mut slab_raw = std::mem::MaybeUninit::<mi_pbc_cell>::zeroed();
    let mask: [c_int; 3] = [1, 1, 0];
    assert_eq!(
        unsafe {
            mi_pbc_cell_from_ase(
                rows.as_ptr(),
                mask.as_ptr(),
                std::ptr::null(),
                slab_raw.as_mut_ptr(),
            )
        },
        0
    );
    let slab_raw = unsafe { slab_raw.assume_init() };
    time("c_slab", &pts, &|p, q| {
        let mut dr = [0.0; 3];
        unsafe {
            mi_displacement_euclidean_pbc(&slab_raw, p.as_ptr(), q.as_ptr(), dr.as_mut_ptr())
        };
        dr
    });
}
