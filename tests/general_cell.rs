use minimage::{pairs_within_images, Cell, CellConfig, Error, ImagePair, PartialCell};

fn norm2(v: [f64; 3]) -> f64 {
    v[0].mul_add(v[0], v[1].mul_add(v[1], v[2] * v[2]))
}

fn add_image(mut d: [f64; 3], rows: [[f64; 3]; 3], shift: [i32; 3]) -> [f64; 3] {
    for axis in 0..3 {
        for (cart, value) in d.iter_mut().enumerate() {
            *value += f64::from(shift[axis]) * rows[axis][cart];
        }
    }
    d
}

fn brute_full(rows: [[f64; 3]; 3], p: [f64; 3], q: [f64; 3]) -> f64 {
    let raw = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let mut best = f64::INFINITY;
    for i in -5..=5 {
        for j in -5..=5 {
            for k in -5..=5 {
                best = best.min(norm2(add_image(raw, rows, [i, j, k])));
            }
        }
    }
    best
}

fn brute_partial(rows: [[f64; 3]; 3], periodic: [bool; 3], p: [f64; 3], q: [f64; 3]) -> f64 {
    let raw = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let range = |axis: usize| if periodic[axis] { -8..=8 } else { 0..=0 };
    let mut best = f64::INFINITY;
    for i in range(0) {
        for j in range(1) {
            for k in range(2) {
                best = best.min(norm2(add_image(raw, rows, [i, j, k])));
            }
        }
    }
    best
}

fn cart(rows: [[f64; 3]; 3], s: [f64; 3]) -> [f64; 3] {
    [
        s[0] * rows[0][0] + s[1] * rows[1][0] + s[2] * rows[2][0],
        s[0] * rows[0][1] + s[1] * rows[1][1] + s[2] * rows[2][1],
        s[0] * rows[0][2] + s[1] * rows[1][2] + s[2] * rows[2][2],
    ]
}

fn check_full(rows: [[f64; 3]; 3]) {
    let cell = Cell::from_ase(rows).unwrap();
    let samples = [
        ([0.03, 0.97, -0.42], [1.91, -0.88, 0.63]),
        ([-1.2, 0.49, 2.1], [0.45, 1.52, -0.77]),
        ([0.0, 0.0, 0.0], [0.49, 0.49, 0.49]),
    ];
    for (sp, sq) in samples {
        let p = cart(rows, sp);
        let q = cart(rows, sq);
        let got = cell.dist2_euclidean(p, q);
        let want = brute_full(rows, p, q);
        assert!(
            (got - want).abs() <= 2.0e-11 * (1.0 + want),
            "{rows:?}: {got} != {want}"
        );
    }
}

#[test]
fn reported_skewed_cell_keeps_the_pair_across_b() {
    let rows = [[20.0, 0.0, 0.0], [6.0, -20.0, 0.0], [0.0, 4.0, -20.0]];
    let cell = Cell::from_ase(rows).unwrap();
    let xyz = [cart(rows, [0.25, 0.02, 0.4]), cart(rows, [0.25, 0.98, 0.4])];
    let mut pairs = Vec::<ImagePair>::new();
    pairs_within_images(&cell, [true; 3], &xyz, 1.0, true, &mut pairs).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!((pairs[0].i, pairs[0].j), (0, 1));
    assert_eq!(pairs[0].shift, [0, -1, 0]);
    assert!((pairs[0].dist2 - brute_full(rows, xyz[0], xyz[1])).abs() < 1.0e-12);
}

#[test]
fn rotated_fcc_bcc_and_left_handed_cells_match_image_sums() {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let rotate = |v: [f64; 3]| [inv_sqrt2 * (v[0] - v[1]), inv_sqrt2 * (v[0] + v[1]), v[2]];
    let base = [[8.0, 0.0, 0.0], [2.0, 7.0, 0.0], [-1.0, 1.5, 9.0]];
    check_full(base.map(rotate));
    check_full([[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]);
    check_full([[-3.0, 3.0, 3.0], [3.0, -3.0, 3.0], [3.0, 3.0, -3.0]]);
    check_full([base[1], base[0], base[2]]);
}

#[test]
fn mixed_periodic_skewed_cells_match_image_sums() {
    let rows = [[7.0, 1.0, 2.0], [-2.0, 8.0, 1.5], [1.0, -3.0, 9.0]];
    let p = [11.0, -7.0, 4.0];
    let q = [-9.0, 13.0, 20.0];
    for periodic in [
        [true, false, false],
        [false, true, false],
        [true, false, true],
        [false, true, true],
    ] {
        let cell = PartialCell::from_ase(rows, periodic).unwrap();
        assert_eq!(cell.periodic(), periodic);
        assert_eq!(cell.rows(), rows);
        let got = cell.dist2_euclidean(p, q);
        let want = brute_partial(rows, periodic, p, q);
        assert!((got - want).abs() <= 1.0e-10 * (1.0 + want));
    }
}

#[test]
fn scaled_inverse_is_scale_invariant_and_guards_condition() {
    let tiny = Cell::from_ase([
        [2.0e-100, 0.0, 0.0],
        [0.4e-100, 3.0e-100, 0.0],
        [0.2e-100, -0.3e-100, 4.0e-100],
    ])
    .unwrap();
    for axis in 0..3 {
        assert!(tiny.widths()[axis] > 0.0);
        for row in 0..3 {
            let identity = tiny.h()[0][row] * tiny.hinv()[axis][0]
                + tiny.h()[1][row] * tiny.hinv()[axis][1]
                + tiny.h()[2][row] * tiny.hinv()[axis][2];
            assert!((identity - f64::from(axis == row)).abs() < 2.0e-14);
        }
    }

    let near = [[1.0, 0.0, 0.0], [1.0, 1.0e-14, 0.0], [0.0, 0.0, 1.0]];
    assert!(matches!(
        Cell::from_ase(near),
        Err(Error::IllConditioned { .. })
    ));
    let permissive = CellConfig::default().with_max_condition(1.0e16);
    assert!(Cell::from_ase_with_config(near, permissive).is_ok());
}

#[test]
fn left_handed_fractional_and_fixed_coordinates_keep_orientation() {
    let cell = Cell::from_ase([[-8.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 9.0]]).unwrap();
    let r = [-2.0, 3.5, 2.25];
    let s = cell.fractional(r);
    assert!((s[0] - 0.25).abs() < 1.0e-15);
    assert!((s[1] - 0.5).abs() < 1.0e-15);
    assert!((s[2] - 0.25).abs() < 1.0e-15);
    assert_eq!(cell.cartesian(s), r);
    let a = cell.fixed(r);
    let b = cell.fixed([-6.0, 3.5, 2.25]);
    let d = cell.displacement_fixed(a, b);
    assert!((d[0] + 4.0).abs() < 1.0e-12);
}

#[cfg(feature = "capi")]
#[test]
fn c_geometry_and_explicit_periodicity_match_rust() {
    let rows = [20.0, 0.0, 0.0, 6.0, -20.0, 0.0, 0.0, 4.0, -20.0];
    let nested = [[20.0, 0.0, 0.0], [6.0, -20.0, 0.0], [0.0, 4.0, -20.0]];
    let rust = Cell::from_ase(nested).unwrap();
    let mut raw = std::mem::MaybeUninit::<minimage::mi_cell>::zeroed();
    assert_eq!(
        unsafe {
            minimage::minimage_cell_from_ase(rows.as_ptr(), std::ptr::null(), raw.as_mut_ptr())
        },
        0
    );
    let raw = unsafe { raw.assume_init() };
    let mut hinv = [0.0; 9];
    let mut widths = [0.0; 3];
    assert_eq!(
        unsafe { minimage::minimage_cell_hinv(&raw, hinv.as_mut_ptr()) },
        0
    );
    assert_eq!(
        unsafe { minimage::minimage_cell_widths(&raw, widths.as_mut_ptr()) },
        0
    );
    let expected_hinv: [f64; 9] = rust.hinv().concat().try_into().unwrap();
    assert_eq!(hinv, expected_hinv);
    assert_eq!(widths, rust.widths());
    let position = cart(nested, [1.25, -0.2, 0.4]);
    let mut fractional = [0.0; 3];
    assert_eq!(
        unsafe { minimage::minimage_fractional(&raw, position.as_ptr(), fractional.as_mut_ptr(),) },
        0
    );
    assert_eq!(
        fractional.map(f64::to_bits),
        rust.fractional(position).map(f64::to_bits)
    );

    let pbc = [1, 1, 1];
    let mut partial = std::mem::MaybeUninit::<minimage::mi_pbc_cell>::zeroed();
    assert_eq!(
        unsafe {
            minimage::minimage_pbc_cell_from_ase(
                rows.as_ptr(),
                pbc.as_ptr(),
                std::ptr::null(),
                partial.as_mut_ptr(),
            )
        },
        0
    );
    let partial = unsafe { partial.assume_init() };
    assert_eq!([partial.pbc_a, partial.pbc_b, partial.pbc_c], pbc);
    let p = cart(nested, [0.25, 0.02, 0.4]);
    let q = cart(nested, [0.25, 0.98, 0.4]);
    let mut d2 = 0.0;
    assert_eq!(
        unsafe {
            minimage::minimage_dist2_euclidean_pbc(&partial, p.as_ptr(), q.as_ptr(), &mut d2)
        },
        0
    );
    assert!((d2 - brute_full(nested, p, q)).abs() < 1.0e-12);
}
