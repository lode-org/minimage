//! A lattice does not depend on the sign or the handedness of the basis
//! that names it: the edge `-L` spans the points the edge `+L` spans.
//!
//! Every check here compares a cell against a brute-force scan over the
//! images of the lattice the caller passed, so a constructor may store a
//! different basis of that lattice but never a different lattice. The
//! cells are diagonal with a negative edge on one, two, and three axes,
//! lower-triangular with negative diagonal entries, left-handed, and
//! rotated, with full and with mixed periodicity.

use minimage::{
    dist2_euclidean_pairs, dist2_pairs, filter_shifted, pairs_within_images, wrap_many, Cell,
    Error, FilterOut, PartialCell,
};

type Rows = [[f64; 3]; 3];

const ALL: [bool; 3] = [true, true, true];
const ORIGIN: [f64; 3] = [0.3, -1.1, 2.0];
/// Images scanned along each periodic axis, each way.
const REACH: i32 = 4;

fn norm2(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn combination(rows: &Rows, n: [i32; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (row, count) in rows.iter().zip(n) {
        for k in 0..3 {
            out[k] += f64::from(count) * row[k];
        }
    }
    out
}

/// Squared length of the shortest `d + n . rows` over the periodic rows.
fn brute_min2(rows: &Rows, periodic: [bool; 3], d: [f64; 3]) -> f64 {
    let reach = |axis: usize| if periodic[axis] { REACH } else { 0 };
    let mut best = f64::INFINITY;
    for na in -reach(0)..=reach(0) {
        for nb in -reach(1)..=reach(1) {
            for nc in -reach(2)..=reach(2) {
                let shift = combination(rows, [na, nb, nc]);
                best = best.min(norm2([d[0] + shift[0], d[1] + shift[1], d[2] + shift[2]]));
            }
        }
    }
    best
}

/// True when `v - w` is an integer combination of the periodic rows.
fn same_modulo(rows: &Rows, periodic: [bool; 3], v: [f64; 3], w: [f64; 3]) -> bool {
    brute_min2(rows, periodic, sub(v, w)) < 1e-18
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

fn close3(a: [f64; 3], b: [f64; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// `n` points spread over `[-scale, scale]` on each axis.
fn points(n: usize, seed: u64, scale: f64) -> Vec<[f64; 3]> {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0) * scale
    };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

/// Face altitudes of the parallelepiped the rows span.
fn altitudes(rows: &Rows) -> [f64; 3] {
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let [a, b, c] = *rows;
    let bc = cross(b, c);
    let volume = (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]).abs();
    [
        volume / norm2(bc).sqrt(),
        volume / norm2(cross(c, a)).sqrt(),
        volume / norm2(cross(a, b)).sqrt(),
    ]
}

fn negate(rows: Rows, mask: usize) -> Rows {
    let mut out = rows;
    for (axis, row) in out.iter_mut().enumerate() {
        if mask & (1 << axis) != 0 {
            for value in row.iter_mut() {
                *value = -*value;
            }
        }
    }
    out
}

fn diagonal(edges: [f64; 3]) -> Rows {
    [
        [edges[0], 0.0, 0.0],
        [0.0, edges[1], 0.0],
        [0.0, 0.0, edges[2]],
    ]
}

/// Lower-triangular rows: `a` along x and `b` in the xy plane.
const TRIANGULAR: Rows = [[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]];

/// `TRIANGULAR` turned about an axis off every coordinate axis.
fn rotated() -> Rows {
    let (s, c) = 0.7_f64.sin_cos();
    let axis = [1.0 / 3.0_f64.sqrt(); 3];
    let turn = |v: [f64; 3]| {
        let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
        let cross = [
            axis[1] * v[2] - axis[2] * v[1],
            axis[2] * v[0] - axis[0] * v[2],
            axis[0] * v[1] - axis[1] * v[0],
        ];
        [
            v[0] * c + cross[0] * s + axis[0] * dot * (1.0 - c),
            v[1] * c + cross[1] * s + axis[1] * dot * (1.0 - c),
            v[2] * c + cross[2] * s + axis[2] * dot * (1.0 - c),
        ]
    };
    [
        turn(TRIANGULAR[0]),
        turn(TRIANGULAR[1]),
        turn(TRIANGULAR[2]),
    ]
}

/// Every stored quantity of the cell describes the lattice of `rows`.
fn check_cell(rows: Rows) {
    let cell = Cell::from_vectors(rows[0], rows[1], rows[2], ORIGIN)
        .unwrap_or_else(|err| panic!("{rows:?} refused: {err}"));
    let stored: Rows = [cell.a(), cell.b(), cell.c()];

    // The stored basis and the caller's basis span one lattice.
    for axis in 0..3 {
        assert!(
            same_modulo(&rows, ALL, stored[axis], [0.0; 3]),
            "{rows:?}: stored vector {axis} is not a lattice vector"
        );
        assert!(
            same_modulo(&stored, ALL, rows[axis], [0.0; 3]),
            "{rows:?}: input vector {axis} is not in the stored lattice"
        );
    }
    // The lattice shift, the widths, and H agree with the stored basis.
    for (axis, unit) in [[1, 0, 0], [0, 1, 0], [0, 0, 1]].into_iter().enumerate() {
        assert!(
            close3(cell.lattice_shift(unit[0], unit[1], unit[2]), stored[axis]),
            "{rows:?}: lattice_shift along {axis} is not the stored vector"
        );
    }
    assert_eq!(cell.h(), stored);
    let want = altitudes(&rows);
    let have = cell.widths();
    for axis in 0..3 {
        assert!(
            have[axis] > 0.0 && close(have[axis], want[axis]),
            "{rows:?}"
        );
    }

    let scale = 1.2 * want.iter().fold(0.0_f64, |m, w| m.max(*w));
    let ps = points(48, 0x9e3779b97f4a7c15, scale);
    let qs = points(48, 0xd1b54a32d192ed03, scale);

    // A point wrapped into the cell is an image of that point.
    for &p in &ps {
        let s = cell.fractional(p);
        assert!(s.iter().all(|f| (0.0..1.0).contains(f)), "{rows:?}: {s:?}");
        let wrapped = cell.cartesian(s);
        assert!(
            same_modulo(&rows, ALL, wrapped, p),
            "{rows:?}: {p:?} wrapped to {wrapped:?}, which is not one of its images"
        );
    }

    let mut diffs = Vec::new();
    for (&p, &q) in ps.iter().zip(&qs) {
        let d = sub(q, p);
        diffs.push(d);
        let nearest = brute_min2(&rows, ALL, d);
        let engine = cell.displacement(p, q);
        assert!(
            same_modulo(&rows, ALL, engine, d),
            "{rows:?}: displacement {engine:?} is not an image of {d:?}"
        );
        assert!(close(cell.dist2(p, q), norm2(engine)), "{rows:?}");
        if cell.is_ortho() {
            assert!(close(norm2(engine), nearest), "{rows:?}: {engine:?}");
        }
        let euclid = cell.displacement_euclidean(p, q);
        assert!(same_modulo(&rows, ALL, euclid, d), "{rows:?}");
        assert!(
            close(norm2(euclid), nearest),
            "{rows:?}: nearest image {} but brute force {nearest}",
            norm2(euclid)
        );
        assert!(close(cell.dist2_euclidean(p, q), nearest), "{rows:?}");
        let fixed = cell.displacement_fixed(cell.fixed(p), cell.fixed(q));
        assert!(
            close3(fixed, engine),
            "{rows:?}: fixed-point {fixed:?} but {engine:?}"
        );
        let fixed32 = cell.displacement_fixed32(cell.fixed32(p), cell.fixed32(q));
        assert!(
            brute_min2(&rows, ALL, sub(fixed32, d)) < 1e-12 * scale * scale,
            "{rows:?}: 32-bit fixed-point {fixed32:?} is not an image of {d:?}"
        );
    }

    // The batches are the per-pair calls.
    let mut wrapped = vec![[0.0; 3]; diffs.len()];
    wrap_many(&cell, &diffs, &mut wrapped).unwrap();
    let mut engine2 = vec![0.0; ps.len()];
    dist2_pairs(&cell, &ps, &qs, &mut engine2).unwrap();
    let mut euclid2 = vec![0.0; ps.len()];
    dist2_euclidean_pairs(&cell, &ps, &qs, &mut euclid2).unwrap();
    for k in 0..ps.len() {
        assert!(
            close3(wrapped[k], cell.displacement(ps[k], qs[k])),
            "{rows:?}"
        );
        assert!(close(engine2[k], cell.dist2(ps[k], qs[k])), "{rows:?}");
        assert!(
            close(euclid2[k], brute_min2(&rows, ALL, diffs[k])),
            "{rows:?}"
        );
    }

    // A caller's integer shift counts the stored vectors.
    let shifts = [[1, 0, 0], [0, -1, 0], [0, 0, 1], [1, 1, -1], [-2, 3, 5]];
    let ii = vec![0_u32; shifts.len()];
    let jj = vec![1_u32; shifts.len()];
    let mut kept = FilterOut::default();
    filter_shifted(
        &cell,
        &ps[..2],
        &ii,
        &jj,
        &shifts,
        f64::MAX,
        true,
        &mut kept,
    )
    .unwrap();
    assert_eq!(kept.vector.len(), shifts.len());
    for (shift, vector) in shifts.iter().zip(&kept.vector) {
        let lattice = combination(&stored, *shift);
        let want = [
            ps[1][0] + lattice[0] - ps[0][0],
            ps[1][1] + lattice[1] - ps[0][1],
            ps[1][2] + lattice[2] - ps[0][2],
        ];
        assert!(close3(*vector, want), "{rows:?}: shift {shift:?}");
    }

    // A pair search finds every image inside the cutoff.
    let cutoff = 0.8 * want.iter().fold(f64::INFINITY, |m, w| m.min(*w));
    let sample = &ps[..10];
    let mut found = Vec::new();
    let count = pairs_within_images(&cell, ALL, sample, cutoff, false, &mut found).unwrap();
    let mut brute = 0;
    for (i, &p) in sample.iter().enumerate() {
        for (j, &q) in sample.iter().enumerate() {
            for na in -REACH..=REACH {
                for nb in -REACH..=REACH {
                    for nc in -REACH..=REACH {
                        if i == j && na == 0 && nb == 0 && nc == 0 {
                            continue;
                        }
                        let shift = combination(&rows, [na, nb, nc]);
                        let d = [
                            p[0] - q[0] - shift[0],
                            p[1] - q[1] - shift[1],
                            p[2] - q[2] - shift[2],
                        ];
                        if norm2(d) < cutoff * cutoff {
                            brute += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(count, brute, "{rows:?}: pairs within {cutoff}");
    for pair in &found {
        let lattice = combination(&stored, pair.shift);
        let (p, q) = (sample[pair.i as usize], sample[pair.j as usize]);
        let want = [
            p[0] - q[0] - lattice[0],
            p[1] - q[1] - lattice[1],
            p[2] - q[2] - lattice[2],
        ];
        assert!(close3(pair.d, want), "{rows:?}: shift {:?}", pair.shift);
    }
}

/// The nearest image over the periodic rows alone.
fn check_partial(rows: Rows, periodic: [bool; 3]) {
    let cell = PartialCell::from_ase(rows, periodic)
        .unwrap_or_else(|err| panic!("{rows:?} {periodic:?} refused: {err}"));
    let scale = 1.2
        * rows
            .iter()
            .fold(0.0_f64, |m, row| m.max(norm2(*row).sqrt()));
    let ps = points(32, 0x2545f4914f6cdd1d, scale);
    let qs = points(32, 0x94d049bb133111eb, scale);
    for (&p, &q) in ps.iter().zip(&qs) {
        let d = sub(q, p);
        let nearest = brute_min2(&rows, periodic, d);
        let have = cell.displacement_euclidean(p, q);
        assert!(
            same_modulo(&rows, periodic, have, d),
            "{rows:?} {periodic:?}: {have:?} is not an image of {d:?}"
        );
        assert!(
            close(norm2(have), nearest),
            "{rows:?} {periodic:?}: nearest image {} but brute force {nearest}",
            norm2(have)
        );
        assert!(close(cell.dist2_euclidean(p, q), nearest));
    }
}

const MASKS: [[bool; 3]; 7] = [
    [true, false, false],
    [false, true, false],
    [false, false, true],
    [true, true, false],
    [true, false, true],
    [false, true, true],
    [false, false, false],
];

#[test]
fn diagonal_cell_with_negative_edges_is_the_positive_lattice() {
    for mask in 0..8 {
        check_cell(negate(diagonal([15.0, 12.0, 9.0]), mask));
    }
    // The reported case: two negative edges on a cube.
    check_cell(diagonal([-15.0, -15.0, 15.0]));
}

#[test]
fn diagonal_cell_stores_positive_edges() {
    for mask in 0..8 {
        let rows = negate(diagonal([15.0, 12.0, 9.0]), mask);
        let cell = Cell::ortho(rows[0][0], rows[1][1], rows[2][2]).unwrap();
        assert!(cell.is_ortho());
        assert_eq!(cell.a(), [15.0, 0.0, 0.0]);
        assert_eq!(cell.b(), [0.0, 12.0, 0.0]);
        assert_eq!(cell.c(), [0.0, 0.0, 9.0]);
        assert_eq!(cell.widths(), [15.0, 12.0, 9.0]);
        assert_eq!(cell, Cell::ortho(15.0, 12.0, 9.0).unwrap());
    }
}

#[test]
fn nearly_diagonal_cell_with_negative_edges_is_the_positive_lattice() {
    // Tilts far below the orthorhombic tolerance, on the triangular and
    // on the general construction path.
    check_cell([[-10.0, 0.0, 0.0], [1e-14, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    check_cell([[-10.0, 0.0, 0.0], [1e-14, -10.0, 0.0], [0.0, 1e-14, -10.0]]);
    check_cell([[-10.0, 1e-14, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 10.0]]);
    check_cell([[10.0, 0.0, 1e-14], [0.0, -10.0, 0.0], [1e-14, 0.0, -10.0]]);
}

#[test]
fn triangular_cell_with_negative_diagonal_is_the_positive_lattice() {
    for mask in 0..8 {
        check_cell(negate(TRIANGULAR, mask));
    }
}

#[test]
fn left_handed_and_rotated_cells_are_the_same_lattice() {
    let [a, b, c] = TRIANGULAR;
    check_cell([b, a, c]);
    check_cell([a, c, b]);
    check_cell([c, b, a]);
    for mask in 0..8 {
        check_cell(negate(rotated(), mask));
    }
    let [a, b, c] = rotated();
    check_cell([b, a, c]);
}

#[test]
fn mixed_periodicity_ignores_the_sign_of_an_edge() {
    for mask in 0..8 {
        for periodic in MASKS {
            check_partial(negate(diagonal([15.0, 12.0, 9.0]), mask), periodic);
            check_partial(negate(TRIANGULAR, mask), periodic);
            check_partial(negate(rotated(), mask), periodic);
        }
        check_partial(negate(diagonal([15.0, 12.0, 9.0]), mask), ALL);
        check_partial(negate(TRIANGULAR, mask), ALL);
    }
}

#[test]
fn zero_or_non_finite_periodic_edge_is_refused() {
    for bad in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for axis in 0..3 {
            let mut edges = [15.0, -12.0, 9.0];
            edges[axis] = bad;
            let rows = diagonal(edges);
            assert_eq!(
                Cell::ortho(edges[0], edges[1], edges[2]),
                Err(Error::BadBox),
                "{edges:?}"
            );
            assert_eq!(Cell::from_ase(rows), Err(Error::BadBox), "{edges:?}");
            let mut sheared = TRIANGULAR;
            sheared[axis][axis] = bad;
            assert_eq!(Cell::from_ase(sheared), Err(Error::BadBox), "{sheared:?}");
            for periodic in MASKS.into_iter().chain([ALL]) {
                let cell = PartialCell::from_ase(rows, periodic);
                if periodic[axis] {
                    assert_eq!(cell, Err(Error::BadBox), "{edges:?} {periodic:?}");
                } else {
                    assert!(cell.is_ok(), "{edges:?} {periodic:?}");
                }
            }
        }
    }
}
