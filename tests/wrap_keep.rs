//! Face-band wrap. Behaviour of the public API, orthorhombic and triclinic.

use minimage::{wrap_many, wrap_many_with, Cell, Error, WrapConfig};

fn bits(v: [f64; 3]) -> [u64; 3] {
    v.map(f64::to_bits)
}

fn keep() -> WrapConfig {
    WrapConfig::default().with_keep_near_face(true)
}

fn engine(cell: &Cell, rows: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; rows.len()];
    wrap_many(cell, rows, &mut out).unwrap();
    out
}

fn kept(cell: &Cell, rows: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; rows.len()];
    wrap_many_with(cell, rows, &mut out, keep()).unwrap();
    out
}

fn documented_slack() -> f64 {
    let flops = f64::from(1u32 << 8);
    let unit_roundoff = f64::EPSILON / 2.0;
    flops * unit_roundoff
}

/// Next double above a positive finite value.
fn next_up(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

/// Next double below a finite value.
fn next_down(x: f64) -> f64 {
    if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

fn ortho() -> Cell {
    Cell::ortho(8.0, 16.0, 32.0).unwrap()
}

fn row_on_axis(axis: usize, value: f64) -> [f64; 3] {
    let mut row = [0.0; 3];
    row[axis] = value;
    row
}

#[test]
fn face_band_is_256_units_of_roundoff_times_the_face_width() {
    let slack = WrapConfig::face_slack();
    assert_eq!(slack.to_bits(), documented_slack().to_bits());
    let width = 8.0;
    let tol = WrapConfig::face_tolerance(width);
    assert_eq!(tol.to_bits(), (slack * width).to_bits());
    let wider = WrapConfig::face_tolerance(32.0);
    assert_eq!(wider.to_bits(), (tol * 4.0).to_bits());
}

#[test]
fn default_config_matches_engine_wrap_bit_for_bit() {
    let cell = ortho();
    let slack = WrapConfig::face_slack();
    let widths = cell.widths();
    let mut rows = vec![
        [0.0; 3],
        row_on_axis(0, widths[0] / 2.0),
        row_on_axis(0, -widths[0] / 2.0),
        row_on_axis(1, widths[1] / 2.0 + WrapConfig::face_tolerance(widths[1])),
        row_on_axis(2, -widths[2] * 3.25),
        [1.0, -2.5, 40.0],
        [widths[0], widths[1], widths[2]],
    ];
    rows.push(row_on_axis(0, widths[0] * (0.5 + slack / 2.0)));
    let mut via = vec![[0.0; 3]; rows.len()];
    wrap_many_with(&cell, &rows, &mut via, WrapConfig::default()).unwrap();
    let plain = engine(&cell, &rows);
    for (got, want) in via.iter().zip(plain.iter()) {
        assert_eq!(bits(*got), bits(*want));
    }
    // The upper face still folds when the option is off.
    assert_eq!(plain[1][0].to_bits(), (-widths[0] / 2.0).to_bits());
    assert_eq!(plain[2][0].to_bits(), (-widths[0] / 2.0).to_bits());
}

#[test]
fn ortho_face_keeps_original_inside_the_band_and_folds_just_outside() {
    let cell = ortho();
    let widths = cell.widths();
    let slack = WrapConfig::face_slack();
    for axis in 0..3 {
        let width = widths[axis];
        let half = width / 2.0;
        let on_upper = row_on_axis(axis, half);
        let inside = row_on_axis(axis, width * (0.5 - slack / 2.0));
        let outside = row_on_axis(axis, width * (0.5 + slack / 2.0));
        let past_s = next_up(0.5 + slack);
        let past = row_on_axis(axis, width * past_s);
        let on_lower = row_on_axis(axis, -half);
        let lower_inside = row_on_axis(axis, width * (-0.5 + slack / 2.0));
        let lower_outside = row_on_axis(axis, width * (-0.5 - slack / 2.0));
        let lower_past_s = next_down(-(0.5 + slack));
        let lower_past = row_on_axis(axis, width * lower_past_s);

        let rows = [
            on_upper,
            inside,
            outside,
            past,
            on_lower,
            lower_inside,
            lower_outside,
            lower_past,
        ];
        let folded = engine(&cell, &rows);
        let stay = kept(&cell, &rows);

        // Exactly on the upper face: engine maps +W/2 to -W/2; the option keeps it.
        assert_eq!(bits(folded[0]), bits(row_on_axis(axis, -half)));
        assert_eq!(bits(stay[0]), bits(on_upper));

        // Inside the band, still in the primary cell: both leave the bits alone.
        assert_eq!(bits(folded[1]), bits(inside));
        assert_eq!(bits(stay[1]), bits(inside));

        // Outside the face, still inside the band: engine folds, option keeps.
        assert_ne!(bits(folded[2]), bits(outside));
        assert_eq!(bits(stay[2]), bits(outside));
        assert_eq!(
            bits(folded[2]),
            bits(row_on_axis(axis, outside[axis] - width))
        );

        // The next double past the band folds, same bits as the engine.
        assert_ne!(bits(folded[3]), bits(past));
        assert_eq!(bits(stay[3]), bits(folded[3]));

        // Lower face is already the engine's tie. Outside that face, within the
        // band, is the other side: the option keeps it and the engine folds.
        assert_eq!(bits(folded[4]), bits(on_lower));
        assert_eq!(bits(stay[4]), bits(on_lower));
        assert_eq!(bits(folded[5]), bits(lower_inside));
        assert_eq!(bits(stay[5]), bits(lower_inside));
        assert_ne!(bits(folded[6]), bits(lower_outside));
        assert_eq!(bits(stay[6]), bits(lower_outside));
        assert_ne!(bits(folded[7]), bits(lower_past));
        assert_eq!(bits(stay[7]), bits(folded[7]));
    }
}

#[test]
fn the_same_cartesian_offset_tracks_the_face_width() {
    let delta = WrapConfig::face_tolerance(8.0) * 2.0;
    let narrow = Cell::ortho(8.0, 8.0, 8.0).unwrap();
    let wide = Cell::ortho(32.0, 32.0, 32.0).unwrap();
    let near_narrow = [4.0 + delta, 0.0, 0.0];
    let near_wide = [16.0 + delta, 0.0, 0.0];
    let narrow_keep = kept(&narrow, &[near_narrow]);
    let narrow_fold = engine(&narrow, &[near_narrow]);
    let wide_keep = kept(&wide, &[near_wide]);
    // Two slacks past an edge of length 8 is outside that band.
    assert_eq!(bits(narrow_keep[0]), bits(narrow_fold[0]));
    assert_ne!(bits(narrow_fold[0]), bits(near_narrow));
    // The same distance is half a slack on an edge of length 32, so it stays.
    assert_eq!(bits(wide_keep[0]), bits(near_wide));
}

#[test]
fn a_far_image_still_wraps_when_another_axis_is_on_a_face() {
    let cell = ortho();
    let widths = cell.widths();
    // x sits on +W/2. y is 2.25 boxes along, which is not a face.
    let row = [widths[0] / 2.0, widths[1] * 2.25, 0.0];
    let folded = engine(&cell, &[row]);
    let stay = kept(&cell, &[row]);
    assert_eq!(stay[0][0].to_bits(), row[0].to_bits());
    assert_eq!(stay[0][1].to_bits(), folded[0][1].to_bits());
    assert_eq!(stay[0][2].to_bits(), folded[0][2].to_bits());
    assert_ne!(bits(folded[0]), bits(stay[0]));
    assert_eq!(folded[0][0].to_bits(), (-widths[0] / 2.0).to_bits());
}

fn lammpc_triclinic() -> Cell {
    Cell::from_vectors(
        [8.0, 0.0, 0.0],
        [1.0, 8.0, 0.0],
        [0.5, 0.25, 8.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap()
}

fn general_triclinic() -> Cell {
    Cell::from_vectors(
        [8.0, 1.0, 0.0],
        [0.4, 8.0, 0.2],
        [0.1, 0.3, 8.0],
        [1.0, -2.0, 0.5],
    )
    .unwrap()
}

fn at(cell: &Cell, s: [f64; 3]) -> [f64; 3] {
    let r = cell.cartesian(s);
    let o = cell.origin();
    [r[0] - o[0], r[1] - o[1], r[2] - o[2]]
}

fn assert_triclinic_face(cell: &Cell) {
    assert!(!cell.is_ortho());
    let slack = WrapConfig::face_slack();
    let on = at(cell, [0.5, 0.0, 0.0]);
    let inside = at(cell, [0.5 - slack / 2.0, 0.1, -0.2]);
    let outside = at(cell, [0.5 + slack / 2.0, 0.1, -0.2]);
    let past = at(cell, [0.5 + slack + slack, 0.1, -0.2]);
    let lower = at(cell, [-0.5, 0.0, 0.0]);
    let lower_inside = at(cell, [-0.5 + slack / 2.0, 0.0, 0.0]);
    let lower_outside = at(cell, [-0.5 - slack / 2.0, 0.0, 0.0]);
    let lower_past = at(cell, [-0.5 - slack - slack, 0.0, 0.0]);
    let corner = at(cell, [0.5, 0.5, 0.5]);
    let rows = [
        on,
        inside,
        outside,
        past,
        lower,
        lower_inside,
        lower_outside,
        lower_past,
        corner,
    ];
    let folded = engine(cell, &rows);
    let stay = kept(cell, &rows);

    assert_eq!(bits(stay[0]), bits(on));
    assert_ne!(bits(folded[0]), bits(on));

    assert_eq!(bits(stay[1]), bits(inside));
    assert_eq!(bits(stay[2]), bits(outside));
    assert_ne!(bits(folded[2]), bits(outside));

    assert_eq!(bits(stay[3]), bits(folded[3]));
    assert_ne!(bits(folded[3]), bits(past));

    // Both faces fold under the fractional round. The band keeps each side.
    assert_eq!(bits(stay[4]), bits(lower));
    assert_ne!(bits(folded[4]), bits(lower));
    assert_eq!(bits(stay[5]), bits(lower_inside));
    assert_eq!(bits(stay[6]), bits(lower_outside));
    assert_ne!(bits(folded[6]), bits(lower_outside));
    assert_eq!(bits(stay[7]), bits(folded[7]));
    assert_ne!(bits(folded[7]), bits(lower_past));

    assert_eq!(bits(stay[8]), bits(corner));
    assert_ne!(bits(folded[8]), bits(corner));
}

#[test]
fn triclinic_face_keeps_each_side_of_the_band() {
    let restricted = lammpc_triclinic();
    assert!(restricted.is_restricted());
    assert_triclinic_face(&restricted);

    let general = general_triclinic();
    assert!(!general.is_restricted());
    assert_triclinic_face(&general);
}

#[test]
fn triclinic_mixed_face_and_far_image() {
    let cell = lammpc_triclinic();
    let row = at(&cell, [0.5, 2.25, 0.0]);
    let want = at(&cell, [0.5, 0.25, 0.0]);
    let folded_want = at(&cell, [-0.5, 0.25, 0.0]);
    let folded = engine(&cell, &[row]);
    let stay = kept(&cell, &[row]);
    assert_eq!(bits(folded[0]), bits(folded_want));
    assert_eq!(bits(stay[0]), bits(want));
}

#[test]
fn origin_does_not_enter_the_face_band() {
    let at_origin = lammpc_triclinic();
    let shifted = Cell::from_vectors(
        [8.0, 0.0, 0.0],
        [1.0, 8.0, 0.0],
        [0.5, 0.25, 8.0],
        [100.0, -40.0, 7.0],
    )
    .unwrap();
    let row = at(
        &at_origin,
        [0.5 + WrapConfig::face_slack() / 2.0, 0.2, -1.5],
    );
    assert_eq!(
        bits(kept(&at_origin, &[row])[0]),
        bits(kept(&shifted, &[row])[0])
    );
    assert_eq!(
        bits(engine(&at_origin, &[row])[0]),
        bits(engine(&shifted, &[row])[0])
    );
}

#[test]
fn empty_batch_and_a_short_buffer_are_unchanged() {
    let cell = ortho();
    let mut none: [[f64; 3]; 0] = [];
    wrap_many_with(&cell, &[], &mut none, keep()).unwrap();
    let rows = [[0.0; 3]];
    let mut out = [[0.0; 3]; 2];
    assert_eq!(
        wrap_many_with(&cell, &rows, &mut out, WrapConfig::default()),
        Err(Error::BufferSize)
    );
    assert_eq!(
        wrap_many_with(&cell, &rows, &mut out, keep()),
        Err(Error::BufferSize)
    );
}

#[test]
fn non_primary_half_integer_images_still_wrap() {
    let cell = ortho();
    let widths = cell.widths();
    let slack = WrapConfig::face_slack();
    for (axis, &width) in widths.iter().enumerate() {
        let rows = [
            row_on_axis(axis, width * 1.5),
            row_on_axis(axis, width * 2.5),
            row_on_axis(axis, width * -1.5),
            row_on_axis(axis, width * (1.5 + slack / 2.0)),
            row_on_axis(axis, width * (-1.5 - slack / 2.0)),
        ];
        let folded = engine(&cell, &rows);
        let stay = kept(&cell, &rows);
        for (got, want) in stay.iter().zip(folded.iter()) {
            assert_eq!(bits(*got), bits(*want));
        }
    }

    let mixed = [widths[0] * 0.5, widths[1] * 1.5, widths[2] * -2.5];
    let mixed_folded = engine(&cell, &[mixed]);
    let mixed_stay = kept(&cell, &[mixed]);
    assert_eq!(mixed_stay[0][0].to_bits(), (widths[0] * 0.5).to_bits());
    assert_eq!(mixed_stay[0][1].to_bits(), mixed_folded[0][1].to_bits());
    assert_eq!(mixed_stay[0][2].to_bits(), mixed_folded[0][2].to_bits());

    for tri in [lammpc_triclinic(), general_triclinic()] {
        let rows = [
            at(&tri, [1.5, 0.0, 0.0]),
            at(&tri, [-1.5, 0.0, 0.0]),
            at(&tri, [2.5, -1.5, 0.0]),
            at(&tri, [1.5 + slack / 2.0, 0.1, -0.2]),
            at(&tri, [-1.5 - slack / 2.0, -0.1, 0.2]),
        ];
        let folded = engine(&tri, &rows);
        let stay = kept(&tri, &rows);
        for (got, want) in stay.iter().zip(folded.iter()) {
            assert_eq!(bits(*got), bits(*want));
        }

        let tri_mixed = at(&tri, [0.5, 1.5, -1.5]);
        let tri_got = kept(&tri, &[tri_mixed])[0];
        let tri_want = at(&tri, [0.5, -0.5, 0.5]);
        for k in 0..3 {
            assert!((tri_got[k] - tri_want[k]).abs() <= 1e-14);
        }
    }

    let lam = lammpc_triclinic();
    let lam_mixed = at(&lam, [0.5, 1.5, -1.5]);
    assert_eq!(
        bits(kept(&lam, &[lam_mixed])[0]),
        bits(at(&lam, [0.5, -0.5, 0.5]))
    );
}
