//! Opt-in face band for [`crate::wrap_many`].
//!
//! The engine wrap is half-open. An orthorhombic axis keeps `-W/2` and
//! maps `+W/2` onto `-W/2`. Restricted and general cells round each
//! fractional coordinate half away from zero, so both `+1/2` and `-1/2`
//! leave the face they sit on. A coordinate that already lies on that
//! face, or within a few ulps of it on either side, is the same atom;
//! folding it jumps by a full edge.
//!
//! The band is Higham's first-order bound `γ_n ≈ n u`, with unit
//! roundoff `u = ε/2` and `n = 256` flops (`2^8`). That product is
//! `128 ε`. Those 256 flops cover the two `3×3` products with `H⁻¹`
//! and `H`, the centering add and subtract a caller does around this
//! wrap, and margin for the rounding mode at a tie. `n u = 2^-45`,
//! far below one face width, so a real image step still wraps.
//!
//! Scaled by the perpendicular face width `W` (the orthorhombic box
//! length, and [`crate::Cell::widths`] on a triclinic cell) the
//! Cartesian half-width of the band is `128 ε W`. In fractional
//! coordinates the half-width is `128 ε` on each lattice axis, because
//! the width cancels. [`WrapConfig::face_slack`] is that fractional
//! half-width. [`WrapConfig::face_tolerance`] multiplies it by `W`.

use crate::kernel;
use crate::Cell;

/// Flops in the first-order bound `γ_n ≈ n u`.
///
/// `2^8` operations. Unit roundoff is `ε/2`, so the bound is
/// `256 * (ε/2) = 128 ε`.
const FACE_BOUND_FLOPS: u32 = 1 << 8;

/// Fractional half-width of the face band, `n u = 128 ε`.
fn face_bound_ulps() -> f64 {
    let flops = f64::from(FACE_BOUND_FLOPS);
    let unit_roundoff = f64::EPSILON / 2.0;
    flops * unit_roundoff
}

/// How [`crate::wrap_many_with`] treats a difference that sits on a face.
///
/// [`Self::default`] is the engine wrap, bit for bit with
/// [`crate::wrap_many`]. [`Self::with_keep_near_face`]`(true)` returns
/// the input difference when the engine result is only a face fold, or
/// when that result already matches the input to within one
/// [`Self::face_tolerance`] on every axis (a round trip through `H⁻¹ H`
/// does not replace an atom that did not move). An axis that is many
/// images away still wraps. Only axes that sit in the face band keep
/// their original fractional coordinate; the Cartesian result is the
/// input bits when every folded axis is in that band.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WrapConfig {
    keep_near_face: bool,
}

impl WrapConfig {
    /// Keep coordinates that already lie in the face band.
    ///
    /// `keep = false` is [`Self::default`].
    #[must_use]
    pub const fn with_keep_near_face(mut self, keep: bool) -> Self {
        self.keep_near_face = keep;
        self
    }

    /// Whether face-band coordinates keep their input value.
    #[must_use]
    pub const fn keep_near_face(self) -> bool {
        self.keep_near_face
    }

    /// Fractional half-width of the face band, `128 ε`.
    ///
    /// A fractional coordinate within this distance of `±1/2` lies on a
    /// primary box face. See the module docs for the `256 u` bound this
    /// number comes from.
    #[must_use]
    pub fn face_slack() -> f64 {
        face_bound_ulps()
    }

    /// Cartesian half-width of the face band on one face.
    ///
    /// [`Self::face_slack`] times `box_length`. `box_length` is the
    /// perpendicular width of that face: the orthorhombic edge, or the
    /// matching entry of [`crate::Cell::widths`].
    #[must_use]
    pub fn face_tolerance(box_length: f64) -> f64 {
        face_bound_ulps() * box_length
    }
}

/// Put face-band coordinates back after the engine wrap.
pub(crate) fn restore_near_faces(cell: &Cell, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (src, dst) in diffs.iter().zip(out.iter_mut()) {
        *dst = restore_one(cell, *src, *dst);
    }
}

fn restore_one(cell: &Cell, d: [f64; 3], w: [f64; 3]) -> [f64; 3] {
    if components_within_tolerance(d, w, cell.widths()) {
        return d;
    }
    let s = fractional(cell, d);
    let sw = fractional(cell, w);
    let h = cell.h();
    let widths = cell.widths();
    let mut out = w;
    let mut any_near = false;
    let mut any_undo = false;
    let mut any_far = false;
    for i in 0..3 {
        let near = match fractional_face_distance(s[i]) {
            Some(dist) => dist <= WrapConfig::face_slack(),
            None => false,
        };
        any_near |= near;
        let image = (s[i] - sw[i]).round();
        let folded = image != 0.0;
        if near && image.abs() == 1.0 {
            any_undo = true;
            if cell.is_ortho() {
                out[i] += image * widths[i];
            } else {
                out[0] += image * h[i][0];
                out[1] += image * h[i][1];
                out[2] += image * h[i][2];
            }
        } else if folded {
            any_far = true;
        }
    }
    // Every change was a face fold, or the point already sits in the
    // band and no axis took a real image. The input is that coordinate.
    if !any_far && (any_undo || any_near) {
        return d;
    }
    if any_undo {
        return out;
    }
    w
}

fn components_within_tolerance(d: [f64; 3], w: [f64; 3], widths: [f64; 3]) -> bool {
    let mut close = true;
    for i in 0..3 {
        let tol = WrapConfig::face_tolerance(widths[i]);
        close &= (d[i] - w[i]).abs() <= tol;
    }
    close
}

/// Fractional coordinates of a Cartesian difference, in the same basis
/// the engine wrap folds. Orthorhombic axes divide by the positive face
/// widths. Every other cell uses the stored inverse.
fn fractional(cell: &Cell, d: [f64; 3]) -> [f64; 3] {
    if cell.is_ortho() {
        let w = cell.widths();
        return [d[0] / w[0], d[1] / w[1], d[2] / w[2]];
    }
    kernel::mul(cell.hinv(), d)
}

/// Distance from `s` to the nearest primary face `±1/2`, or `None` when
/// `s` is not finite.
///
/// For `|s| ∈ [1/4, 1]` the subtraction `|s| - 1/2` is exact by Sterbenz's
/// lemma, so one extra ulp past the band is still outside it, and any
/// non-primary image (`|s| >= 1`) stays outside the band.
fn fractional_face_distance(s: f64) -> Option<f64> {
    if !s.is_finite() {
        return None;
    }
    Some((s.abs() - 0.5).abs())
}
