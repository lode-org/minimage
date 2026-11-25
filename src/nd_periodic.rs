//! Axis-aligned minimum-image distances in arbitrary dimensions.
//!
//! [`PeriodicSpace`] describes a product metric: each `Some(period)`
//! axis is a circle of that circumference and each `None` axis is the
//! ordinary real line. This is the toroidal metric used by periodic
//! spatial trees. It is separate from [`crate::Cell`], whose lattice
//! vectors may be skew and coupled.
//!
//! Points wrap into `[0, period)`. Differences wrap into
//! `[-period / 2, period / 2)`, so an exact `+period / 2` tie maps to
//! `-period / 2`, while `-period / 2` stays unchanged.

use crate::kernel::{floor_fast, INTEGRAL};
use crate::Error;

/// Squared-distance metric used by N-dimensional spatial trees.
///
/// The default scalar is `f64`. The scalar is a type parameter, rather
/// than an associated implementation detail, so a future float
/// abstraction can add `Metric<f32>` implementations without changing
/// existing `Metric` bounds or method names.
pub trait Metric<T = f64> {
    /// Squared distance between equal-length points.
    fn dist2(&self, p: &[T], q: &[T]) -> T;

    /// Minimum squared distance from coordinate `x` to `[lo, hi]`.
    fn axis_min_dist2(&self, axis: usize, x: T, lo: T, hi: T) -> T;

    /// Minimum squared distance from `point` to `[lower, upper]`.
    fn aabb_min_dist2(&self, point: &[T], lower: &[T], upper: &[T]) -> T;
}

/// Ordinary N-dimensional Euclidean metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Euclidean;

impl Metric for Euclidean {
    #[inline]
    fn dist2(&self, p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len(), "point dimensions differ");
        p.iter()
            .zip(q)
            .map(|(&a, &b)| {
                let delta = b - a;
                delta * delta
            })
            .sum()
    }

    #[inline]
    fn axis_min_dist2(&self, _axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
        open_interval_gap(x, lo, hi).map_or(f64::NAN, |gap| gap * gap)
    }

    #[inline]
    fn aabb_min_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        assert_eq!(point.len(), lower.len(), "point and AABB dimensions differ");
        assert_eq!(point.len(), upper.len(), "point and AABB dimensions differ");
        sum_squared_gaps(
            point
                .iter()
                .zip(lower)
                .zip(upper)
                .map(|((&x, &lo), &hi)| open_interval_gap(x, lo, hi).unwrap_or(f64::NAN)),
        )
    }
}

/// Per-axis periods for an N-dimensional product metric.
///
/// Construction validates the descriptor once. Distance calls then perform
/// only slice-length checks before their inner loops. Batch inputs are
/// contiguous row-major slices with `dimensions()` values per row.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicSpace {
    periods: Box<[Option<f64>]>,
}

trait PeriodSource {
    fn dimensions(&self) -> usize;
    fn period(&self, axis: usize) -> Option<f64>;
}

impl PeriodSource for PeriodicSpace {
    #[inline]
    fn dimensions(&self) -> usize {
        self.periods.len()
    }

    #[inline]
    fn period(&self, axis: usize) -> Option<f64> {
        self.periods[axis]
    }
}

/// Borrowed C period encoding: zero is open and positive is periodic.
pub(crate) struct PeriodicSlice<'a> {
    periods: &'a [f64],
}

impl<'a> PeriodicSlice<'a> {
    pub(crate) fn new(periods: &'a [f64]) -> Result<Self, Error> {
        if periods.is_empty() {
            return Err(Error::Empty);
        }
        if periods
            .iter()
            .any(|period| !period.is_finite() || *period < 0.0)
        {
            return Err(Error::BadBox);
        }
        Ok(Self { periods })
    }

    pub(crate) fn wrap_point(&self, point: &mut [f64]) {
        wrap_point_row(self, point);
    }

    pub(crate) fn wrap_difference(&self, difference: &mut [f64]) {
        wrap_difference_row(self, difference);
    }

    pub(crate) fn dist2(&self, p: &[f64], q: &[f64]) -> f64 {
        dist2_rows(self, p, q)
    }

    pub(crate) fn dist2_many(&self, point: &[f64], points: &[f64], out: &mut [f64]) {
        for (row, value) in points.chunks_exact(self.dimensions()).zip(out) {
            *value = dist2_rows(self, point, row);
        }
    }

    pub(crate) fn axis_min_dist2(&self, axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
        axis_min_dist2(self, axis, x, lo, hi)
    }

    pub(crate) fn aabb_min_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        aabb_min_dist2(self, point, lower, upper)
    }
}

impl PeriodSource for PeriodicSlice<'_> {
    #[inline]
    fn dimensions(&self) -> usize {
        self.periods.len()
    }

    #[inline]
    fn period(&self, axis: usize) -> Option<f64> {
        let period = self.periods[axis];
        (period != 0.0).then_some(period)
    }
}

impl PeriodicSpace {
    /// Construct a descriptor from one optional period per dimension.
    ///
    /// `Some(length)` selects a periodic axis and `None` an open axis.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Empty`] for no dimensions and [`Error::BadBox`] if
    /// a period is non-finite or not strictly positive.
    pub fn new(periods: &[Option<f64>]) -> Result<Self, Error> {
        if periods.is_empty() {
            return Err(Error::Empty);
        }
        if periods
            .iter()
            .flatten()
            .any(|length| !length.is_finite() || *length <= 0.0)
        {
            return Err(Error::BadBox);
        }
        Ok(Self {
            periods: periods.into(),
        })
    }

    /// Number of dimensions.
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.periods.len()
    }

    /// Validated per-axis periods.
    #[inline]
    pub fn periods(&self) -> &[Option<f64>] {
        &self.periods
    }

    /// Wrap one point in place into the half-open primary domain.
    ///
    /// Periodic components are in `[0, period)` and open components are
    /// unchanged. A non-finite periodic component becomes `NaN`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] unless `point.len() == dimensions()`.
    pub fn wrap_point(&self, point: &mut [f64]) -> Result<(), Error> {
        self.check_row(point)?;
        wrap_point_row(self, point);
        Ok(())
    }

    /// Wrap one difference vector in place into the central half-open image.
    ///
    /// Periodic components are in `[-period / 2, period / 2)` and open
    /// components are unchanged. A non-finite periodic component becomes
    /// `NaN`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] unless `difference.len()` equals
    /// [`dimensions()`](Self::dimensions).
    pub fn wrap_difference(&self, difference: &mut [f64]) -> Result<(), Error> {
        self.check_row(difference)?;
        wrap_difference_row(self, difference);
        Ok(())
    }

    /// Write the wrapped displacement `q - p` to `out`.
    ///
    /// Each periodic component is the exact minimum image of `q - p` in
    /// `[-period / 2, period / 2)`, rounded to nearest. That rounding can
    /// reach `period / 2` itself when the exact image is within half an ulp
    /// below it. Open components are `q - p` rounded to nearest.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] unless all slices have
    /// [`dimensions()`](Self::dimensions) entries.
    pub fn displacement(&self, p: &[f64], q: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.check_row(p)?;
        self.check_row(q)?;
        self.check_row(out)?;
        for axis in 0..self.dimensions() {
            out[axis] = difference_component(p[axis], q[axis], self.periods[axis]);
        }
        Ok(())
    }

    /// Squared minimum-image distance between two points.
    ///
    /// This is the sum of the squared [`displacement`](Self::displacement)
    /// components and the allocation-free hot path. It panics if either
    /// point does not have [`dimensions()`](Self::dimensions) entries.
    #[inline]
    pub fn dist2(&self, p: &[f64], q: &[f64]) -> f64 {
        <Self as Metric>::dist2(self, p, q)
    }

    /// Minimum-image distance between two points.
    ///
    /// Panics under the same conditions as [`dist2`](Self::dist2).
    #[inline]
    pub fn distance(&self, p: &[f64], q: &[f64]) -> f64 {
        self.dist2(p, q).sqrt()
    }

    /// Wrap contiguous row-major points in place.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] unless `points.len()` is a multiple of
    /// [`dimensions()`](Self::dimensions).
    pub fn wrap_points(&self, points: &mut [f64]) -> Result<(), Error> {
        self.check_rows(points)?;
        for row in points.chunks_exact_mut(self.dimensions()) {
            wrap_point_row(self, row);
        }
        Ok(())
    }

    /// Wrap contiguous row-major difference vectors in place.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] unless `differences.len()` is a
    /// multiple of [`dimensions()`](Self::dimensions).
    pub fn wrap_differences(&self, differences: &mut [f64]) -> Result<(), Error> {
        self.check_rows(differences)?;
        for row in differences.chunks_exact_mut(self.dimensions()) {
            wrap_difference_row(self, row);
        }
        Ok(())
    }

    /// Squared distances from one point to contiguous row-major points.
    ///
    /// `out.len()` is the number of rows in `points`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] for a point of the wrong dimension, a
    /// partial row, or an output length different from the row count.
    pub fn dist2_many(&self, point: &[f64], points: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.check_row(point)?;
        self.check_rows(points)?;
        if out.len() != points.len() / self.dimensions() {
            return Err(Error::BufferSize);
        }
        for (row, value) in points.chunks_exact(self.dimensions()).zip(out) {
            *value = dist2_rows(self, point, row);
        }
        Ok(())
    }

    /// Squared distances between corresponding contiguous row-major points.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] for partial rows, unequal input
    /// lengths, or an output length different from the row count.
    pub fn dist2_pairs(&self, left: &[f64], right: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.check_rows(left)?;
        if left.len() != right.len() || out.len() != left.len() / self.dimensions() {
            return Err(Error::BufferSize);
        }
        for ((p, q), value) in left
            .chunks_exact(self.dimensions())
            .zip(right.chunks_exact(self.dimensions()))
            .zip(out)
        {
            *value = dist2_rows(self, p, q);
        }
        Ok(())
    }

    /// Minimum squared distance from one coordinate to an interval.
    ///
    /// `lo` and `hi` describe the ordinary non-crossing interval
    /// `[lo, hi]`. On a periodic axis an interval at least one period wide
    /// covers the circle. Shorter intervals may lie outside the primary
    /// domain; translated images are considered. Intervals wider than half
    /// a period are handled directly, without complement approximations.
    /// The exact distance is rounded toward zero before squaring, so the
    /// result never exceeds the correctly rounded exact squared distance.
    ///
    /// Returns `NaN` for non-finite bounds or `lo > hi`. Panics if `axis`
    /// is outside this space.
    #[inline]
    pub fn axis_min_dist2(&self, axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
        <Self as Metric>::axis_min_dist2(self, axis, x, lo, hi)
    }

    /// Squared lower-bound distance from a point to an axis-aligned box.
    ///
    /// `lower` and `upper` describe the ordinary non-crossing interval
    /// `[lower[i], upper[i]]` on each axis. On a periodic axis an interval
    /// at least one period wide covers the circle. Shorter intervals may
    /// lie outside the primary domain; translated images are considered.
    /// The exact one-axis minima sum to the exact product-metric distance
    /// to the box. The result never exceeds that distance correctly
    /// rounded, nor [`dist2`](Self::dist2) to any point of the box, so it
    /// is a sound tree-pruning bound.
    ///
    /// Panics unless all slices have [`dimensions()`](Self::dimensions)
    /// entries. Invalid bounds produce `NaN`.
    #[inline]
    pub fn aabb_min_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        <Self as Metric>::aabb_min_dist2(self, point, lower, upper)
    }

    /// Compatibility spelling for [`aabb_min_dist2`](Self::aabb_min_dist2).
    #[inline]
    pub fn box_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        self.aabb_min_dist2(point, lower, upper)
    }

    /// Lower-bound distance from a point to an axis-aligned box.
    ///
    /// Panics under the same conditions as
    /// [`aabb_min_dist2`](Self::aabb_min_dist2).
    #[inline]
    pub fn box_distance(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        self.aabb_min_dist2(point, lower, upper).sqrt()
    }

    #[inline]
    fn check_row(&self, row: &[f64]) -> Result<(), Error> {
        if row.len() == self.dimensions() {
            Ok(())
        } else {
            Err(Error::BufferSize)
        }
    }

    #[inline]
    fn check_rows(&self, rows: &[f64]) -> Result<(), Error> {
        if rows.len() % self.dimensions() == 0 {
            Ok(())
        } else {
            Err(Error::BufferSize)
        }
    }
}

impl Metric for PeriodicSpace {
    #[inline]
    fn dist2(&self, p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), self.dimensions(), "point dimension mismatch");
        assert_eq!(q.len(), self.dimensions(), "point dimension mismatch");
        dist2_rows(self, p, q)
    }

    #[inline]
    fn axis_min_dist2(&self, axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
        axis_min_dist2(self, axis, x, lo, hi)
    }

    #[inline]
    fn aabb_min_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        assert_eq!(point.len(), self.dimensions(), "point dimension mismatch");
        assert_eq!(lower.len(), self.dimensions(), "AABB dimension mismatch");
        assert_eq!(upper.len(), self.dimensions(), "AABB dimension mismatch");
        aabb_min_dist2(self, point, lower, upper)
    }
}

#[inline]
fn wrap_point_row(periods: &impl PeriodSource, point: &mut [f64]) {
    for (axis, value) in point.iter_mut().enumerate() {
        if let Some(period) = periods.period(axis) {
            *value = wrap_point_component(*value, period);
        }
    }
}

#[inline]
fn wrap_difference_row(periods: &impl PeriodSource, difference: &mut [f64]) {
    for (axis, value) in difference.iter_mut().enumerate() {
        *value = wrap_difference_component(*value, periods.period(axis));
    }
}

#[inline]
fn dist2_rows(periods: &impl PeriodSource, p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q)
        .enumerate()
        .map(|(axis, (&a, &b))| {
            let delta = difference_component(a, b, periods.period(axis));
            delta * delta
        })
        .sum()
}

#[inline]
fn axis_min_dist2(periods: &impl PeriodSource, axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
    let gap = axis_gap(periods, axis, x, lo, hi);
    gap * gap
}

/// Sound box bound: never above the correctly rounded exact squared
/// distance, nor above `dist2` to any point of the box.
#[inline]
fn aabb_min_dist2(periods: &impl PeriodSource, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    sum_squared_gaps(
        (0..periods.dimensions())
            .map(|axis| axis_gap(periods, axis, point[axis], lower[axis], upper[axis])),
    )
}

/// Distance from `x` to `[lo, hi]` on one axis, rounded toward zero.
///
/// `NaN` for invalid bounds. Rounding toward zero keeps the square, and
/// sums of squares built with [`sum_squared_gaps`], at or below their exact
/// values.
#[inline]
fn axis_gap(periods: &impl PeriodSource, axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
    match periods.period(axis) {
        None => open_interval_gap(x, lo, hi).unwrap_or(f64::NAN),
        Some(_) if !x.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi => f64::NAN,
        Some(_) if lo <= x && x <= hi => 0.0,
        Some(period) => periodic_interval_gap(x, lo, hi, period),
    }
}

/// Ordinary distance from `x` to `[lo, hi]` rounded toward zero, or `None`
/// for non-finite bounds or `lo > hi`.
#[inline]
fn open_interval_gap(x: f64, lo: f64, hi: f64) -> Option<f64> {
    if !x.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
        None
    } else if x < lo {
        Some(toward_zero(ordered_two_sum(lo, -x)))
    } else if x > hi {
        Some(toward_zero(ordered_two_sum(x, -hi)))
    } else {
        Some(0.0)
    }
}

/// Circular distance from `x` to the arc `[lo, hi]`, rounded toward zero.
///
/// Every comparison is on exact unevaluated sums, so the inside test and
/// the nearest endpoint are decided exactly.
#[inline]
fn periodic_interval_gap(x: f64, lo: f64, hi: f64, period: f64) -> f64 {
    let width = ordered_two_sum(hi, -lo);
    if width.0 > period || (width.0 == period && width.1 >= 0.0) {
        return 0.0;
    }
    let to_lo = normalize(image_pair(lo, x, period));
    // `x` is inside when its forward offset from `lo`, in `[0, period)`, is
    // at most the width. A negative image offsets by `image + period`.
    let inside = if to_lo.0 >= 0.0 {
        pair_le(to_lo, width)
    } else {
        // Below half a period the shifted width is under every image. At or
        // above it, `width - period` is exact by Sterbenz.
        2.0 * width.0 >= period && pair_le(to_lo, two_sum(width.0 - period, width.1))
    };
    if inside {
        return 0.0;
    }
    let to_hi = normalize(image_pair(hi, x, period));
    toward_zero(to_lo).abs().min(toward_zero(to_hi).abs())
}

/// Sum of squared gaps that is never above the correctly rounded exact
/// sum.
///
/// A single nonzero gap squares with one rounding, which is monotone. Two
/// or more round each square and partial sum down, so the result is also
/// at or below the rounded-to-nearest `dist2` of any point at least that
/// far away on every axis.
#[inline]
fn sum_squared_gaps(gaps: impl Iterator<Item = f64>) -> f64 {
    let mut nonzero = 0_usize;
    let mut single = 0.0;
    let mut sum = 0.0;
    for gap in gaps {
        if gap != 0.0 {
            nonzero += 1;
            single = gap;
            sum = add_down(sum, square_down(gap));
        }
    }
    if nonzero == 1 {
        single * single
    } else {
        sum
    }
}

/// `(fl(a + b), a + b - fl(a + b))` by Knuth's branch-free two-sum.
///
/// Exact unless an intermediate overflows, which can happen for operands
/// near `f64::MAX` even when the sum is finite. Any overflow leaves the
/// error term infinite or `NaN`.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_part = sum - a;
    let a_part = sum - b_part;
    (sum, (a - a_part) + (b - b_part))
}

/// [`two_sum`] by Fast2Sum on operands ordered by magnitude, where no
/// intermediate can overflow before the sum does.
#[inline]
fn ordered_two_sum(a: f64, b: f64) -> (f64, f64) {
    let (large, small) = if a.abs() < b.abs() { (b, a) } else { (a, b) };
    let sum = large + small;
    (sum, small - (sum - large))
}

/// The same exact sum with its leading term rounded to nearest.
#[inline]
fn normalize((head, tail): (f64, f64)) -> (f64, f64) {
    two_sum(head, tail)
}

/// Exact `a <= b` for normalized pairs.
///
/// The head of a normalized pair is its sum rounded to nearest, which is
/// monotone and unique, so heads order the sums unless they tie.
#[inline]
fn pair_le(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 <= b.1)
}

/// A normalized pair's sum rounded toward zero.
#[inline]
fn toward_zero((head, tail): (f64, f64)) -> f64 {
    step_down(head, tail != 0.0 && (tail < 0.0) != (head < 0.0))
}

/// The next double toward zero from a nonzero `value` when `step` holds.
///
/// Selected arithmetically: these steps follow rounding errors, whose
/// signs a branch predictor cannot learn.
#[inline]
fn step_down(value: f64, step: bool) -> f64 {
    f64::from_bits(value.to_bits() - u64::from(step))
}

/// Dekker's split product is exact for a square between these limits: the
/// error term stays normal and no partial product overflows.
const EXACT_SQUARE_MIN: f64 = 3.203_332_952_292_961_5e-145; // 2^-480
const EXACT_SQUARE_MAX: f64 = 3.273_390_607_896_142e150; // 2^500

/// `value * value` rounded down for `value >= 0`, or an infinity whose
/// exact square also rounds to infinity.
#[inline]
fn square_down(value: f64) -> f64 {
    let square = value * value;
    if !(EXACT_SQUARE_MIN..EXACT_SQUARE_MAX).contains(&value) {
        return step_down(square, square != 0.0 && square.is_finite());
    }
    let split = 134_217_729.0 * value;
    let high = split - (split - value);
    let low = value - high;
    let error = ((high * high - square) + 2.0 * high * low) + low * low;
    step_down(square, error < 0.0)
}

/// `a + b` rounded down for non-negative terms. An infinite term stays
/// infinite.
#[inline]
fn add_down(a: f64, b: f64) -> f64 {
    let (sum, error) = two_sum(a, b);
    step_down(sum, error < 0.0)
}

/// Minimum image of `to - from`, correctly rounded.
///
/// The exact image lies in `[-period / 2, period / 2)`; rounding it to
/// nearest can give `period / 2` itself.
#[inline]
fn difference_component(from: f64, to: f64, period: Option<f64>) -> f64 {
    let raw = to - from;
    let Some(period) = period else {
        return raw;
    };
    // `|raw| < period / 2` puts the exact difference strictly inside the
    // central image, so `raw` is already its correctly rounded value.
    if 2.0 * raw.abs() < period {
        return raw;
    }
    let (head, tail) = image_pair(from, to, period);
    head + tail
}

/// Exact minimum image of `to - from` as an unevaluated sum in
/// `[-period / 2, period / 2)`, or `NaN` for a non-finite input.
///
/// A difference within a few periods keeps its two-sum error term while
/// whole periods come off its head: subtracting `2^j period` from a head in
/// `(2^j period, 2^(j + 1) period]` is exact by Sterbenz. A longer
/// difference reduces each coordinate with `%`, which is exact, so the pair
/// always sums to the exact image.
#[inline]
fn image_pair(from: f64, to: f64, period: f64) -> (f64, f64) {
    let (mut head, tail) = two_sum(to, -from);
    if !tail.is_finite() || head.abs() > 16.0 * period {
        return far_image_pair(from, to, period);
    }
    if head.abs() <= period {
        return fold_pair(head, tail, period);
    }
    for scale in [8.0, 4.0, 2.0, 1.0] {
        let step = scale * period;
        if head.abs() > step {
            head -= step.copysign(head);
        }
    }
    // The tail may now exceed half an ulp of the smaller head.
    let (head, tail) = two_sum(head, tail);
    fold_pair(head, tail, period)
}

#[cold]
#[inline(never)]
fn far_image_pair(from: f64, to: f64, period: f64) -> (f64, f64) {
    if !from.is_finite() || !to.is_finite() {
        return (f64::NAN, 0.0);
    }
    let (sum, error) = ordered_two_sum(
        fold_half(to % period, period),
        -fold_half(from % period, period),
    );
    fold_pair(sum, error, period)
}

/// Shift the exact sum `head + tail`, within one period of zero, into
/// `[-period / 2, period / 2)`.
///
/// `head` is the sum rounded to nearest and `period / 2` is a double, so
/// by monotonicity `head` is on the same side of `+-period / 2` as the
/// exact sum unless it equals it, when the tail's sign decides. A nonzero
/// tail needs a sum of at least `2^-1021`, so `period / 2` is exact
/// whenever the tail matters. The shift by `period` is exact by Sterbenz.
#[inline]
fn fold_pair(head: f64, tail: f64, period: f64) -> (f64, f64) {
    let twice = 2.0 * head;
    let up = twice > period || (twice == period && tail >= 0.0);
    let down = twice < -period || (twice == -period && tail < 0.0);
    let shift = if up {
        period
    } else if down {
        -period
    } else {
        0.0
    };
    (head - shift, tail)
}

#[inline]
fn wrap_point_component(value: f64, period: f64) -> f64 {
    if (0.0..period).contains(&value) {
        return if value <= 0.0 { 0.0 } else { value };
    }
    let quotient = value / period;
    if quotient.is_finite() && quotient.abs() < INTEGRAL {
        let mut wrapped = value - period * floor_fast(quotient);
        if wrapped >= period {
            wrapped -= period;
        }
        // The image shift can overflow near f64::MAX, and its rounding can
        // land outside the period; both take the exact remainder instead.
        if (0.0..period).contains(&wrapped) {
            return if wrapped <= 0.0 { 0.0 } else { wrapped };
        }
    }
    wrap_point_exact(value, period)
}

#[cold]
#[inline(never)]
fn wrap_point_exact(value: f64, period: f64) -> f64 {
    let mut wrapped = value.rem_euclid(period);
    if wrapped >= period {
        wrapped -= period;
    }
    if wrapped <= 0.0 {
        0.0
    } else {
        wrapped
    }
}

#[inline]
fn wrap_difference_component(value: f64, period: Option<f64>) -> f64 {
    let Some(period) = period else {
        return value;
    };
    if value.abs() < period {
        return wrap_near(value, period);
    }
    if !value.is_finite() {
        return f64::NAN;
    }
    let quotient = value / period;
    if quotient.is_finite() && quotient.abs() < INTEGRAL {
        let wrapped = fold_half(value - period * floor_fast(quotient + 0.5), period);
        // As for points, an overflowed or misrounded shift takes the exact
        // remainder.
        if 2.0 * wrapped >= -period && 2.0 * wrapped < period {
            return wrapped;
        }
    }
    fold_half(value % period, period)
}

// The half-period tests compare 2 * value with the period. Doubling is exact
// and an overflow to infinity keeps the order, while 0.5 * period rounds to
// zero for the smallest subnormal periods. For a normal period both forms
// decide alike, so this matches `wrap_one`, which `Cell` keeps, bit for bit.

/// `wrap_one` for `|value| < period`.
#[inline]
fn wrap_near(value: f64, period: f64) -> f64 {
    let wrapped = value + if 2.0 * value < -period { period } else { 0.0 };
    wrapped - if 2.0 * wrapped >= period { period } else { 0.0 }
}

/// One correction toward `[-period / 2, period / 2)`. Exact when `value`
/// is an exact remainder, which is within one period of zero.
#[inline]
fn fold_half(mut value: f64, period: f64) -> f64 {
    if 2.0 * value < -period {
        value += period;
    }
    if 2.0 * value >= period {
        value -= period;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        }

        fn range(&mut self, low: f64, high: f64) -> f64 {
            low + (high - low) * self.next()
        }
    }

    #[test]
    fn rejects_empty_zero_negative_nan_and_infinite_periods() {
        assert_eq!(PeriodicSpace::new(&[]), Err(Error::Empty));
        assert_eq!(
            PeriodicSpace::new(&[]).unwrap_err().to_string(),
            "empty input"
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(PeriodicSpace::new(&[Some(bad)]), Err(Error::BadBox));
        }
    }

    #[test]
    fn half_period_and_nan_conventions() {
        let metric = PeriodicSpace::new(&[Some(10.0), None]).unwrap();
        let mut positive = [5.0, f64::NAN];
        metric.wrap_difference(&mut positive).unwrap();
        assert_eq!(positive[0], -5.0);
        assert!(positive[1].is_nan());

        let mut negative = [-5.0, 2.0];
        metric.wrap_difference(&mut negative).unwrap();
        assert_eq!(negative, [-5.0, 2.0]);

        let mut point = [f64::NAN, 3.0];
        metric.wrap_point(&mut point).unwrap();
        assert!(point[0].is_nan());
        assert_eq!(point[1], 3.0);
        assert!(metric.dist2(&[f64::NAN, 0.0], &[0.0, 0.0]).is_nan());
    }

    #[test]
    fn point_wrap_is_always_in_the_half_open_period() {
        let period = 10.0;
        let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
        let below_period = period * (1.0 - f64::EPSILON);
        for value in [
            -1e-20,
            -0.0,
            below_period,
            -2.0 * period,
            -period,
            0.0,
            period,
            2.0 * period,
        ] {
            let mut point = [value];
            metric.wrap_point(&mut point).unwrap();
            assert!(
                point[0] >= 0.0 && point[0] < period,
                "{value} -> {}",
                point[0]
            );
        }

        let mut rng = Lcg(0x5752_4150_504f_494e);
        for _ in 0..200_000 {
            let image = rng.range(-1_000_000.0, 1_000_000.0).trunc();
            let side = if rng.next() < 0.5 { -1.0 } else { 1.0 };
            let jitter = side * period * f64::EPSILON * rng.range(0.0, 8.0);
            let mut point = [image * period + jitter];
            metric.wrap_point(&mut point).unwrap();
            assert!(point[0] >= 0.0 && point[0] < period);
        }
    }

    #[test]
    fn point_wrap_survives_an_image_shift_past_f64_max() {
        for (period, expected) in [
            (1e307, 2.306_865_137_684_266_7e305),
            (1e308, 2.023_068_651_376_843_1e307),
        ] {
            let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
            let mut point = [-f64::MAX];
            metric.wrap_point(&mut point).unwrap();
            assert_eq!(point[0], expected, "period {period:e}");
            // The shift for +f64::MAX does not overflow, so the fast path
            // keeps its product rounding of about one ulp of the input.
            let mut point = [f64::MAX];
            metric.wrap_point(&mut point).unwrap();
            let ulp_max = f64::MAX - f64::from_bits(f64::MAX.to_bits() - 1);
            assert!(point[0] >= 0.0 && point[0] < period);
            assert!(
                (point[0] - (period - expected)).abs() <= 2.0 * ulp_max,
                "period {period:e}: {}",
                point[0]
            );
        }
    }

    #[test]
    fn difference_wrap_survives_an_image_shift_past_f64_max() {
        for (period, expected) in [
            (1e307, 2.306_865_137_684_266_7e305),
            (1e308, 2.023_068_651_376_843_1e307),
        ] {
            let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
            let mut difference = [-f64::MAX, f64::MAX];
            for value in &mut difference {
                *value = wrap_difference_component(*value, Some(period));
            }
            assert_eq!(difference, [expected, -expected], "period {period:e}");
            let mut difference = [-f64::MAX];
            metric.wrap_difference(&mut difference).unwrap();
            assert_eq!(difference[0], expected, "period {period:e}");
        }
    }

    #[test]
    fn subnormal_period_keeps_the_half_open_difference_interval() {
        let unit = f64::from_bits(1);
        let metric = PeriodicSpace::new(&[Some(unit)]).unwrap();
        let mut difference = [0.0];
        metric.wrap_difference(&mut difference).unwrap();
        assert_eq!(difference[0].to_bits(), 0.0_f64.to_bits());

        // Multiples of the smallest subnormal and their minimum images.
        for (period, cases) in [
            (1, &[(0, 0), (1, 0), (-1, 0), (7, 0)][..]),
            (2, &[(0, 0), (1, -1), (-1, -1), (2, 0), (3, -1)][..]),
            (3, &[(0, 0), (1, 1), (-1, -1), (2, -1), (-2, 1), (4, 1)][..]),
        ] {
            let period = f64::from(period) * unit;
            for &(value, expected) in cases {
                let value = f64::from(value) * unit;
                let expected = f64::from(expected) * unit;
                let got = wrap_difference_component(value, Some(period));
                assert_eq!(got, expected, "{value:e} mod {period:e}");
                assert!(2.0 * got >= -period && 2.0 * got < period);
            }
        }
    }

    /// Remainder of `value` by `period` with the sign of `value`, by binary
    /// long division over `period * 2^k`. Each step subtracts `s` from a
    /// remainder in `[s, 2s)`, which is exact, so no `%` is involved.
    fn exact_rem(value: f64, period: f64) -> f64 {
        let mut rem = value.abs();
        let mut scales = vec![period];
        while let Some(&top) = scales.last() {
            if top > f64::MAX / 2.0 || top + top > rem {
                break;
            }
            scales.push(top + top);
        }
        for &scale in scales.iter().rev() {
            if rem >= scale {
                rem -= scale;
            }
        }
        rem.copysign(value)
    }

    fn exact_point(value: f64, period: f64) -> f64 {
        let rem = exact_rem(value, period);
        if rem == 0.0 {
            return 0.0;
        }
        if rem > 0.0 {
            return rem;
        }
        let shifted = rem + period;
        if shifted >= period {
            0.0
        } else {
            shifted
        }
    }

    fn exact_difference(value: f64, period: f64) -> f64 {
        let rem = exact_rem(value, period);
        if 2.0 * rem >= period {
            rem - period
        } else if 2.0 * rem < -period {
            rem + period
        } else {
            rem
        }
    }

    fn ulp(value: f64) -> f64 {
        let magnitude = value.abs();
        if magnitude == 0.0 {
            return f64::from_bits(1);
        }
        magnitude - f64::from_bits(magnitude.to_bits() - 1)
    }

    fn circular_gap(a: f64, b: f64, period: f64) -> f64 {
        let gap = (a - b).abs();
        gap.min(period - gap)
    }

    /// A finite double with a uniformly drawn exponent field.
    fn any_magnitude(rng: &mut Lcg) -> f64 {
        let exponent = (rng.next() * 2047.0) as u64;
        let mantissa = rng.0 >> 12;
        let sign = if rng.next() < 0.5 { 1 << 63 } else { 0 };
        f64::from_bits(sign | (exponent << 52) | mantissa)
    }

    #[test]
    fn wraps_match_an_exact_reference_over_extreme_magnitudes() {
        let mut rng = Lcg(0x4558_5452_454d_4553);
        for case in 0..20_000 {
            let period = match case % 4 {
                0 => any_magnitude(&mut rng).abs(),
                1 => f64::from_bits(1 + (rng.0 >> 12) % 16),
                2 => f64::from_bits(1 + (rng.0 >> 12)),
                _ => f64::MAX * rng.range(0.5, 1.0),
            };
            if period == 0.0 {
                continue;
            }
            let near = (rng.range(-8.0, 8.0).trunc() * period).clamp(-f64::MAX, f64::MAX);
            for value in [
                any_magnitude(&mut rng),
                any_magnitude(&mut rng),
                near,
                f64::from_bits(near.to_bits().wrapping_add(1)),
                f64::from_bits(near.to_bits().wrapping_sub(1)),
                f64::MAX,
                -f64::MAX,
                0.0,
                -0.0,
                f64::from_bits(1),
                -f64::from_bits(1),
                period * rng.range(-1.0, 1.0),
            ] {
                if !value.is_finite() {
                    continue;
                }
                let point = wrap_point_component(value, period);
                let difference = wrap_difference_component(value, Some(period));
                assert!(
                    (0.0..period).contains(&point),
                    "{value:e} mod {period:e}: point {point:e}"
                );
                assert!(
                    2.0 * difference >= -period && 2.0 * difference < period,
                    "{value:e} mod {period:e}: difference {difference:e}"
                );

                let want_point = exact_point(value, period);
                let want_difference = exact_difference(value, period);
                // The fast path rounds one image shift, which is bounded by
                // the input's own ulp. Every other path is exact.
                let quotient = value / period;
                let fast = quotient.is_finite() && quotient.abs() < INTEGRAL;
                if !fast || value.abs() < period {
                    assert_eq!(point, want_point, "{value:e} mod {period:e}");
                    assert_eq!(difference, want_difference, "{value:e} mod {period:e}");
                } else {
                    let tolerance = 4.0 * (ulp(value) + ulp(period));
                    assert!(
                        circular_gap(point, want_point, period) <= tolerance,
                        "{value:e} mod {period:e}: point {point:e}, exact {want_point:e}"
                    );
                    assert!(
                        circular_gap(difference, want_difference, period) <= tolerance,
                        "{value:e} mod {period:e}: difference {difference:e}, exact {want_difference:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn difference_wrap_enforces_half_open_range_near_ties_and_far_away() {
        let period = 3.0;
        let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
        for value in [
            1.499_999_999_999_999_8,
            1.5,
            -1.5,
            -1.500_000_000_000_000_2,
            1e17,
            -1e17,
        ] {
            let mut difference = [value];
            metric.wrap_difference(&mut difference).unwrap();
            assert!(
                difference[0] >= -0.5 * period && difference[0] < 0.5 * period,
                "{value} -> {}",
                difference[0]
            );
        }

        let tiny = PeriodicSpace::new(&[Some(1e-100)]).unwrap();
        let value = tiny.dist2(&[1e300], &[0.0]);
        assert!(value.is_finite());
    }

    #[test]
    fn properties_match_brute_force_image_shifts_in_one_to_eight_dimensions() {
        let mut rng = Lcg(0x4d49_4e49_4d41_4745);
        for dimensions in 1..=8 {
            for case in 0..160 {
                let periods: Vec<Option<f64>> = (0..dimensions)
                    .map(|axis| {
                        if (axis + case) % 3 == 0 {
                            None
                        } else {
                            Some(rng.range(0.25, 20.0))
                        }
                    })
                    .collect();
                let metric = PeriodicSpace::new(&periods).unwrap();
                let mut p = Vec::with_capacity(dimensions);
                let mut q = Vec::with_capacity(dimensions);
                for period in &periods {
                    let scale = period.unwrap_or(8.0);
                    p.push(rng.range(-1.25 * scale, 1.25 * scale));
                    q.push(rng.range(-1.25 * scale, 1.25 * scale));
                }

                let got = metric.dist2(&p, &q);
                let expected: f64 = (0..dimensions)
                    .map(|axis| {
                        let raw = q[axis] - p[axis];
                        match periods[axis] {
                            None => raw * raw,
                            Some(period) => (-3..=3)
                                .map(|shift| {
                                    let image = raw + f64::from(shift) * period;
                                    image * image
                                })
                                .fold(f64::INFINITY, f64::min),
                        }
                    })
                    .sum();
                assert!(
                    (got - expected).abs() <= 1e-12 * (1.0 + expected),
                    "d={dimensions} case={case}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn row_major_batches_match_scalar_paths() {
        let metric = PeriodicSpace::new(&[Some(360.0), None, Some(2.0)]).unwrap();
        let point = [350.0, 2.0, 0.25];
        let points = [10.0, 5.0, 1.75, 180.0, -1.0, 0.5];
        let mut out = [0.0; 2];
        metric.dist2_many(&point, &points, &mut out).unwrap();
        assert_eq!(out[0], metric.dist2(&point, &points[..3]));
        assert_eq!(out[1], metric.dist2(&point, &points[3..]));

        let mut wrapped = points;
        metric.wrap_points(&mut wrapped).unwrap();
        assert_eq!(wrapped, [10.0, 5.0, 1.75, 180.0, -1.0, 0.5]);
        assert_eq!(
            metric.dist2_many(&point, &points[..5], &mut out),
            Err(Error::BufferSize)
        );
    }

    #[test]
    fn box_bound_never_exceeds_sampled_distance() {
        let mut rng = Lcg(0x424f_585f_424f_554e);
        for dimensions in 1..=8 {
            for case in 0..120 {
                let periods: Vec<Option<f64>> = (0..dimensions)
                    .map(|axis| {
                        if (axis + case) % 2 == 0 {
                            Some(rng.range(0.5, 12.0))
                        } else {
                            None
                        }
                    })
                    .collect();
                let metric = PeriodicSpace::new(&periods).unwrap();
                let mut point = Vec::new();
                let mut lower = Vec::new();
                let mut upper = Vec::new();
                for period in &periods {
                    let scale = period.unwrap_or(6.0);
                    let lo = rng.range(-scale, scale);
                    let width = rng.range(0.0, 1.4 * scale);
                    lower.push(lo);
                    upper.push(lo + width);
                    point.push(rng.range(-2.0 * scale, 2.0 * scale));
                }
                let bound = metric.aabb_min_dist2(&point, &lower, &upper);
                for _ in 0..24 {
                    let sample: Vec<f64> = lower
                        .iter()
                        .zip(&upper)
                        .map(|(&lo, &hi)| rng.range(lo, hi))
                        .collect();
                    let actual = metric.dist2(&point, &sample);
                    assert!(
                        bound <= actual,
                        "d={dimensions} case={case}: bound {bound}, actual {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn axis_bound_matches_brute_force_for_unwrapped_intervals() {
        let mut rng = Lcg(0x4158_4953_424f_554e);
        for _ in 0..4_000 {
            let period = rng.range(0.25, 20.0);
            let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
            let lo = rng.range(-4.0 * period, 4.0 * period);
            let width = rng.range(0.0, 1.5 * period);
            let hi = lo + width;
            let x = rng.range(-4.0 * period, 4.0 * period);
            let got = metric.axis_min_dist2(0, x, lo, hi);
            let expected = if width >= period {
                0.0
            } else {
                (-8..=8)
                    .map(|shift| {
                        let shifted = x + f64::from(shift) * period;
                        let delta = if shifted < lo {
                            lo - shifted
                        } else if shifted > hi {
                            shifted - hi
                        } else {
                            0.0
                        };
                        delta * delta
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            assert!((got - expected).abs() <= 1e-12 * (1.0 + expected));
        }
    }

    #[test]
    fn euclidean_implements_the_same_tree_metric_contract() {
        let metric = Euclidean;
        assert_eq!(metric.dist2(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
        assert_eq!(metric.axis_min_dist2(7, 4.0, 1.0, 2.0), 4.0);
        assert_eq!(
            metric.aabb_min_dist2(&[4.0, -1.0], &[1.0, 0.0], &[2.0, 3.0]),
            5.0
        );
    }

    #[test]
    fn box_bound_handles_seam_translation_and_full_period() {
        let metric = PeriodicSpace::new(&[Some(10.0)]).unwrap();
        assert_eq!(metric.axis_min_dist2(0, 0.1, 0.1, 0.3), 0.0);
        assert_eq!(metric.axis_min_dist2(0, 3.0, 12.0, 14.0), 0.0);
        assert_eq!(metric.axis_min_dist2(0, 9.5, 0.0, 1.0), 0.25);
        assert_eq!(metric.axis_min_dist2(0, 8.0, 2.0, 12.0), 0.0);
        assert!(metric.aabb_min_dist2(&[0.0], &[f64::NAN], &[1.0]).is_nan());
    }

    #[test]
    fn extreme_coordinates_stay_finite_and_match_wrapped_points() {
        let metric = PeriodicSpace::new(&[Some(10.0), None, Some(3.0)]).unwrap();
        let p = [-f64::MAX, -2.5, f64::MAX];
        let q = [f64::MAX, 1.5, -f64::MAX];

        let mut disp = [0.0; 3];
        metric.displacement(&p, &q, &mut disp).unwrap();
        let d2 = metric.dist2(&p, &q);
        let dist = metric.distance(&p, &q);
        assert!(disp.iter().all(|x| x.is_finite()), "disp={disp:?}");
        assert!(d2.is_finite(), "d2={d2}");
        assert!(dist.is_finite(), "dist={dist}");

        let mut wp = p;
        let mut wq = q;
        metric.wrap_point(&mut wp).unwrap();
        metric.wrap_point(&mut wq).unwrap();
        let mut wdisp = [0.0; 3];
        metric.displacement(&wp, &wq, &mut wdisp).unwrap();
        assert_eq!(disp, wdisp);
        assert_eq!(d2, metric.dist2(&wp, &wq));
        assert_eq!(dist, d2.sqrt());

        let mut out_many = [0.0; 2];
        metric
            .dist2_many(&p, &[q[0], q[1], q[2], p[0], p[1], p[2]], &mut out_many)
            .unwrap();
        assert_eq!(out_many, [d2, 0.0]);

        let mut out_pairs = [0.0; 2];
        metric
            .dist2_pairs(
                &[p[0], p[1], p[2], q[0], q[1], q[2]],
                &[q[0], q[1], q[2], p[0], p[1], p[2]],
                &mut out_pairs,
            )
            .unwrap();
        assert_eq!(out_pairs, [d2, d2]);

        // Tiny bounds near zero against a huge coordinate must not be
        // swallowed by raw subtraction.
        let one_d = PeriodicSpace::new(&[Some(10.0)]).unwrap();
        assert_eq!(one_d.axis_min_dist2(0, 1e20 + 16384.0, 2.5, 4.5), 0.0);
        assert_eq!(one_d.axis_min_dist2(0, f64::MAX, -1.0, 1.0), 1.0);
        // The exact image gap here is about 8.3e291, whose square rounds to
        // infinity rather than to the zero a rounded image shift would give.
        let huge_period = PeriodicSpace::new(&[Some(1e300)]).unwrap();
        assert_eq!(
            huge_period.axis_min_dist2(0, 1e308, -1e308, -1e308 + 1e295),
            f64::INFINITY
        );
        let mut huge_disp = [0.0];
        huge_period
            .displacement(&[-1e308], &[1e308], &mut huge_disp)
            .unwrap();
        assert_eq!(huge_disp[0], -8.305_139_325_152_793e291);

        // Negative coordinates much smaller than the period keep precision.
        let large = PeriodicSpace::new(&[Some(1e10)]).unwrap();
        let mut small_disp = [0.0];
        large
            .displacement(&[-1e-7], &[-2e-7], &mut small_disp)
            .unwrap();
        assert!((small_disp[0] + 1e-7).abs() <= 1e-20);
        assert!((large.axis_min_dist2(0, -2e-7, -1e-7, -0.5e-7) - 1e-14).abs() <= 1e-28);
    }

    #[test]
    fn ordinary_inputs_match_the_exact_image() {
        let mut rng = Lcg(0x4f52_4449_4e41_5259);
        for dimensions in 1..=6 {
            for case in 0..120 {
                let periods: Vec<Option<f64>> = (0..dimensions)
                    .map(|axis| {
                        if (axis + case) % 3 == 0 {
                            None
                        } else {
                            Some(rng.range(0.05, 50.0))
                        }
                    })
                    .collect();
                let metric = PeriodicSpace::new(&periods).unwrap();
                let mut p = Vec::with_capacity(dimensions);
                let mut q = Vec::with_capacity(dimensions);
                for &period in &periods {
                    let scale = period.unwrap_or(10.0);
                    p.push(rng.range(-25.0 * scale, 25.0 * scale));
                    q.push(rng.range(-25.0 * scale, 25.0 * scale));
                }
                let mut disp = vec![0.0; dimensions];
                metric.displacement(&p, &q, &mut disp).unwrap();
                let mut expected_d2 = 0.0;
                for axis in 0..dimensions {
                    let expected = exact::displacement(p[axis], q[axis], periods[axis]);
                    assert_eq!(
                        disp[axis], expected,
                        "d={dimensions} case={case} axis={axis}"
                    );
                    expected_d2 += expected * expected;
                }
                assert_eq!(
                    metric.dist2(&p, &q),
                    expected_d2,
                    "d={dimensions} case={case}"
                );
            }
        }
    }

    #[test]
    fn properties_over_extreme_magnitudes_negatives_and_mixed_axes() {
        let mut rng = Lcg(0x5052_4f50_4552_5459);
        for dimensions in 1..=8 {
            for case in 0..400 {
                let periods: Vec<Option<f64>> = (0..dimensions)
                    .map(|axis| match (axis + case) % 4 {
                        0 => None,
                        1 => Some(rng.range(0.1, 25.0)),
                        2 => Some(f64::from_bits(1 + (rng.0 >> 12) % 32)),
                        _ => Some(any_magnitude(&mut rng).abs().clamp(1e-12, 1e150)),
                    })
                    .collect();
                let metric = PeriodicSpace::new(&periods).unwrap();
                let mut p = Vec::with_capacity(dimensions);
                let mut q = Vec::with_capacity(dimensions);
                for (axis, &period) in periods.iter().enumerate() {
                    let val_p = match (case + axis) % 5 {
                        0 => f64::MAX,
                        1 => -f64::MAX,
                        2 => -rng.range(0.0, period.unwrap_or(10.0) * 1e6),
                        3 => any_magnitude(&mut rng),
                        _ => rng.range(-100.0, 100.0),
                    };
                    let val_q = match (case + axis) % 6 {
                        0 => -f64::MAX,
                        1 => f64::MAX,
                        2 => -rng.range(0.0, period.unwrap_or(10.0) * 1e6),
                        3 => any_magnitude(&mut rng),
                        4 => rng.range(-10.0, 10.0),
                        _ => -0.0,
                    };
                    // Keep non-periodic axes within finite difference square range.
                    if period.is_none() {
                        p.push(val_p.clamp(-1e100, 1e100));
                        q.push(val_q.clamp(-1e100, 1e100));
                    } else {
                        p.push(val_p);
                        q.push(val_q);
                    }
                }
                if p.iter().chain(&q).any(|x| !x.is_finite()) {
                    continue;
                }

                let mut disp = vec![0.0; dimensions];
                metric.displacement(&p, &q, &mut disp).unwrap();
                let d2 = metric.dist2(&p, &q);
                let d2_rev = metric.dist2(&q, &p);
                assert!(disp.iter().all(|x| x.is_finite()), "disp={disp:?}");
                assert!(d2.is_finite() && d2 >= 0.0, "d2={d2}");
                assert_eq!(d2, d2_rev, "symmetry");

                let mut wp = p.clone();
                let mut wq = q.clone();
                metric.wrap_point(&mut wp).unwrap();
                metric.wrap_point(&mut wq).unwrap();

                let mut wdisp = vec![0.0; dimensions];
                metric.displacement(&wp, &wq, &mut wdisp).unwrap();
                for axis in 0..dimensions {
                    let period = periods[axis];
                    assert_eq!(
                        disp[axis],
                        exact::displacement(p[axis], q[axis], period),
                        "p={:e} q={:e} period={period:?}",
                        p[axis],
                        q[axis]
                    );
                    assert_eq!(
                        wdisp[axis],
                        exact::displacement(wp[axis], wq[axis], period),
                        "wrapped p={:e} q={:e} period={period:?}",
                        wp[axis],
                        wq[axis]
                    );
                }
                let squares: f64 = disp.iter().map(|x| x * x).sum();
                assert_eq!(d2, squares);
            }
        }
    }

    #[test]
    fn box_bound_stays_finite_and_sound_for_extreme_coordinates() {
        let mut rng = Lcg(0x4558_5452_4141_4242);
        for case in 0..300 {
            let periods = [
                Some(rng.range(0.5, 20.0)),
                None,
                Some(rng.range(1e-6, 10.0)),
            ];
            let metric = PeriodicSpace::new(&periods).unwrap();
            let mut point = [0.0; 3];
            let mut lower = [0.0; 3];
            let mut upper = [0.0; 3];
            for axis in 0..3 {
                match periods[axis] {
                    None => {
                        let lo = rng.range(-100.0, 100.0);
                        lower[axis] = lo;
                        upper[axis] = lo + rng.range(0.0, 20.0);
                        point[axis] = rng.range(-200.0, 200.0);
                    }
                    Some(period) => {
                        let lo = rng.range(-4.0 * period, 4.0 * period);
                        lower[axis] = lo;
                        upper[axis] = lo + rng.range(0.0, 1.3 * period);
                        point[axis] = match (case + axis) % 4 {
                            0 => f64::MAX,
                            1 => -f64::MAX,
                            2 => -rng.range(0.0, 1e12 * period),
                            _ => any_magnitude(&mut rng),
                        };
                        if !point[axis].is_finite() {
                            point[axis] = -f64::MAX;
                        }
                    }
                }
            }
            let bound = metric.aabb_min_dist2(&point, &lower, &upper);
            assert!(bound.is_finite() && bound >= 0.0, "bound={bound}");
            for _ in 0..16 {
                let sample = [
                    rng.range(lower[0], upper[0]),
                    rng.range(lower[1], upper[1]),
                    rng.range(lower[2], upper[2]),
                ];
                let actual = metric.dist2(&point, &sample);
                assert!(
                    bound <= actual,
                    "case={case}: bound={bound}, actual={actual}"
                );
            }
        }
    }

    #[test]
    fn huge_image_counts_keep_a_representable_separation() {
        let period = 9.664_251_935_223_536e-9;
        let (a, b) = (35_602_323.555_269_06, 35_602_323.555_269_07);
        let expected = -2.213_671_338_299_708_4e-9;
        assert_eq!(wrap_difference_component(b - a, Some(period)), expected);

        let metric = PeriodicSpace::new(&[Some(period), None]).unwrap();
        let mut disp = [0.0; 2];
        metric
            .displacement(&[a, 1.0], &[b, 3.0], &mut disp)
            .unwrap();
        assert_eq!(disp, [expected, 2.0]);
        assert_eq!(
            metric.dist2(&[a, 1.0], &[b, 3.0]),
            expected * expected + 4.0
        );
        assert_eq!(metric.dist2(&[b, 0.0], &[a, 0.0]), expected * expected);
        assert_eq!(metric.axis_min_dist2(0, b, a, a), expected * expected);
        assert_eq!(metric.axis_min_dist2(0, a, b, b), expected * expected);
        assert_eq!(
            metric.aabb_min_dist2(&[b, 0.0], &[a, -1.0], &[a, 1.0]),
            expected * expected
        );

        let periods = [period];
        let slice = PeriodicSlice::new(&periods).unwrap();
        assert_eq!(slice.dist2(&[a], &[b]), expected * expected);
        assert_eq!(slice.axis_min_dist2(0, b, a, a), expected * expected);
        assert_eq!(slice.aabb_min_dist2(&[b], &[a], &[a]), expected * expected);
    }

    #[test]
    fn adjacent_coordinates_keep_their_separation_at_any_image_count() {
        let mut rng = Lcg(0x4144_4a41_4345_4e54);
        for _ in 0..20_000 {
            let period = rng.range(1e-9, 10.0);
            let images = 2f64.powf(rng.range(1.0, 60.0));
            let side = if rng.next() < 0.5 { -1.0 } else { 1.0 };
            let a = side * images * period * rng.range(1.0, 2.0);
            let b = f64::from_bits(a.to_bits() + 1);
            let raw = b - a;
            if raw.abs() >= period {
                continue;
            }
            let expected = wrap_near(raw, period);
            assert_ne!(expected, 0.0);
            let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
            let mut disp = [0.0];
            metric.displacement(&[a], &[b], &mut disp).unwrap();
            assert_eq!(disp[0], expected, "{a:e} -> {b:e} mod {period:e}");
            assert_eq!(metric.dist2(&[a], &[b]), expected * expected);
            assert_eq!(metric.axis_min_dist2(0, b, a, a), expected * expected);
        }
    }

    #[test]
    fn exact_images_at_large_magnitudes_and_tiny_periods() {
        let ten = PeriodicSpace::new(&[Some(10.0)]).unwrap();
        let two_54 = 18_014_398_509_481_984.0;
        let mut disp = [0.0];
        ten.displacement(&[two_54], &[two_54 - 4.0], &mut disp)
            .unwrap();
        assert_eq!(disp, [-4.0]);
        assert_eq!(ten.dist2(&[two_54], &[two_54 - 4.0]), 16.0);
        let edge = two_54 + 4.0;
        assert_eq!(ten.axis_min_dist2(0, two_54, edge, edge), 16.0);
        assert_eq!(ten.axis_min_dist2(0, edge, two_54, two_54), 16.0);
        assert_eq!(ten.aabb_min_dist2(&[two_54], &[edge], &[edge]), 16.0);

        let micro = PeriodicSpace::new(&[Some(1e-6)]).unwrap();
        let two_30 = 1_073_741_824.0;
        let near = 4.768_371_582_031_25e-7;
        micro
            .displacement(&[two_30], &[two_30 + 4e-7], &mut disp)
            .unwrap();
        assert_eq!(disp, [near]);
        assert_eq!(micro.dist2(&[two_30], &[two_30 + 4e-7]), near * near);

        // The exact image of 10 by 1e-6 is far below one ulp of 10.
        let far = 4.525_188_817_411_374e-16;
        micro.displacement(&[-10.0], &[0.0], &mut disp).unwrap();
        assert_eq!(disp, [far]);
        assert_eq!(micro.dist2(&[-10.0], &[0.0]), 2.047_733_383_322_495e-31);
        assert_eq!(micro.axis_min_dist2(0, 0.0, -10.0, -10.0), far * far);
        let periods = [1e-6];
        let slice = PeriodicSlice::new(&periods).unwrap();
        assert_eq!(slice.dist2(&[-10.0], &[0.0]), far * far);
        assert_eq!(slice.aabb_min_dist2(&[0.0], &[-10.0], &[-10.0]), far * far);
        assert_eq!(exact::displacement(-10.0, 0.0, Some(1e-6)), far);
    }

    #[test]
    fn rounding_the_exact_image_can_reach_half_a_period() {
        let metric = PeriodicSpace::new(&[Some(2.0)]).unwrap();
        let from = -(2f64.powi(-54) + 2f64.powi(-60));
        let to = 1.0 - f64::EPSILON / 2.0;
        assert!(exact::image(from, to, Some(2.0)).lt(exact::Big::from_f64(1.0)));
        let mut disp = [0.0];
        metric.displacement(&[from], &[to], &mut disp).unwrap();
        assert_eq!(disp, [1.0]);
        assert_eq!(exact::displacement(from, to, Some(2.0)), 1.0);
        metric.displacement(&[to], &[from], &mut disp).unwrap();
        assert_eq!(disp, [-1.0]);
    }

    /// Exact rational reference built on integers only.
    ///
    /// Every finite double is an integer multiple of `2^-1074`, so values
    /// are fixed-width two's-complement integers at that scale and squares
    /// are at scale `2^-2148`. Reduction by a period uses modular
    /// arithmetic on the integer significands, never floating point.
    mod exact {
        const LIMBS: usize = 72;
        /// Binary scale of a coordinate: one unit is `2^-1074`.
        pub const SCALE: i32 = 1074;

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct Big([u64; LIMBS]);

        #[derive(Clone, Copy)]
        pub enum Round {
            Nearest,
            TowardZero,
        }

        impl Big {
            pub const ZERO: Self = Self([0; LIMBS]);

            fn from_u128(value: u128) -> Self {
                let mut limbs = [0; LIMBS];
                limbs[0] = value as u64;
                limbs[1] = (value >> 64) as u64;
                Self(limbs)
            }

            pub fn from_f64(value: f64) -> Self {
                let (negative, significand, exponent) = decompose(value);
                let magnitude = Self::from_u128(u128::from(significand)).shl(exponent + SCALE);
                if negative {
                    magnitude.neg()
                } else {
                    magnitude
                }
            }

            fn shl(self, shift: i32) -> Self {
                let shift = u32::try_from(shift).unwrap();
                let words = (shift / 64) as usize;
                let bits = shift % 64;
                let mut out = [0; LIMBS];
                for (source, word) in out[words..].iter_mut().enumerate() {
                    *word = self.0[source] << bits;
                    if bits > 0 && source > 0 {
                        *word |= self.0[source - 1] >> (64 - bits);
                    }
                }
                Self(out)
            }

            pub fn is_negative(self) -> bool {
                self.0[LIMBS - 1] >> 63 == 1
            }

            pub fn neg(self) -> Self {
                let mut out = [0; LIMBS];
                let mut carry = 1;
                for (word, &limb) in out.iter_mut().zip(&self.0) {
                    let (value, overflow) = (!limb).overflowing_add(carry);
                    *word = value;
                    carry = u64::from(overflow);
                }
                Self(out)
            }

            pub fn add(self, other: Self) -> Self {
                let mut out = [0; LIMBS];
                let mut carry = false;
                for (word, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(&other.0)) {
                    let (value, first) = a.overflowing_add(b);
                    let (value, second) = value.overflowing_add(u64::from(carry));
                    *word = value;
                    carry = first || second;
                }
                Self(out)
            }

            pub fn sub(self, other: Self) -> Self {
                self.add(other.neg())
            }

            pub fn lt(self, other: Self) -> bool {
                self.sub(other).is_negative()
            }

            pub fn abs(self) -> Self {
                if self.is_negative() {
                    self.neg()
                } else {
                    self
                }
            }

            pub fn min(self, other: Self) -> Self {
                if other.lt(self) {
                    other
                } else {
                    self
                }
            }

            pub fn mul(self, other: Self) -> Self {
                let negative = self.is_negative() != other.is_negative();
                let (a, b) = (self.abs(), other.abs());
                let (len_a, len_b) = (a.words(), b.words());
                let mut out = [0_u64; LIMBS];
                for (i, &left) in a.0[..len_a].iter().enumerate() {
                    let mut carry = 0_u128;
                    for (slot, &right) in out[i..i + len_b].iter_mut().zip(&b.0[..len_b]) {
                        let value =
                            u128::from(left) * u128::from(right) + u128::from(*slot) + carry;
                        *slot = value as u64;
                        carry = value >> 64;
                    }
                    out[i + len_b] = carry as u64;
                }
                let product = Self(out);
                if negative {
                    product.neg()
                } else {
                    product
                }
            }

            fn words(self) -> usize {
                self.0
                    .iter()
                    .rposition(|&word| word != 0)
                    .map_or(0, |i| i + 1)
            }

            fn bit_len(self) -> i32 {
                let words = self.words();
                if words == 0 {
                    return 0;
                }
                (64 * words) as i32 - self.0[words - 1].leading_zeros() as i32
            }

            fn bit(self, index: i32) -> bool {
                index >= 0 && (self.0[(index / 64) as usize] >> (index % 64)) & 1 == 1
            }

            fn any_below(self, index: i32) -> bool {
                (0..index).any(|i| self.bit(i))
            }

            /// Value times `2^-scale`, rounded to a double.
            pub fn to_f64(self, scale: i32, round: Round) -> f64 {
                let magnitude = self.abs();
                let len = magnitude.bit_len();
                if len == 0 {
                    return 0.0;
                }
                // The lowest kept bit: 53 significant bits, but none below
                // the subnormal unit 2^-1074.
                let low = (len - 53).max(scale - SCALE).max(0);
                let mut significand = (low..len)
                    .rev()
                    .fold(0_u64, |acc, i| (acc << 1) | u64::from(magnitude.bit(i)));
                if let Round::Nearest = round {
                    let half = magnitude.bit(low - 1);
                    let sticky = magnitude.any_below(low - 1);
                    if half && (sticky || significand & 1 == 1) {
                        significand += 1;
                    }
                }
                let mut value = significand as f64;
                let mut exponent = low - scale;
                while exponent != 0 {
                    let step = exponent.clamp(-1000, 1000);
                    value *= f64::from_bits(((1023 + step) as u64) << 52);
                    exponent -= step;
                }
                if value.is_infinite() {
                    if let Round::TowardZero = round {
                        value = f64::MAX;
                    }
                }
                if self.is_negative() {
                    -value
                } else {
                    value
                }
            }
        }

        fn decompose(value: f64) -> (bool, u64, i32) {
            assert!(value.is_finite());
            let bits = value.to_bits();
            let biased = ((bits >> 52) & 0x7ff) as i32;
            let fraction = bits & ((1 << 52) - 1);
            if biased == 0 {
                (bits >> 63 == 1, fraction, -SCALE)
            } else {
                (bits >> 63 == 1, fraction | (1 << 52), biased - 1075)
            }
        }

        fn pow2_mod(mut exponent: i32, modulus: u128) -> u128 {
            let mut result = 1 % modulus;
            let mut base = 2 % modulus;
            while exponent > 0 {
                if exponent & 1 == 1 {
                    result = result * base % modulus;
                }
                base = base * base % modulus;
                exponent >>= 1;
            }
            result
        }

        /// `value mod period` in `[0, period)`.
        pub fn rem_euclid(value: f64, period: f64) -> Big {
            let (negative, significand, exponent) = decompose(value);
            let (_, period_significand, period_exponent) = decompose(period);
            let modulus = u128::from(period_significand);
            let (remainder, unit) = if exponent >= period_exponent {
                let scaled = pow2_mod(exponent - period_exponent, modulus);
                (
                    u128::from(significand) % modulus * scaled % modulus,
                    period_exponent,
                )
            } else {
                let shift = period_exponent - exponent;
                if shift >= 64 {
                    (u128::from(significand), exponent)
                } else {
                    (u128::from(significand) % (modulus << shift), exponent)
                }
            };
            let remainder = Big::from_u128(remainder).shl(unit + SCALE);
            if negative && remainder != Big::ZERO {
                Big::from_f64(period).sub(remainder)
            } else {
                remainder
            }
        }

        /// Exact minimum image of `to - from` in `[-period / 2, period / 2)`.
        pub fn image(from: f64, to: f64, period: Option<f64>) -> Big {
            let Some(period) = period else {
                return Big::from_f64(to).sub(Big::from_f64(from));
            };
            let length = Big::from_f64(period);
            let difference = rem_euclid(to, period).sub(rem_euclid(from, period));
            let twice = difference.add(difference);
            if !twice.lt(length) {
                difference.sub(length)
            } else if twice.lt(length.neg()) {
                difference.add(length)
            } else {
                difference
            }
        }

        pub fn displacement(from: f64, to: f64, period: Option<f64>) -> f64 {
            image(from, to, period).to_f64(SCALE, Round::Nearest)
        }

        /// Exact distance from `x` to `[lo, hi]` on one axis.
        pub fn gap(x: f64, lo: f64, hi: f64, period: Option<f64>) -> Big {
            assert!(lo <= hi);
            if lo <= x && x <= hi {
                return Big::ZERO;
            }
            let (x_big, lo_big, hi_big) = (Big::from_f64(x), Big::from_f64(lo), Big::from_f64(hi));
            let Some(period) = period else {
                return if x < lo {
                    lo_big.sub(x_big)
                } else {
                    x_big.sub(hi_big)
                };
            };
            let length = Big::from_f64(period);
            let width = hi_big.sub(lo_big);
            if !width.lt(length) {
                return Big::ZERO;
            }
            let mut forward = rem_euclid(x, period).sub(rem_euclid(lo, period));
            if forward.is_negative() {
                forward = forward.add(length);
            }
            if !width.lt(forward) {
                return Big::ZERO;
            }
            forward.sub(width).min(length.sub(forward))
        }

        pub fn square(value: Big) -> Big {
            value.mul(value)
        }
    }

    fn period_sample(rng: &mut Lcg, case: usize) -> f64 {
        let period = match case % 8 {
            0 => rng.range(0.05, 50.0),
            1 => 1e-6 * rng.range(0.5, 2.0),
            2 => 1e-6,
            3 => 10f64.powf(rng.range(-300.0, 300.0)),
            4 => any_magnitude(rng).abs(),
            5 => f64::from_bits(1 + (rng.0 >> 12) % 64),
            6 => f64::MAX * rng.range(0.25, 1.0),
            _ => 10.0,
        };
        if period == 0.0 || !period.is_finite() {
            1.0
        } else {
            period
        }
    }

    fn coordinate_sample(rng: &mut Lcg, period: f64, kind: usize) -> f64 {
        let value = match kind % 9 {
            0 => rng.range(-3.0, 3.0) * period,
            1 => any_magnitude(rng),
            2 => {
                let magnitude = 2f64.powi((rng.next() * 1023.0) as i32);
                if rng.next() < 0.5 {
                    -magnitude
                } else {
                    magnitude
                }
            }
            3 => rng.range(-1e6, 1e6).trunc() * period,
            4 => rng.range(-100.0, 100.0),
            5 => f64::MAX,
            6 => -f64::MAX,
            7 => {
                if rng.next() < 0.5 {
                    -0.0
                } else {
                    0.0
                }
            }
            _ => rng.range(-1.0, 1.0) * 2f64.powf(rng.range(0.0, 80.0)),
        };
        if value.is_finite() {
            value
        } else {
            0.0
        }
    }

    /// A second coordinate near `from`: a few ulps, a multiple of half the
    /// period, or unrelated.
    fn partner_sample(rng: &mut Lcg, from: f64, period: f64, kind: usize) -> f64 {
        let to = match kind % 9 {
            0 => f64::from_bits(from.to_bits().wrapping_add((rng.0 >> 60) + 1)),
            1 => f64::from_bits(from.to_bits().wrapping_sub((rng.0 >> 60) + 1)),
            2 => from + rng.range(-4.0, 4.0).round() * 0.5 * period,
            3 => from + rng.range(-1.0, 1.0) * period,
            4 => -from,
            5 => rng.range(-1.0, 1.0) * f64::MAX,
            6 => from + rng.range(-20.0, 20.0) * period,
            7 => from + rng.range(-18.0, 18.0).round() * period,
            _ => coordinate_sample(rng, period, kind / 9),
        };
        if to.is_finite() {
            to
        } else {
            from
        }
    }

    #[test]
    fn displacement_matches_an_exact_rational_oracle() {
        let mut rng = Lcg(0x4f52_4143_4c45_4449);
        let mut checked = 0_usize;
        let mut mismatches = Vec::new();
        for case in 0..60_000 {
            let period = period_sample(&mut rng, case);
            let from = coordinate_sample(&mut rng, period, case / 8);
            let to = partner_sample(&mut rng, from, period, case / 72);
            let metric = PeriodicSpace::new(&[Some(period)]).unwrap();
            for (p, q) in [(from, to), (to, from)] {
                let want = exact::displacement(p, q, Some(period));
                let mut disp = [0.0];
                metric.displacement(&[p], &[q], &mut disp).unwrap();
                let dist2 = metric.dist2(&[p], &[q]);
                checked += 1;
                if disp[0] != want || dist2 != want * want {
                    mismatches.push((p, q, period, disp[0], want));
                }
            }
        }
        assert!(checked >= 120_000);
        assert!(
            mismatches.is_empty(),
            "{} of {checked} mismatches, first {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(5)]
        );
    }

    /// A random interval for one axis: degenerate, short, near half or one
    /// period, or wider than the period.
    fn interval_sample(rng: &mut Lcg, period: f64, x: f64, kind: usize) -> (f64, f64) {
        let lo = match kind % 4 {
            0 => coordinate_sample(rng, period, kind / 4),
            1 => partner_sample(rng, x, period, kind / 4),
            2 => rng.range(-4.0, 4.0) * period,
            _ => x + rng.range(-2.0, 2.0) * period,
        };
        let width = match (kind / 4) % 7 {
            0 => 0.0,
            1 => rng.range(0.0, 0.5) * period,
            2 => 0.5 * period,
            3 => f64::from_bits(period.to_bits().saturating_sub(1 + (rng.0 >> 62))),
            4 => period,
            5 => rng.range(0.0, 1.5) * period,
            _ => any_magnitude(rng).abs(),
        };
        let lo = if lo.is_finite() { lo } else { 0.0 };
        let hi = lo + width;
        if hi.is_finite() {
            (lo, hi)
        } else {
            (lo, lo)
        }
    }

    #[test]
    fn interval_bounds_never_exceed_an_exact_rational_oracle() {
        let mut rng = Lcg(0x424f_554e_4445_5841);
        let mut axis_checked = 0_usize;
        let mut box_checked = 0_usize;
        let mut failures = Vec::new();
        for case in 0..60_000 {
            let open = case % 5 == 4;
            let period = period_sample(&mut rng, case);
            let x = coordinate_sample(&mut rng, period, case / 8);
            let (lo, hi) = interval_sample(&mut rng, period, x, case / 40);
            let periods = if open { [None] } else { [Some(period)] };
            let metric = PeriodicSpace::new(&periods).unwrap();
            let gap = exact::gap(x, lo, hi, periods[0]);
            let tight = gap.to_f64(exact::SCALE, exact::Round::TowardZero);
            let rounded = exact::square(gap).to_f64(2 * exact::SCALE, exact::Round::Nearest);
            let bound = metric.axis_min_dist2(0, x, lo, hi);
            axis_checked += 1;
            if bound != tight * tight || bound > rounded {
                failures.push((x, lo, hi, periods[0], bound, rounded));
            }
            if open {
                assert_eq!(Euclidean.axis_min_dist2(0, x, lo, hi), bound);
            } else {
                let slice_periods = [period];
                let slice = PeriodicSlice::new(&slice_periods).unwrap();
                assert_eq!(slice.axis_min_dist2(0, x, lo, hi), bound);
            }
        }

        for case in 0..10_000 {
            let dimensions = 2 + case % 5;
            let mut periods = Vec::with_capacity(dimensions);
            let mut point = Vec::with_capacity(dimensions);
            let mut lower = Vec::with_capacity(dimensions);
            let mut upper = Vec::with_capacity(dimensions);
            let mut exact_sum = exact::Big::ZERO;
            for axis in 0..dimensions {
                let period = period_sample(&mut rng, case + axis);
                let period = ((case + axis) % 4 != 3).then_some(period);
                let scale = period.unwrap_or(1.0);
                // Small coordinates keep several axes nonzero at comparable
                // scales, which is where a rounded sum can overshoot.
                let x = if case % 3 == 0 {
                    coordinate_sample(&mut rng, scale, case / 3 + axis)
                } else {
                    (rng.range(-2.0, 2.0) * scale).clamp(-f64::MAX, f64::MAX)
                };
                let (lo, hi) = interval_sample(&mut rng, scale, x, case + 7 * axis);
                exact_sum = exact_sum.add(exact::square(exact::gap(x, lo, hi, period)));
                periods.push(period);
                point.push(x);
                lower.push(lo);
                upper.push(hi);
            }
            let metric = PeriodicSpace::new(&periods).unwrap();
            let bound = metric.aabb_min_dist2(&point, &lower, &upper);
            let rounded = exact_sum.to_f64(2 * exact::SCALE, exact::Round::Nearest);
            box_checked += 1;
            if bound.is_nan() || bound > rounded {
                failures.push((point[0], lower[0], upper[0], periods[0], bound, rounded));
            }
            // A sound bound may lose a few ulps to the downward rounding.
            if rounded > 1e-280 && rounded.is_finite() {
                assert!(
                    bound >= rounded * (1.0 - 16.0 * f64::EPSILON),
                    "case={case}: bound {bound:e} far below {rounded:e}"
                );
            }
            for corner in 0..4_u32 {
                let sample: Vec<f64> = (0..dimensions)
                    .map(|axis| match (corner >> (axis % 2)) & 1 {
                        0 => lower[axis],
                        _ => upper[axis],
                    })
                    .collect();
                assert!(
                    bound <= metric.dist2(&point, &sample),
                    "case={case}: {bound:e} vs {:e} p={point:?} lo={lower:?} hi={upper:?} periods={periods:?}",
                    metric.dist2(&point, &sample)
                );
            }
            let open_sum = (0..dimensions).fold(exact::Big::ZERO, |sum, axis| {
                sum.add(exact::square(exact::gap(
                    point[axis],
                    lower[axis],
                    upper[axis],
                    None,
                )))
            });
            let euclidean = Euclidean.aabb_min_dist2(&point, &lower, &upper);
            box_checked += 1;
            if euclidean.is_nan()
                || euclidean > open_sum.to_f64(2 * exact::SCALE, exact::Round::Nearest)
            {
                failures.push((point[0], lower[0], upper[0], None, euclidean, f64::NAN));
            }
        }
        assert_eq!((axis_checked, box_checked), (60_000, 20_000));
        assert!(
            failures.is_empty(),
            "{} bounds above the exact minimum, first {:?}",
            failures.len(),
            &failures[..failures.len().min(5)]
        );
    }
}
