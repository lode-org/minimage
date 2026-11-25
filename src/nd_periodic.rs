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

use crate::kernel::floor_fast;
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
        axis_interval_delta(x, lo, hi).map_or(f64::NAN, |delta| delta * delta)
    }

    #[inline]
    fn aabb_min_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        assert_eq!(point.len(), lower.len(), "point and AABB dimensions differ");
        assert_eq!(point.len(), upper.len(), "point and AABB dimensions differ");
        point
            .iter()
            .zip(lower)
            .zip(upper)
            .map(|((&x, &lo), &hi)| self.axis_min_dist2(0, x, lo, hi))
            .sum()
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
        self.wrap_point_row(point);
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
        self.wrap_difference_row(difference);
        Ok(())
    }

    /// Write the wrapped displacement `q - p` to `out`.
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
            out[axis] = wrap_difference_component(q[axis] - p[axis], self.periods[axis]);
        }
        Ok(())
    }

    /// Squared minimum-image distance between two points.
    ///
    /// This is the allocation-free hot path. It panics if either point
    /// does not have [`dimensions()`](Self::dimensions) entries.
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
            self.wrap_point_row(row);
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
            self.wrap_difference_row(row);
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
            *value = self.dist2_rows_unchecked(point, row);
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
            *value = self.dist2_rows_unchecked(p, q);
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
    /// Summing the exact one-axis minima gives the exact product-metric
    /// distance to the box and therefore a sound tree-pruning bound.
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

    #[inline]
    fn wrap_point_row(&self, point: &mut [f64]) {
        for (value, period) in point.iter_mut().zip(self.periods.iter().copied()) {
            if let Some(length) = period {
                *value = wrap_point_component(*value, length);
            }
        }
    }

    #[inline]
    fn wrap_difference_row(&self, difference: &mut [f64]) {
        for (value, period) in difference.iter_mut().zip(self.periods.iter().copied()) {
            *value = wrap_difference_component(*value, period);
        }
    }

    #[inline]
    fn dist2_rows_unchecked(&self, p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q)
            .zip(self.periods.iter().copied())
            .map(|((&a, &b), period)| {
                let delta = wrap_difference_component(b - a, period);
                delta * delta
            })
            .sum()
    }
}

impl Metric for PeriodicSpace {
    #[inline]
    fn dist2(&self, p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), self.dimensions(), "point dimension mismatch");
        assert_eq!(q.len(), self.dimensions(), "point dimension mismatch");
        self.dist2_rows_unchecked(p, q)
    }

    #[inline]
    fn axis_min_dist2(&self, axis: usize, x: f64, lo: f64, hi: f64) -> f64 {
        let period = self.periods[axis];
        let Some(open_delta) = axis_interval_delta(x, lo, hi) else {
            return f64::NAN;
        };
        let Some(period) = period else {
            return open_delta * open_delta;
        };
        let width = hi - lo;
        if width >= period {
            return 0.0;
        }
        let center = lo + 0.5 * width;
        let from_center = wrap_difference_component(x - center, Some(period)).abs();
        let delta = (from_center - 0.5 * width).max(0.0);
        delta * delta
    }

    #[inline]
    fn aabb_min_dist2(&self, point: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        assert_eq!(point.len(), self.dimensions(), "point dimension mismatch");
        assert_eq!(lower.len(), self.dimensions(), "AABB dimension mismatch");
        assert_eq!(upper.len(), self.dimensions(), "AABB dimension mismatch");
        (0..self.dimensions())
            .map(|axis| self.axis_min_dist2(axis, point[axis], lower[axis], upper[axis]))
            .sum()
    }
}

#[inline]
fn axis_interval_delta(x: f64, lo: f64, hi: f64) -> Option<f64> {
    if !x.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
        None
    } else if x < lo {
        Some(lo - x)
    } else if x > hi {
        Some(x - hi)
    } else {
        Some(0.0)
    }
}

#[inline]
fn wrap_point_component(value: f64, period: f64) -> f64 {
    value - period * floor_fast(value / period)
}

#[inline]
fn wrap_difference_component(value: f64, period: Option<f64>) -> f64 {
    let Some(period) = period else {
        return value;
    };
    value - period * floor_fast(value / period + 0.5)
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
                        bound <= actual + 1e-12 * (1.0 + actual),
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
        assert_eq!(metric.axis_min_dist2(0, 9.5, 0.0, 1.0), 0.25);
        assert_eq!(metric.axis_min_dist2(0, 8.0, 2.0, 12.0), 0.0);
        assert!(metric.aabb_min_dist2(&[0.0], &[f64::NAN], &[1.0]).is_nan());
    }
}
