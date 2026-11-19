//! Periodic image pairs within a cutoff for small or skewed cells.
//!
//! A minimum-image pair list keeps one image of each partner, which
//! covers the cutoff sphere only while `2 * cutoff` is strictly below
//! every periodic face altitude ([`Cell::widths`]). When a periodic
//! face is narrower than `2 * cutoff`, a pair can have several periodic
//! images inside the cutoff and an atom can interact with its own
//! non-zero lattice translations.
//!
//! [`pairs_within_images`] enumerates every image of `n` points with
//! `0 < |d| < cutoff`, where `d = (p_i - p_j) - H S` (`p_i` is source
//! `i`, `p_j` is target `j`, and `S = [sa, sb, sc]` is the integer
//! lattice shift along `a, b, c`). With `half` set, only `i <= j` are
//! visited and self-images (`i == j`) are restricted to the positive
//! half-space (`sa > 0`, or `sa == 0 && sb > 0`, or
//! `sa == 0 && sb == 0 && sc > 0`).
//!
//! Along each periodic axis `a` with face altitude `w_a`, any image
//! with `|d| < cutoff` has fractional separation `f_a` obeying
//! `|f_a - s_a| = |n_a . d| / w_a <= |d| / w_a < cutoff / w_a`. The
//! search centres on `c_a = round(f_a)` (`|f_a - c_a| <= 1/2`), so
//! `|s_a - c_a| < cutoff / w_a + 1/2 <= ceil(cutoff / w_a)`. Thus
//! `extent_a = ceil(cutoff / w_a)` shells around `c_a` suffice for any
//! triclinic cell.

use crate::{Cell, Error};

/// Largest per-axis shell extent `ceil(cutoff / width)` accepted by
/// [`pairs_within_images`].
pub const MAX_IMAGE_EXTENT: i32 = 1024;

/// Largest product `(2 * extent_a + 1) * (2 * extent_b + 1) * (2 * extent_c + 1)`
/// of per-axis shift counts accepted by [`pairs_within_images`] (`64^3`).
pub const MAX_IMAGE_COUNT: i64 = 64 * 64 * 64;

/// One periodic image of pair `(i, j)` within the cutoff.
///
/// `d` is `(p_i - p_j) - H shift`, and `dist2` is `|d|^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePair {
    /// Source atom index (`0 <= i < n`).
    pub i: i32,
    /// Target atom index (`0 <= j < n`; `i <= j` when `half` is `true`).
    pub j: i32,
    /// Integer lattice shift `[sa, sb, sc]` subtracted from `p_i - p_j`.
    pub shift: [i32; 3],
    /// Cartesian separation `(p_i - p_j) - H shift` in cell length units.
    pub d: [f64; 3],
    /// Squared distance `d[0]^2 + d[1]^2 + d[2]^2` in squared cell length units.
    pub dist2: f64,
}

/// True when `2 * cutoff >= width` on any periodic axis of `cell`.
///
/// When this returns `false`, the single Euclidean minimum image is the
/// only image that can lie strictly inside `cutoff`.
///
/// ```
/// use minimage::{cutoff_exceeds_half_width, Cell};
///
/// # fn main() -> Result<(), minimage::Error> {
/// let wide = Cell::ortho(12.0, 12.0, 12.0)?;
/// assert!(!cutoff_exceeds_half_width(&wide, [true, true, true], 4.95));
/// let narrow = Cell::ortho(12.0, 12.0, 9.0)?;
/// assert!(cutoff_exceeds_half_width(&narrow, [true, true, true], 4.95));
/// assert!(!cutoff_exceeds_half_width(&narrow, [true, true, false], 4.95));
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn cutoff_exceeds_half_width(cell: &Cell, periodic: [bool; 3], cutoff: f64) -> bool {
    if !cutoff.is_finite() {
        return true;
    }
    let widths = cell.widths();
    for axis in 0..3 {
        if !periodic[axis] {
            continue;
        }
        if !widths[axis].is_finite() || 2.0 * cutoff >= widths[axis] {
            return true;
        }
    }
    false
}

/// Per-axis image extents `ceil(cutoff / width)` for `periodic` axes
/// (`0` on open axes), checked against [`MAX_IMAGE_EXTENT`] and
/// [`MAX_IMAGE_COUNT`].
///
/// # Errors
///
/// - [`Error::BadCutoff`] if `cutoff` is not finite or `< 0`.
/// - [`Error::BadBox`] if any periodic face width is not finite and positive.
/// - [`Error::ImageExtent`] if a periodic axis requires an extent above
///   [`MAX_IMAGE_EXTENT`].
/// - [`Error::ImageCount`] if the product of `2 * extent + 1` over the
///   three axes exceeds [`MAX_IMAGE_COUNT`].
pub fn image_extents(cell: &Cell, periodic: [bool; 3], cutoff: f64) -> Result<[i32; 3], Error> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(Error::BadCutoff);
    }
    if cutoff == 0.0 {
        return Ok([0; 3]);
    }
    let widths = cell.widths();
    let mut extent = [0i32; 3];
    for axis in 0..3 {
        if !periodic[axis] {
            continue;
        }
        let spacing = widths[axis];
        if !spacing.is_finite() || spacing <= 0.0 {
            return Err(Error::BadBox);
        }
        let shells = (cutoff / spacing).ceil();
        if !shells.is_finite() || shells < 0.0 {
            return Err(Error::BadCutoff);
        }
        if shells > f64::from(MAX_IMAGE_EXTENT) {
            return Err(Error::ImageExtent {
                axis,
                extent: shells,
                cap: MAX_IMAGE_EXTENT,
                cutoff,
                spacing,
            });
        }
        extent[axis] = shells as i32;
    }
    let mut image_count: i64 = 1;
    for &ext in &extent {
        let span = 2 * i64::from(ext) + 1;
        if span < 1 || image_count > MAX_IMAGE_COUNT / span {
            return Err(Error::ImageCount {
                extents: extent,
                cap: MAX_IMAGE_COUNT,
            });
        }
        image_count *= span;
    }
    Ok(extent)
}

#[inline(always)]
fn image_center(fractional: f64, extent: i32) -> Result<i32, Error> {
    if !fractional.is_finite() {
        return Err(Error::NonFinite);
    }
    let min_round = i64::MIN as f64;
    let max_round = i64::MAX as f64;
    if fractional <= min_round || fractional >= max_round {
        return Err(Error::ImageShiftOverflow);
    }
    let shift = fractional.round() as i64;
    let span = i64::from(extent);
    let min_idx = i64::from(i32::MIN);
    let max_idx = i64::from(i32::MAX);
    if shift < min_idx + span || shift > max_idx - span {
        return Err(Error::ImageShiftOverflow);
    }
    Ok(shift as i32)
}

/// Visit every periodic image of pairs in `xyz` strictly inside `cutoff`.
///
/// Calls `visit(ImagePair)` for each `(i, j, shift)` with
/// `0 < |d| < cutoff`, where `d = (xyz[i] - xyz[j]) - H shift`.
/// When `half` is `true`, only `i <= j` are visited, and for `i == j`
/// only shifts in the positive half-space (`sa > 0`, or
/// `sa == 0 && sb > 0`, or `sa == 0 && sb == 0 && sc > 0`) are emitted.
/// When `half` is `false`, all `i, j` in `0..xyz.len()` are visited
/// (excluding the zero self-shift `i == j && shift == [0, 0, 0]`).
///
/// Returns the total number of emitted pair images.
///
/// # Errors
///
/// - [`Error::BadCutoff`] if `cutoff` is not finite or `< 0`.
/// - [`Error::BufferSize`] if `xyz.len() > i32::MAX as usize`.
/// - [`Error::ImageExtent`] if any periodic axis requires an extent
///   above [`MAX_IMAGE_EXTENT`].
/// - [`Error::ImageCount`] if the product of `2 * extent + 1` exceeds
///   [`MAX_IMAGE_COUNT`].
/// - [`Error::NonFinite`] if a pair's fractional separation is not finite.
/// - [`Error::ImageShiftOverflow`] if the nearest integer to a
///   fractional separation plus `extent` does not fit in `i32`.
pub fn visit_pairs_within_images<F>(
    cell: &Cell,
    periodic: [bool; 3],
    xyz: &[[f64; 3]],
    cutoff: f64,
    half: bool,
    mut visit: F,
) -> Result<usize, Error>
where
    F: FnMut(ImagePair),
{
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(Error::BadCutoff);
    }
    let n = xyz.len();
    if n > i32::MAX as usize {
        return Err(Error::BufferSize);
    }
    if n == 0 || cutoff == 0.0 {
        return Ok(0);
    }
    let extent = image_extents(cell, periodic, cutoff)?;
    let [a0, a1, a2] = cell.h();
    let [inv0, inv1, inv2] = cell.hinv();
    let [ext0, ext1, ext2] = extent;
    let [pbc0, pbc1, pbc2] = periodic;
    let cut2 = cutoff * cutoff;
    let mut count = 0usize;

    for &[x, y, z] in xyz {
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(Error::NonFinite);
        }
    }

    // Precompute lattice shifts (sa * a0 + sb * a1 + sc * a2) and integer
    // shift offsets [sa, sb, sc] in [-ext0..=ext0] x [-ext1..=ext1] x [-ext2..=ext2].
    let span0 = (2 * ext0 + 1) as usize;
    let span1 = (2 * ext1 + 1) as usize;
    let span2 = (2 * ext2 + 1) as usize;
    let total_shifts = span0 * span1 * span2;
    let mut shifts_xyz_stack = [std::mem::MaybeUninit::<[f64; 3]>::uninit(); 1024];
    let mut shifts_ijk_stack = [std::mem::MaybeUninit::<[i32; 3]>::uninit(); 1024];
    let mut shifts_xyz_heap = Vec::new();
    let mut shifts_ijk_heap = Vec::new();

    if half {
        // Self-images (j == i): displacement -(sa * a0 + sb * a1 + sc * a2) is
        // identical for every atom i. Scan the positive half-space once and
        // emit each in-cutoff shift for every atom i, while populating `shifts`.
        let mut self_shifts =
            [std::mem::MaybeUninit::<(i32, i32, i32, f64, f64, f64, f64)>::uninit(); 128];
        let mut n_self = 0usize;
        let mut self_overflow = Vec::new();
        let use_stack = total_shifts <= shifts_xyz_stack.len();
        if !use_stack {
            shifts_xyz_heap.reserve(total_shifts);
            shifts_ijk_heap.reserve(total_shifts);
        }
        let mut idx = 0usize;
        for sa in -ext0..=ext0 {
            let fa = f64::from(sa);
            let sx_a = fa * a0[0];
            let sy_a = fa * a0[1];
            let sz_a = fa * a0[2];
            for sb in -ext1..=ext1 {
                let fb = f64::from(sb);
                let sx_ab = sx_a + fb * a1[0];
                let sy_ab = sy_a + fb * a1[1];
                let sz_ab = sz_a + fb * a1[2];
                for sc in -ext2..=ext2 {
                    let fc = f64::from(sc);
                    let sx = sx_ab + fc * a2[0];
                    let sy = sy_ab + fc * a2[1];
                    let sz = sz_ab + fc * a2[2];
                    if use_stack {
                        shifts_xyz_stack[idx].write([sx, sy, sz]);
                        shifts_ijk_stack[idx].write([sa, sb, sc]);
                    } else {
                        shifts_xyz_heap.push([sx, sy, sz]);
                        shifts_ijk_heap.push([sa, sb, sc]);
                    }
                    idx += 1;
                    if sa > 0 || (sa == 0 && (sb > 0 || (sb == 0 && sc > 0))) {
                        let dx = -sx;
                        let dy = -sy;
                        let dz = -sz;
                        let r2 = dx * dx + dy * dy + dz * dz;
                        if r2 > 0.0 && r2 < cut2 {
                            let entry = (sa, sb, sc, dx, dy, dz, r2);
                            if n_self < self_shifts.len() {
                                self_shifts[n_self].write(entry);
                            } else {
                                if n_self == self_shifts.len() {
                                    // SAFETY: all `self_shifts.len()` elements were written above.
                                    let init_self: &[(i32, i32, i32, f64, f64, f64, f64)] = unsafe {
                                        std::slice::from_raw_parts(
                                            self_shifts.as_ptr().cast(),
                                            self_shifts.len(),
                                        )
                                    };
                                    self_overflow.extend_from_slice(init_self);
                                }
                                self_overflow.push(entry);
                            }
                            n_self += 1;
                        }
                    }
                }
            }
        }
        // SAFETY: `idx == total_shifts` elements of `shifts_*_stack` and
        // `n_self` elements of `self_shifts` were written above.
        let (shifts_xyz, shifts_ijk): (&[[f64; 3]], &[[i32; 3]]) = if use_stack {
            unsafe {
                (
                    std::slice::from_raw_parts(shifts_xyz_stack.as_ptr().cast(), total_shifts),
                    std::slice::from_raw_parts(shifts_ijk_stack.as_ptr().cast(), total_shifts),
                )
            }
        } else {
            (shifts_xyz_heap.as_slice(), shifts_ijk_heap.as_slice())
        };
        let self_slice: &[(i32, i32, i32, f64, f64, f64, f64)] = if n_self <= self_shifts.len() {
            unsafe { std::slice::from_raw_parts(self_shifts.as_ptr().cast(), n_self) }
        } else {
            self_overflow.as_slice()
        };

        for (i, &[xi, yi, zi]) in xyz.iter().enumerate() {
            let ii = i as i32;
            for &(sa, sb, sc, dx, dy, dz, r2) in self_slice {
                count += 1;
                visit(ImagePair {
                    i: ii,
                    j: ii,
                    shift: [sa, sb, sc],
                    d: [dx, dy, dz],
                    dist2: r2,
                });
            }
            // Distinct pairs (j > i).
            for (j, &[xj, yj, zj]) in xyz.iter().enumerate().skip(i + 1) {
                let jj = j as i32;
                let dx0 = xi - xj;
                let dy0 = yi - yj;
                let dz0 = zi - zj;
                let f0 = inv0[0] * dx0 + inv1[0] * dy0 + inv2[0] * dz0;
                let f1 = inv0[1] * dx0 + inv1[1] * dy0 + inv2[1] * dz0;
                let f2 = inv0[2] * dx0 + inv1[2] * dy0 + inv2[2] * dz0;
                let c0 = if pbc0 {
                    image_center(f0, ext0)?
                } else {
                    if !f0.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    0
                };
                let c1 = if pbc1 {
                    image_center(f1, ext1)?
                } else {
                    if !f1.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    0
                };
                let c2 = if pbc2 {
                    image_center(f2, ext2)?
                } else {
                    if !f2.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    0
                };
                let fc0 = f64::from(c0);
                let fc1 = f64::from(c1);
                let fc2 = f64::from(c2);
                let rx = dx0 - (fc0 * a0[0] + fc1 * a1[0] + fc2 * a2[0]);
                let ry = dy0 - (fc0 * a0[1] + fc1 * a1[1] + fc2 * a2[1]);
                let rz = dz0 - (fc0 * a0[2] + fc1 * a1[2] + fc2 * a2[2]);
                for (k, &[sx, sy, sz]) in shifts_xyz.iter().enumerate() {
                    let dx = rx - sx;
                    let dy = ry - sy;
                    let dz = rz - sz;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 > 0.0 && r2 < cut2 {
                        let [da, db, dc] = shifts_ijk[k];
                        count += 1;
                        visit(ImagePair {
                            i: ii,
                            j: jj,
                            shift: [c0 + da, c1 + db, c2 + dc],
                            d: [dx, dy, dz],
                            dist2: r2,
                        });
                    }
                }
            }
        }
    } else {
        let use_stack = total_shifts <= shifts_xyz_stack.len();
        if !use_stack {
            shifts_xyz_heap.reserve(total_shifts);
            shifts_ijk_heap.reserve(total_shifts);
        }
        let mut idx = 0usize;
        for sa in -ext0..=ext0 {
            let fa = f64::from(sa);
            let sx_a = fa * a0[0];
            let sy_a = fa * a0[1];
            let sz_a = fa * a0[2];
            for sb in -ext1..=ext1 {
                let fb = f64::from(sb);
                let sx_ab = sx_a + fb * a1[0];
                let sy_ab = sy_a + fb * a1[1];
                let sz_ab = sz_a + fb * a1[2];
                for sc in -ext2..=ext2 {
                    let fc = f64::from(sc);
                    let sx = sx_ab + fc * a2[0];
                    let sy = sy_ab + fc * a2[1];
                    let sz = sz_ab + fc * a2[2];
                    if use_stack {
                        shifts_xyz_stack[idx].write([sx, sy, sz]);
                        shifts_ijk_stack[idx].write([sa, sb, sc]);
                    } else {
                        shifts_xyz_heap.push([sx, sy, sz]);
                        shifts_ijk_heap.push([sa, sb, sc]);
                    }
                    idx += 1;
                }
            }
        }
        // SAFETY: `idx == total_shifts` elements of `shifts_*_stack` were written above.
        let (shifts_xyz, shifts_ijk): (&[[f64; 3]], &[[i32; 3]]) = if use_stack {
            unsafe {
                (
                    std::slice::from_raw_parts(shifts_xyz_stack.as_ptr().cast(), total_shifts),
                    std::slice::from_raw_parts(shifts_ijk_stack.as_ptr().cast(), total_shifts),
                )
            }
        } else {
            (shifts_xyz_heap.as_slice(), shifts_ijk_heap.as_slice())
        };
        for (i, &[xi, yi, zi]) in xyz.iter().enumerate() {
            let ii = i as i32;
            for (j, &[xj, yj, zj]) in xyz.iter().enumerate() {
                let jj = j as i32;
                let same_atom = i == j;
                let dx0 = xi - xj;
                let dy0 = yi - yj;
                let dz0 = zi - zj;
                let f0 = inv0[0] * dx0 + inv1[0] * dy0 + inv2[0] * dz0;
                let f1 = inv0[1] * dx0 + inv1[1] * dy0 + inv2[1] * dz0;
                let f2 = inv0[2] * dx0 + inv1[2] * dy0 + inv2[2] * dz0;
                let c0 = if pbc0 {
                    image_center(f0, ext0)?
                } else {
                    if !f0.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    0
                };
                let c1 = if pbc1 {
                    image_center(f1, ext1)?
                } else {
                    if !f1.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    0
                };
                let c2 = if pbc2 {
                    image_center(f2, ext2)?
                } else {
                    if !f2.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    0
                };
                let fc0 = f64::from(c0);
                let fc1 = f64::from(c1);
                let fc2 = f64::from(c2);
                let rx = dx0 - (fc0 * a0[0] + fc1 * a1[0] + fc2 * a2[0]);
                let ry = dy0 - (fc0 * a0[1] + fc1 * a1[1] + fc2 * a2[1]);
                let rz = dz0 - (fc0 * a0[2] + fc1 * a1[2] + fc2 * a2[2]);
                for (k, &[sx, sy, sz]) in shifts_xyz.iter().enumerate() {
                    let dx = rx - sx;
                    let dy = ry - sy;
                    let dz = rz - sz;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 > 0.0 && r2 < cut2 {
                        let [da, db, dc] = shifts_ijk[k];
                        let sa = c0 + da;
                        let sb = c1 + db;
                        let sc = c2 + dc;
                        if same_atom && sa == 0 && sb == 0 && sc == 0 {
                            continue;
                        }
                        count += 1;
                        visit(ImagePair {
                            i: ii,
                            j: jj,
                            shift: [sa, sb, sc],
                            d: [dx, dy, dz],
                            dist2: r2,
                        });
                    }
                }
            }
        }
    }
    Ok(count)
}

/// Collect every periodic image of pairs in `xyz` strictly inside `cutoff`.
///
/// Clears `out` and appends each [`ImagePair`] visited by
/// [`visit_pairs_within_images`], returning the number of appended rows.
///
/// ```
/// use minimage::{pairs_within_images, Cell};
///
/// # fn main() -> Result<(), minimage::Error> {
/// let cell = Cell::ortho(4.0, 4.0, 4.0)?;
/// let xyz = [[0.4, 0.5, 0.6]];
/// let mut rows = Vec::new();
/// let n = pairs_within_images(&cell, [true, true, true], &xyz, 4.5, true, &mut rows)?;
/// // Six nearest lattice images (±4 along each axis, three in positive half-space).
/// assert_eq!(n, 3);
/// assert!(rows.iter().all(|r| r.i == 0 && r.j == 0 && (r.dist2 - 16.0).abs() < 1e-12));
/// # Ok(())
/// # }
/// ```
pub fn pairs_within_images(
    cell: &Cell,
    periodic: [bool; 3],
    xyz: &[[f64; 3]],
    cutoff: f64,
    half: bool,
    out: &mut Vec<ImagePair>,
) -> Result<usize, Error> {
    out.clear();
    visit_pairs_within_images(cell, periodic, xyz, cutoff, half, |pair| out.push(pair))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_half_images(cell: &Cell, xyz: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize, f64)> {
        let h = cell.h();
        let widths = cell.widths();
        let n = xyz.len();
        let mut home = Vec::with_capacity(n);
        for &p in xyz {
            let s = cell.fractional(p);
            home.push(cell.cartesian(s));
        }
        let reach = [
            (cutoff / widths[0]).ceil() as i32 + 2,
            (cutoff / widths[1]).ceil() as i32 + 2,
            (cutoff / widths[2]).ceil() as i32 + 2,
        ];
        let cut2 = cutoff * cutoff;
        let mut all = Vec::new();
        for i in 0..n {
            for j in i..n {
                let mut r2s = Vec::new();
                for sa in -reach[0]..=reach[0] {
                    for sb in -reach[1]..=reach[1] {
                        for sc in -reach[2]..=reach[2] {
                            let sx = f64::from(sa) * h[0][0]
                                + f64::from(sb) * h[1][0]
                                + f64::from(sc) * h[2][0];
                            let sy = f64::from(sa) * h[0][1]
                                + f64::from(sb) * h[1][1]
                                + f64::from(sc) * h[2][1];
                            let sz = f64::from(sa) * h[0][2]
                                + f64::from(sb) * h[1][2]
                                + f64::from(sc) * h[2][2];
                            let dx = home[i][0] - home[j][0] - sx;
                            let dy = home[i][1] - home[j][1] - sy;
                            let dz = home[i][2] - home[j][2] - sz;
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 > 0.0 && r2 < cut2 {
                                r2s.push(r2);
                            }
                        }
                    }
                }
                if i == j {
                    assert_eq!(r2s.len() % 2, 0);
                    let sum: f64 = r2s.iter().sum::<f64>() * 0.5;
                    all.push((i, j, r2s.len() / 2, sum));
                } else {
                    let sum: f64 = r2s.iter().sum();
                    all.push((i, j, r2s.len(), sum));
                }
            }
        }
        all.into_iter()
            .map(|(i, j, c, sum)| (i * n + j, c, sum))
            .collect()
    }

    #[test]
    fn skewed_short_cells_match_brute_force_sweep() {
        let cutoff = 4.95;
        let mut state = 20260301u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        };
        let mut uniform = |lo: f64, hi: f64| lo + (hi - lo) * next();

        let mut tested = 0;
        while tested < 120 {
            let ax = uniform(3.0, 13.0);
            let by = uniform(3.0, 13.0);
            let cz = uniform(3.0, 13.0);
            let rows = [
                [ax, 0.0, 0.0],
                [uniform(-0.5 * ax, 0.5 * ax), by, 0.0],
                [
                    uniform(-0.5 * ax, 0.5 * ax),
                    uniform(-0.5 * by, 0.5 * by),
                    cz,
                ],
            ];
            let cell = Cell::from_ase(rows).unwrap();
            let w = cell.widths();
            if w[0].min(w[1]).min(w[2]) < 2.5
                || !cutoff_exceeds_half_width(&cell, [true, true, true], cutoff)
            {
                continue;
            }
            let n = 1 + (tested % 3);
            let mut xyz = Vec::with_capacity(n);
            for _ in 0..n {
                let fa = uniform(-1.5, 2.5);
                let fb = uniform(-1.5, 2.5);
                let fc = uniform(-1.5, 2.5);
                xyz.push([
                    fa * rows[0][0] + fb * rows[1][0] + fc * rows[2][0],
                    fa * rows[0][1] + fb * rows[1][1] + fc * rows[2][1],
                    fa * rows[0][2] + fb * rows[1][2] + fc * rows[2][2],
                ]);
            }
            let want = brute_half_images(&cell, &xyz, cutoff);
            let mut got_count = vec![0usize; n * n];
            let mut got_sum = vec![0.0f64; n * n];
            visit_pairs_within_images(&cell, [true, true, true], &xyz, cutoff, true, |p| {
                let idx = p.i as usize * n + p.j as usize;
                got_count[idx] += 1;
                got_sum[idx] += p.dist2;
            })
            .unwrap();
            for (idx, want_c, want_s) in want {
                assert_eq!(got_count[idx], want_c, "cell {tested} pair {idx}");
                assert!(
                    (got_sum[idx] - want_s).abs() < 1e-8,
                    "cell {tested} pair {idx}: {} vs {want_s}",
                    got_sum[idx]
                );
            }
            tested += 1;
        }
    }

    #[test]
    fn rejects_extent_and_count_and_shift_overflow() {
        let bad_cell = Cell::ortho(1.0e-6, 4.0, 4.0).unwrap();
        let xyz = [[0.4, 0.5, 0.6]];
        let err = visit_pairs_within_images(&bad_cell, [true, true, true], &xyz, 5.0, true, |_| {})
            .unwrap_err();
        assert!(matches!(err, Error::ImageExtent { axis: 0, .. }));
        assert!(err.to_string().contains("image extent on axis 0"));

        let spacing = 5.0 / (40.0 - 0.25);
        let crowded = Cell::ortho(spacing, spacing, spacing).unwrap();
        let err = visit_pairs_within_images(&crowded, [true, true, true], &xyz, 5.0, true, |_| {})
            .unwrap_err();
        assert!(matches!(err, Error::ImageCount { .. }));
        assert!(err.to_string().contains("image count exceeds cap"));

        let cube = Cell::ortho(4.0, 4.0, 4.0).unwrap();
        for cells in [i64::from(i32::MAX) + 2, i64::from(i32::MIN) - 2] {
            let shift = (cells as f64) * 4.0;
            let far = [[0.25 + shift, 0.3, 0.4], [0.25, 0.3, 0.4]];
            let err = visit_pairs_within_images(&cube, [true, true, true], &far, 5.0, true, |_| {})
                .unwrap_err();
            assert_eq!(err, Error::ImageShiftOverflow);
            assert!(err
                .to_string()
                .contains("fractional separation exceeds lattice index range"));
        }

        let nan_pts = [[0.25, 0.3, 0.4], [f64::NAN, 0.3, 0.4]];
        let err = visit_pairs_within_images(&cube, [true, true, true], &nan_pts, 5.0, true, |_| {})
            .unwrap_err();
        assert_eq!(err, Error::NonFinite);
        assert!(err.to_string().contains("non-finite fractional separation"));

        // Near the i32 shift limit: center = i32::MAX - extent succeeds.
        let extent = (5.0f64 / 4.0).ceil() as i64;
        let center = i64::from(i32::MAX) - extent;
        let shift = (center as f64) * 4.0;
        let near_limit = [[0.75 + shift, 0.3, 0.55], [0.25, 0.3, 0.4]];
        let mut near_rows = Vec::new();
        let mut home_rows = Vec::new();
        let home_pts = [[0.75, 0.3, 0.55], [0.25, 0.3, 0.4]];
        pairs_within_images(
            &cube,
            [true, true, true],
            &near_limit,
            5.0,
            true,
            &mut near_rows,
        )
        .unwrap();
        pairs_within_images(
            &cube,
            [true, true, true],
            &home_pts,
            5.0,
            true,
            &mut home_rows,
        )
        .unwrap();
        assert_eq!(near_rows.len(), home_rows.len());
        for (a, b) in near_rows.iter().zip(&home_rows) {
            assert_eq!((a.i, a.j), (b.i, b.j));
            assert!((a.dist2 - b.dist2).abs() < 1e-9);
        }
    }
}
