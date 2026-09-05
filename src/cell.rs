//! Periodic cell: three lattice vectors and a minimum-image convention.
//!
//! Orthorhombic boxes are the diagonal case ([`Cell::ortho`]). Triclinic
//! (and any parallelepiped) use the same fractional wrap. Lattice
//! vectors are stored as the columns of H, so `r = H s`. The C ABI,
//! ASE, vesin, and CON pass rows `(a, b, c)`; constructors accept either.
//!
//! [`Cell::dist2`] is the per-pair minimum image. After a fold into the
//! primary cell, pair distances in a linked-cell walk are
//! [`Cell::dist2_shifted`] plus [`Cell::lattice_shift`], not a per-pair
//! wrap. [`Cell::is_ortho`] is the cheap path: three independent wraps
//! and a scaled-diagonal shift, skipping the two 3x3 matvecs.

use crate::Error;

/// Periodic parallelepiped: columns of H, origin, and an ortho flag.
///
/// ```
/// # use minimage::Cell;
/// # fn main() -> Result<(), minimage::Error> {
/// let cell = Cell::ortho(10.0, 11.0, 12.0)?;
/// assert!(cell.is_ortho());
/// let sheared = Cell::from_vectors(
///     [10.0, 0.0, 0.0],
///     [5.0, 8.66, 0.0],
///     [0.0, 0.0, 10.0],
///     [0.0, 0.0, 0.0],
/// )?;
/// assert!(!sheared.is_ortho());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    /// Columns of H: `h[0]` is lattice vector a.
    h: [[f64; 3]; 3],
    /// Inverse of H. Fractional coordinates are `s = Hinv (r - origin)`.
    hinv: [[f64; 3]; 3],
    origin: [f64; 3],
    /// Perpendicular widths |a · n_a| etc.
    widths: [f64; 3],
    /// Axis-aligned diagonal box: MIC is three independent wraps.
    ortho: bool,
}

impl Cell {
    /// Diagonal box with origin at zero.
    ///
    /// Sets [`Self::is_ortho`] so [`Self::dist2`] and
    /// [`Self::lattice_shift`] skip the two 3x3 matvecs.
    pub fn ortho(lx: f64, ly: f64, lz: f64) -> Result<Self, Error> {
        Self::from_vectors(
            [lx, 0.0, 0.0],
            [0.0, ly, 0.0],
            [0.0, 0.0, lz],
            [0.0, 0.0, 0.0],
        )
    }

    /// Diagonal box with an explicit dump-cell origin.
    pub fn ortho_origin(lx: f64, ly: f64, lz: f64, origin: [f64; 3]) -> Result<Self, Error> {
        Self::from_vectors([lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz], origin)
    }

    /// Parallelepiped from lattice vectors `a`, `b`, `c` and an origin.
    ///
    /// Vectors are the columns of H. A non-diagonal H clears
    /// [`Self::is_ortho`].
    pub fn from_vectors(
        a: [f64; 3],
        b: [f64; 3],
        c: [f64; 3],
        origin: [f64; 3],
    ) -> Result<Self, Error> {
        let h = [a, b, c];
        let (hinv, det) = invert_columns(h).ok_or(Error::BadBox)?;
        if !det.is_finite() || det.abs() < 1e-18 {
            return Err(Error::BadBox);
        }
        let bc = cross(b, c);
        let ca = cross(c, a);
        let ab = cross(a, b);
        let wa = det.abs() / norm(bc);
        let wb = det.abs() / norm(ca);
        let wc = det.abs() / norm(ab);
        if !(wa > 0.0 && wb > 0.0 && wc > 0.0) {
            return Err(Error::BadBox);
        }
        Ok(Self {
            h,
            hinv,
            origin,
            widths: [wa, wb, wc],
            ortho: is_axis_aligned(h),
        })
    }

    /// ASE-style 3x3 cell: rows are lattice vectors a, b, c.
    ///
    /// Origin is zero. Matches `atoms.cell` in ASE.
    pub fn from_ase(rows: [[f64; 3]; 3]) -> Result<Self, Error> {
        Self::from_vectors(rows[0], rows[1], rows[2], [0.0, 0.0, 0.0])
    }

    /// CON lattice: rows are lattice vectors a, b, c, origin zero.
    ///
    /// Same storage as readcon `lattice_vectors` and an eOn CON
    /// header that already holds the 3x3 basis.
    pub fn from_con(rows: [[f64; 3]; 3]) -> Result<Self, Error> {
        Self::from_ase(rows)
    }

    /// CON / crystallographic lengths and angles (degrees).
    ///
    /// `alpha` is the angle at the origin between b and c, `beta`
    /// between a and c, `gamma` between a and b. The basis is the
    /// standard reduced triclinic frame used by ASE and eOn.
    pub fn from_con_box(boxl: [f64; 3], angles_deg: [f64; 3]) -> Result<Self, Error> {
        let [lx, ly, lz] = boxl;
        if !(lx > 0.0 && ly > 0.0 && lz > 0.0) {
            return Err(Error::BadBox);
        }
        let alpha = angles_deg[0].to_radians();
        let beta = angles_deg[1].to_radians();
        let gamma = angles_deg[2].to_radians();
        let cos_a = alpha.cos();
        let cos_b = beta.cos();
        let cos_g = gamma.cos();
        let sin_g = gamma.sin();
        if sin_g.abs() < 1e-18 {
            return Err(Error::BadBox);
        }
        let a = [lx, 0.0, 0.0];
        let b = [ly * cos_g, ly * sin_g, 0.0];
        let cx = lz * cos_b;
        let cy = lz * (cos_a - cos_b * cos_g) / sin_g;
        let cz2 = lz * lz - cx * cx - cy * cy;
        if cz2 <= 0.0 {
            return Err(Error::BadBox);
        }
        Self::from_vectors(a, b, [cx, cy, cz2.sqrt()], [0.0, 0.0, 0.0])
    }

    /// vesin box: rows are lattice vectors a, b, c, origin zero.
    pub fn from_vesin(rows: [[f64; 3]; 3]) -> Result<Self, Error> {
        Self::from_ase(rows)
    }

    /// LAMMPS restricted-triclinic box from `xlo xhi ylo yhi zlo zhi`
    /// and tilt factors `xy, xz, yz`.
    ///
    /// Recovers the same H and origin as a dump ITEM BOX BOUNDS line
    /// whose bound lo/hi already include the tilt padding.
    #[allow(clippy::too_many_arguments)]
    pub fn from_lammps(
        xlo: f64,
        xhi: f64,
        ylo: f64,
        yhi: f64,
        zlo: f64,
        zhi: f64,
        xy: f64,
        xz: f64,
        yz: f64,
    ) -> Result<Self, Error> {
        Self::from_lammps_bounds(xhi - xlo, yhi - ylo, zhi - zlo, xy, xz, yz, xlo, ylo, zlo)
    }

    /// LAMMPS dump bound spans plus tilts and bound lo.
    ///
    /// `xspan, yspan, zspan` are `xhi_bound - xlo_bound` etc.
    /// `xy, xz, yz` are the tilt factors. `xlo_b, ylo_b, zlo_b` are
    /// the bound lo. Inverse of
    /// `xlo_bound = xlo + min(0, xy, xz, xy+xz)`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_lammps_bounds(
        xspan: f64,
        yspan: f64,
        zspan: f64,
        xy: f64,
        xz: f64,
        yz: f64,
        xlo_b: f64,
        ylo_b: f64,
        zlo_b: f64,
    ) -> Result<Self, Error> {
        let (h, origin) = dump_bounds_to_h(xspan, yspan, zspan, xy, xz, yz, xlo_b, ylo_b, zlo_b);
        Self::from_vectors(h[0], h[1], h[2], origin)
    }

    /// True when H is diagonal. Distances and lattice shifts then skip
    /// the two 3x3 matvecs and use three independent wraps.
    pub fn is_ortho(&self) -> bool {
        self.ortho
    }

    /// Lattice vector a (first column of H).
    pub fn a(&self) -> [f64; 3] {
        self.h[0]
    }

    /// Lattice vector b.
    pub fn b(&self) -> [f64; 3] {
        self.h[1]
    }

    /// Lattice vector c.
    pub fn c(&self) -> [f64; 3] {
        self.h[2]
    }

    /// Dump-cell origin.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Perpendicular widths of the three faces.
    pub fn widths(&self) -> [f64; 3] {
        self.widths
    }

    /// Columns of H.
    pub fn h(&self) -> [[f64; 3]; 3] {
        self.h
    }

    /// Inverse of H, stored by columns.
    pub fn hinv(&self) -> [[f64; 3]; 3] {
        self.hinv
    }

    /// Fractional coordinates in `[0, 1)`.
    ///
    /// Orthorhombic boxes divide by the three widths. The general path
    /// is `s = Hinv (r - origin)`, then wrap.
    #[inline]
    pub fn fractional(&self, r: [f64; 3]) -> [f64; 3] {
        if self.ortho {
            return [
                wrap01((r[0] - self.origin[0]) / self.widths[0]),
                wrap01((r[1] - self.origin[1]) / self.widths[1]),
                wrap01((r[2] - self.origin[2]) / self.widths[2]),
            ];
        }
        let d = [
            r[0] - self.origin[0],
            r[1] - self.origin[1],
            r[2] - self.origin[2],
        ];
        let mut s = mul(self.hinv, d);
        for e in &mut s {
            *e = wrap01(*e);
        }
        s
    }

    /// Cartesian from fractional: `r = H s + origin`.
    #[inline]
    pub fn cartesian(&self, s: [f64; 3]) -> [f64; 3] {
        let r = mul(self.h, s);
        [
            r[0] + self.origin[0],
            r[1] + self.origin[1],
            r[2] + self.origin[2],
        ]
    }

    /// Cartesian translation by integer lattice counts `(na, nb, nc)`.
    #[inline]
    pub fn lattice_shift(&self, na: i32, nb: i32, nc: i32) -> [f64; 3] {
        if self.ortho {
            [
                f64::from(na) * self.widths[0],
                f64::from(nb) * self.widths[1],
                f64::from(nc) * self.widths[2],
            ]
        } else {
            let a = self.h[0];
            let b = self.h[1];
            let c = self.h[2];
            let fa = f64::from(na);
            let fb = f64::from(nb);
            let fc = f64::from(nc);
            [
                fa * a[0] + fb * b[0] + fc * c[0],
                fa * a[1] + fb * b[1] + fc * c[1],
                fa * a[2] + fb * b[2] + fc * c[2],
            ]
        }
    }

    /// Squared Cartesian distance after applying a lattice shift to `q`.
    ///
    /// This is not a minimum-image wrap. The caller supplies the image
    /// via [`Self::lattice_shift`].
    #[inline]
    pub fn dist2_shifted(&self, p: [f64; 3], q: [f64; 3], shift: [f64; 3]) -> f64 {
        let dx = q[0] + shift[0] - p[0];
        let dy = q[1] + shift[1] - p[1];
        let dz = q[2] + shift[2] - p[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Minimum-image displacement from `p` to `q`: wrap(`q - p`).
    ///
    /// Orthorhombic boxes wrap each axis independently. The general path
    /// is `ds = wrap(Hinv (q - p))`, then `dr = H ds`.
    #[inline]
    pub fn displacement(&self, p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        if self.ortho {
            return displacement_ortho(self.widths, p, q);
        }
        displacement_general(self.h, self.hinv, p, q)
    }

    /// Squared minimum-image distance.
    #[inline]
    pub fn dist2(&self, p: [f64; 3], q: [f64; 3]) -> f64 {
        let dr = self.displacement(p, q);
        dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
    }

    /// Euclidean MIC by checking the 27 nearest lattice images.
    ///
    /// [`Self::displacement`] wraps in fractional coordinates. That is
    /// the LAMMPS / eOn / ASE convention and is the Cartesian nearest
    /// image on a reduced (restricted-triclinic) cell. A highly skewed
    /// H can make a neighbouring image shorter in Cartesian space.
    /// This walk is the check those codes omit.
    pub fn displacement_cartesian(&self, p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        let dp = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let mut best = dp;
        let mut best2 = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
        for na in -1_i32..=1 {
            for nb in -1_i32..=1 {
                for nc in -1_i32..=1 {
                    if na == 0 && nb == 0 && nc == 0 {
                        continue;
                    }
                    let s = self.lattice_shift(na, nb, nc);
                    let t = [dp[0] + s[0], dp[1] + s[1], dp[2] + s[2]];
                    let t2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                    if t2 < best2 {
                        best2 = t2;
                        best = t;
                    }
                }
            }
        }
        best
    }

    /// True when a is along x and b lies in the xy plane.
    ///
    /// This is the GROMACS / LAMMPS / HOOMD restricted-triclinic frame.
    /// Those engines refuse or rotate any other orientation before wrap.
    pub fn is_restricted(&self) -> bool {
        let a = self.h[0];
        let b = self.h[1];
        let scale = (norm(a) + norm(b) + norm(self.h[2])).max(1.0);
        let tol = 1e-10 * scale;
        a[1].abs() <= tol && a[2].abs() <= tol && b[2].abs() <= tol
    }

    /// True when the cell is restricted and the tilts sit inside the
    /// GROMACS `correct_box` / LAMMPS half-box limits
    /// `|xy| <= lx/2`, `|xz| <= lx/2`, `|yz| <= ly/2`.
    ///
    /// On that domain the fractional wrap is the Euclidean MIC, which
    /// is why LAMMPS `Domain::minimum_image` and HOOMD `BoxDim::minImage`
    /// never search extra images.
    pub fn tilts_reduced(&self) -> bool {
        if !self.is_restricted() {
            return false;
        }
        let lx = self.h[0][0].abs();
        let ly = self.h[1][1].abs();
        let xy = self.h[1][0];
        let xz = self.h[2][0];
        let yz = self.h[2][1];
        let tol = 1e-12 * (lx + ly).max(1.0);
        xy.abs() <= 0.5 * lx + tol && xz.abs() <= 0.5 * lx + tol && yz.abs() <= 0.5 * ly + tol
    }

    /// Unimodular tilt reduction: GROMACS `correct_box`.
    ///
    /// Subtracts integer lattice vectors so `|xy|`, `|xz|`, `|yz|` sit
    /// inside half the corresponding edge. Same Cartesian lattice, new
    /// basis. No-op when [`Self::tilts_reduced`] is already true.
    /// A general orientation is rotated into the restricted frame first
    /// (LAMMPS `define_general_triclinic`).
    pub fn reduce_tilts(&self) -> Result<Self, Error> {
        let (cell, _rot) = self.restricted_frame()?;
        if cell.tilts_reduced() {
            return Ok(cell);
        }
        let a = cell.h[0];
        let mut b = cell.h[1];
        let mut c = cell.h[2];
        correct_box_elem(&mut c, b, 1);
        correct_box_elem(&mut c, a, 0);
        correct_box_elem(&mut b, a, 0);
        Self::from_vectors(a, b, c, cell.origin)
    }

    /// Rotate a general parallelepiped onto the restricted-triclinic
    /// frame (LAMMPS general-to-restricted). Identity when already
    /// restricted.
    pub fn to_restricted(&self) -> Result<Self, Error> {
        Ok(self.restricted_frame()?.0)
    }

    /// Euclidean MIC for displacements that may be long.
    ///
    /// Smith, CCP5 Newsletter 1989: the fractional wrap is the
    /// nearest image when its length is below half the shortest edge.
    /// That is the GROMACS/LAMMPS/HOOMD regime (cutoffs). Linkcell
    /// k-NN has no cutoff; a hex-prism body diagonal is a fractional
    /// wrap that is longer than a neighbouring image.
    ///
    /// When the Smith test fails, the basis is Minkowski-reduced
    /// (Nguyen–Stehlé, ACM Trans. Algorithms 2009,
    /// 10.1145/1597036.1597050) and the 27-image in that basis is
    /// the nearest lattice vector. [`Self::displacement`] stays the
    /// engine fractional wrap on the caller's H.
    pub fn displacement_euclidean(&self, p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        let frac = self.displacement(p, q);
        let f2 = frac[0] * frac[0] + frac[1] * frac[1] + frac[2] * frac[2];
        let w = self.widths();
        let half_min = 0.5 * w[0].min(w[1]).min(w[2]);
        if f2.sqrt() + 1e-12 < half_min {
            return frac;
        }
        let reduced = match crate::minkowski::minkowski_reduce(self) {
            Ok(c) => c,
            Err(_) => return self.displacement_cartesian(p, q),
        };
        reduced.displacement_cartesian(p, q)
    }

    /// Squared Euclidean MIC distance.
    pub fn dist2_euclidean(&self, p: [f64; 3], q: [f64; 3]) -> f64 {
        let dr = self.displacement_euclidean(p, q);
        dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
    }

    fn restricted_frame(&self) -> Result<(Self, [[f64; 3]; 3]), Error> {
        if self.is_restricted() {
            return Ok((*self, IDENTITY));
        }
        let a = self.h[0];
        let b = self.h[1];
        let c = self.h[2];
        let al = norm(a);
        if al < 1e-18 {
            return Err(Error::BadBox);
        }
        let a_hat = [a[0] / al, a[1] / al, a[2] / al];
        let b_par = dot(b, a_hat);
        let b_perp = [
            b[0] - b_par * a_hat[0],
            b[1] - b_par * a_hat[1],
            b[2] - b_par * a_hat[2],
        ];
        let bl = norm(b_perp);
        if bl < 1e-18 {
            return Err(Error::BadBox);
        }
        let n = [b_perp[0] / bl, b_perp[1] / bl, b_perp[2] / bl];
        let k = cross(a_hat, n);
        let rot = [a_hat, n, k];
        let a_new = [al, 0.0, 0.0];
        let b_new = [b_par, bl, 0.0];
        let c_new = [dot(c, a_hat), dot(c, n), dot(c, k)];
        let cell = Self::from_vectors(a_new, b_new, c_new, mul_rows(rot, self.origin))?;
        Ok((cell, rot))
    }

    /// True when fractional wrap matches the 27-image check on a grid
    /// of short Cartesian probes (the cutoff-scale displacements MD
    /// engines actually wrap).
    ///
    /// A hex prism at the tilt limit can still make a body-diagonal
    /// fractional wrap longer than a neighbouring image. Production
    /// codes ignore that because the cutoff is well below half the
    /// long diagonal.
    pub fn fractional_matches_cartesian(&self) -> bool {
        let samples = [
            [0.2, 0.0, 0.0],
            [0.49, 0.49, 0.49],
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.8, 0.8, 0.2],
        ];
        for p in samples {
            let q = [0.0, 0.0, 0.0];
            let a = self.displacement(p, q);
            let b = self.displacement_cartesian(p, q);
            let d2 = (a[0] - b[0]) * (a[0] - b[0])
                + (a[1] - b[1]) * (a[1] - b[1])
                + (a[2] - b[2]) * (a[2] - b[2]);
            if d2 > 1e-20 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod cartesian_tests {
    use super::Cell;

    #[test]
    fn ortho_fractional_is_cartesian() {
        let c = Cell::ortho(10.0, 11.0, 12.0).unwrap();
        assert!(c.fractional_matches_cartesian());
    }

    #[test]
    fn restricted_shear_fractional_is_cartesian() {
        let c = Cell::from_vectors(
            [10.0, 0.0, 0.0],
            [5.0, 8.660254037844386, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(c.fractional_matches_cartesian());
    }

    #[test]
    fn extreme_skew_cartesian_can_be_shorter() {
        let c = Cell::from_vectors(
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        let p = [0.0, 0.0, 0.0];
        let q = [0.95, 0.05, 0.0];
        let frac = c.displacement(p, q);
        let cart = c.displacement_cartesian(p, q);
        let f2 = frac[0] * frac[0] + frac[1] * frac[1] + frac[2] * frac[2];
        let c2 = cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2];
        assert!(c2 <= f2 + 1e-15);
        assert!(c2 < 0.01);
    }

    #[test]
    fn extreme_skew_lattice_point_needs_tilt_reduce() {
        let c = Cell::from_vectors(
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(c.is_restricted());
        assert!(!c.tilts_reduced());
        let p = [0.0, 0.0, 0.0];
        let q = [0.02, -0.02, 0.0];
        let cart = c.displacement_cartesian(p, q);
        let euc = c.displacement_euclidean(p, q);
        let c2 = cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2];
        let e2 = euc[0] * euc[0] + euc[1] * euc[1] + euc[2] * euc[2];
        assert!(c2 > 1e-8, "27-image from the unreduced basis is not 0");
        assert!(e2 < 1e-24, "tilt-reduced wrap sends the lattice point to 0");
        let red = c.reduce_tilts().unwrap();
        assert!(red.tilts_reduced());
        assert!(red.h()[1][0].abs() <= 0.5 * red.h()[0][0].abs() + 1e-12);
    }

    #[test]
    fn hex_body_diagonal_fractional_is_not_nearest() {
        let c = Cell::from_vectors(
            [10.0, 0.0, 0.0],
            [5.0, 8.660254037844386, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(c.tilts_reduced());
        let p = c.cartesian([0.49, 0.49, 0.49]);
        let q = c.cartesian([0.0, 0.0, 0.0]);
        let frac = c.displacement(p, q);
        let euc = c.displacement_euclidean(p, q);
        let f2 = frac[0] * frac[0] + frac[1] * frac[1] + frac[2] * frac[2];
        let e2 = euc[0] * euc[0] + euc[1] * euc[1] + euc[2] * euc[2];
        assert!(
            e2 + 1e-8 < f2,
            "Nguyen-Stehle 27-image in the Minkowski basis is shorter than the fractional wrap"
        );
    }

    #[test]
    fn restricted_hex_is_already_reduced() {
        let c = Cell::from_vectors(
            [10.0, 0.0, 0.0],
            [5.0, 8.660254037844386, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(c.is_restricted());
        assert!(c.tilts_reduced());
        let p = [0.2, 0.1, 1.0];
        let q = [9.7, 0.1, 1.0];
        let a = c.displacement(p, q);
        let b = c.displacement_euclidean(p, q);
        assert!((a[0] - b[0]).abs() < 1e-15);
        assert!((a[1] - b[1]).abs() < 1e-15);
        assert!((a[2] - b[2]).abs() < 1e-15);
    }
}

/// Recover restricted-triclinic H (columns a, b, c) and origin from a
/// LAMMPS dump bound box.
#[allow(clippy::too_many_arguments)]
pub fn dump_bounds_to_h(
    xspan: f64,
    yspan: f64,
    zspan: f64,
    xy: f64,
    xz: f64,
    yz: f64,
    xlo_b: f64,
    ylo_b: f64,
    zlo_b: f64,
) -> ([[f64; 3]; 3], [f64; 3]) {
    let xmin = 0.0_f64.min(xy).min(xz).min(xy + xz);
    let xmax = 0.0_f64.max(xy).max(xz).max(xy + xz);
    let ymin = 0.0_f64.min(yz);
    let ymax = 0.0_f64.max(yz);
    let lx = xspan - xmax + xmin;
    let ly = yspan - ymax + ymin;
    let lz = zspan;
    let a = [lx, 0.0, 0.0];
    let b = [xy, ly, 0.0];
    let c = [xz, yz, lz];
    let origin = [xlo_b - xmin, ylo_b - ymin, zlo_b];
    ([a, b, c], origin)
}

#[inline]
fn wrap01(mut s: f64) -> f64 {
    s -= s.floor();
    if s >= 1.0 {
        0.0
    } else {
        s
    }
}

/// Orthorhombic signed wrap into `[-L/2, L/2)`.
///
/// Keeps `-L/2` and maps `+L/2` onto `-L/2`, matching the dump
/// `relDist` half-box test. Squared distance agrees with
/// `abs` then `round`.
#[inline]
fn wrap_half(d: f64, length: f64) -> f64 {
    let half = 0.5 * length;
    let mut w = d;
    if w < -half {
        w += length;
    }
    if w >= half {
        w -= length;
    }
    w
}

#[inline]
fn displacement_ortho(l: [f64; 3], p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
    [
        wrap_half(q[0] - p[0], l[0]),
        wrap_half(q[1] - p[1], l[1]),
        wrap_half(q[2] - p[2], l[2]),
    ]
}

#[inline]
fn displacement_general(
    h: [[f64; 3]; 3],
    hinv: [[f64; 3]; 3],
    p: [f64; 3],
    q: [f64; 3],
) -> [f64; 3] {
    let dp = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let mut ds = mul(hinv, dp);
    for e in &mut ds {
        *e -= e.round();
    }
    mul(h, ds)
}

fn is_axis_aligned(h: [[f64; 3]; 3]) -> bool {
    let scale = (norm(h[0]) + norm(h[1]) + norm(h[2])).max(1.0);
    let tol = 1e-12 * scale;
    h[0][1].abs() <= tol
        && h[0][2].abs() <= tol
        && h[1][0].abs() <= tol
        && h[1][2].abs() <= tol
        && h[2][0].abs() <= tol
        && h[2][1].abs() <= tol
}

const IDENTITY: [[f64; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// GROMACS `correct_box_elem`: subtract integer copies of `edge` from
/// `vec` until component `d` sits inside half of `edge[d]`.
fn correct_box_elem(vec: &mut [f64; 3], edge: [f64; 3], d: usize) {
    const MARGIN: f64 = 1.001;
    const MAX_SHIFT: i32 = 20;
    let half = 0.5 * edge[d];
    if half.abs() < 1e-18 {
        return;
    }
    let mut n = 0;
    while vec[d] > MARGIN * half && n < MAX_SHIFT {
        vec[0] -= edge[0];
        vec[1] -= edge[1];
        vec[2] -= edge[2];
        n += 1;
    }
    while vec[d] < -MARGIN * half && n < MAX_SHIFT {
        vec[0] += edge[0];
        vec[1] += edge[1];
        vec[2] += edge[2];
        n += 1;
    }
}

#[inline]
fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

#[inline]
fn mul_rows(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
}

/// H is stored by columns. `mul(h, s)` is H s.
#[inline]
pub(crate) fn mul(h: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        h[0][0] * v[0] + h[1][0] * v[1] + h[2][0] * v[2],
        h[0][1] * v[0] + h[1][1] * v[1] + h[2][1] * v[2],
        h[0][2] * v[0] + h[1][2] * v[1] + h[2][2] * v[2],
    ]
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn invert_columns(h: [[f64; 3]; 3]) -> Option<([[f64; 3]; 3], f64)> {
    let a = h[0];
    let b = h[1];
    let c = h[2];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let invdet = 1.0 / det;
    let inv = [
        [
            (b[1] * c[2] - b[2] * c[1]) * invdet,
            (a[2] * c[1] - a[1] * c[2]) * invdet,
            (a[1] * b[2] - a[2] * b[1]) * invdet,
        ],
        [
            (b[2] * c[0] - b[0] * c[2]) * invdet,
            (a[0] * c[2] - a[2] * c[0]) * invdet,
            (a[2] * b[0] - a[0] * b[2]) * invdet,
        ],
        [
            (b[0] * c[1] - b[1] * c[0]) * invdet,
            (a[1] * c[0] - a[0] * c[1]) * invdet,
            (a[0] * b[1] - a[1] * b[0]) * invdet,
        ],
    ];
    Some((inv, det))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_flag_and_dist2_match_general() {
        let b = Cell::ortho(10.0, 11.0, 12.0).unwrap();
        assert!(b.is_ortho());
        let p = [0.2, 1.0, 11.5];
        let q = [9.7, 10.8, 0.4];
        let fast = b.dist2(p, q);
        let slow = {
            let dr = displacement_general(b.h, b.hinv, p, q);
            dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
        };
        assert!((fast - slow).abs() <= 1e-12 * (1.0 + fast.abs()));
        let left = [0.2, 0.0, 0.0];
        let right = [9.4, 0.0, 0.0];
        let mic = b.dist2(left, right);
        let via = b.dist2_shifted(left, right, b.lattice_shift(-1, 0, 0));
        assert!((via - mic).abs() <= 1e-12 * (1.0 + mic.abs()));
        assert!((mic - 0.64).abs() <= 1e-12);
    }

    #[test]
    fn lammps_sheared_a_image_is_quarter() {
        let cell =
            Cell::from_lammps_bounds(15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap();
        assert!(!cell.is_ortho());
        assert!((cell.a()[0] - 10.0).abs() < 1e-12);
        assert!((cell.b()[0] - 5.0).abs() < 1e-12);
        let p = [0.2, 0.1, 1.0];
        let q = [9.7, 0.1, 1.0];
        assert!((cell.dist2(p, q) - 0.25).abs() <= 1e-12);
    }

    #[test]
    fn ase_rows_match_from_vectors() {
        let rows = [[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 10.0]];
        let a = Cell::from_ase(rows).unwrap();
        let b = Cell::from_vectors(rows[0], rows[1], rows[2], [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(a, b);
        assert_eq!(Cell::from_vesin(rows).unwrap(), a);
        assert_eq!(Cell::from_con(rows).unwrap(), a);
    }

    #[test]
    fn con_box_ortho_is_diagonal() {
        let cell = Cell::from_con_box([10.0, 11.0, 12.0], [90.0, 90.0, 90.0]).unwrap();
        assert!(cell.is_ortho());
        assert!((cell.widths()[0] - 10.0).abs() < 1e-12);
        assert!((cell.widths()[1] - 11.0).abs() < 1e-12);
        assert!((cell.widths()[2] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn rel_from_j_matches_seams_mixed_image() {
        let cell =
            Cell::from_lammps_bounds(15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap();
        let i = [0.5, 0.5, 1.0];
        let j = [1.0, 8.0, 1.0];
        let dr = cell.displacement(j, i);
        assert!((dr[0] - 4.5).abs() < 1e-10);
        assert!((dr[1] - 1.160254037844386).abs() < 1e-10);
        assert!(dr[2].abs() < 1e-12);
    }
}
