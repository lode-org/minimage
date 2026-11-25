//! Thin Python bindings for [`minimage::Cell`].

mod dlpack;

use minimage::{
    dist2_euclidean_many, dist2_euclidean_pairs, dist2_euclidean_pairs_warm, dist2_many,
    dist2_many_fixed, dist2_many_fixed32, dist2_pairs, dist2_pairs_fixed, dist2_pairs_fixed32,
    fixed32_many, fixed_many, reduce_pairs, wrap_many, Cell as RustCell,
    PeriodicSpace as RustPeriodicSpace,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

fn as_nested<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.extract::<Vec<f64>>().is_ok() || obj.extract::<Vec<Vec<f64>>>().is_ok() {
        return Ok(obj.clone());
    }
    obj.call_method0("tolist")
}

fn triple(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 3]> {
    let src = as_nested(obj)?;
    let v: Vec<f64> = src
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must have length 3")))?;
    if v.len() != 3 {
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    Ok([v[0], v[1], v[2]])
}

fn rows3(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[[f64; 3]; 3]> {
    let src = as_nested(obj)?;
    let v: Vec<Vec<f64>> = src
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be 3x3")))?;
    if v.len() != 3 || v.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err(format!("{what} must be 3x3")));
    }
    Ok([
        [v[0][0], v[0][1], v[0][2]],
        [v[1][0], v[1][1], v[1][2]],
        [v[2][0], v[2][1], v[2][2]],
    ])
}

fn triples(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let src = as_nested(obj)?;
    let v: Vec<Vec<f64>> = src
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be (N, 3)")))?;
    let mut out = Vec::with_capacity(v.len());
    for (i, row) in v.iter().enumerate() {
        if row.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "{what}[{i}] must have length 3"
            )));
        }
        out.push([row[0], row[1], row[2]]);
    }
    Ok(out)
}

fn vector(obj: &Bound<'_, PyAny>, dimensions: usize, what: &str) -> PyResult<Vec<f64>> {
    let values: Vec<f64> = as_nested(obj)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must have length {dimensions}")))?;
    if values.len() != dimensions {
        return Err(PyValueError::new_err(format!(
            "{what} must have length {dimensions}"
        )));
    }
    Ok(values)
}

fn rows(obj: &Bound<'_, PyAny>, dimensions: usize, what: &str) -> PyResult<(Vec<f64>, usize)> {
    let values: Vec<Vec<f64>> = as_nested(obj)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be (N, {dimensions})")))?;
    if let Some((row, _)) = values
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != dimensions)
    {
        return Err(PyValueError::new_err(format!(
            "{what}[{row}] must have length {dimensions}"
        )));
    }
    let count = values.len();
    Ok((values.into_iter().flatten().collect(), count))
}

fn map_err(err: minimage::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn fixed_triple(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[u64; 3]> {
    if let Some(b) = dlpack::borrow(obj)? {
        let rows = b.rows_u64(what)?;
        if rows.len() == 1 {
            return Ok(rows[0]);
        }
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    let v: Vec<u64> = as_nested(obj)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be three uint64")))?;
    if v.len() != 3 {
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    Ok([v[0], v[1], v[2]])
}

fn fixed_triples(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[u64; 3]>> {
    let v: Vec<Vec<u64>> = as_nested(obj)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be (N, 3) uint64")))?;
    v.iter()
        .enumerate()
        .map(|(i, row)| {
            if row.len() == 3 {
                Ok([row[0], row[1], row[2]])
            } else {
                Err(PyValueError::new_err(format!(
                    "{what}[{i}] must have length 3"
                )))
            }
        })
        .collect()
}

fn fixed32_triple(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[u32; 3]> {
    if let Some(b) = dlpack::borrow(obj)? {
        let rows = b.rows_u32(what)?;
        if rows.len() == 1 {
            return Ok(rows[0]);
        }
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    let v: Vec<u32> = as_nested(obj)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be three uint32")))?;
    if v.len() != 3 {
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    Ok([v[0], v[1], v[2]])
}

fn fixed32_triples(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[u32; 3]>> {
    let v: Vec<Vec<u32>> = as_nested(obj)?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be (N, 3) uint32")))?;
    v.iter()
        .enumerate()
        .map(|(i, row)| {
            if row.len() == 3 {
                Ok([row[0], row[1], row[2]])
            } else {
                Err(PyValueError::new_err(format!(
                    "{what}[{i}] must have length 3"
                )))
            }
        })
        .collect()
}

fn out_rows_u32(py: Python<'_>, rows: Vec<[u32; 3]>) -> PyResult<PyObject> {
    if dlpack::have_numpy(py) {
        dlpack::array_rows_u32(py, rows)
    } else {
        rows.into_py_any(py)
    }
}

/// A numpy array when numpy is present, else a list.
fn out_f64(py: Python<'_>, values: Vec<f64>) -> PyResult<PyObject> {
    if dlpack::have_numpy(py) {
        dlpack::array_f64(py, values)
    } else {
        values.into_py_any(py)
    }
}

fn out_rows_f64(py: Python<'_>, rows: Vec<[f64; 3]>) -> PyResult<PyObject> {
    if dlpack::have_numpy(py) {
        dlpack::array_rows_f64(py, rows)
    } else {
        rows.into_py_any(py)
    }
}

fn out_rows_u64(py: Python<'_>, rows: Vec<[u64; 3]>) -> PyResult<PyObject> {
    if dlpack::have_numpy(py) {
        dlpack::array_rows_u64(py, rows)
    } else {
        rows.into_py_any(py)
    }
}

/// Periodic parallelepiped: lattice vectors `a`, `b`, `c` and origin.
///
/// Coordinates, lattice vectors, origins, and displacements share one
/// caller-chosen Cartesian length unit; squared distances use that unit
/// squared. Constructors validate the box and raise `ValueError` on a
/// singular or non-finite cell.
#[pyclass(name = "Cell", module = "minimage")]
struct PyCell {
    inner: RustCell,
}

/// N-dimensional Euclidean product metric with optional per-axis periods.
///
/// Each period is a positive finite float for a periodic axis or `None` for
/// an open axis. Points use `[0, period)` and differences use
/// `[-period / 2, period / 2)`.
#[pyclass(name = "PeriodicSpace", module = "minimage")]
struct PyPeriodicSpace {
    inner: RustPeriodicSpace,
}

#[pymethods]
impl PyPeriodicSpace {
    /// Construct from one positive period or `None` per dimension.
    #[new]
    fn new(periods: Vec<Option<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: RustPeriodicSpace::new(&periods).map_err(map_err)?,
        })
    }

    /// Number of dimensions.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Validated periods (`None` marks an open axis).
    #[getter]
    fn periods(&self) -> Vec<Option<f64>> {
        self.inner.periods().to_vec()
    }

    /// Wrap one point into `[0, period)` on periodic axes.
    fn wrap_point(&self, point: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let mut out = vector(point, self.inner.dimensions(), "point")?;
        self.inner.wrap_point(&mut out).map_err(map_err)?;
        Ok(out)
    }

    /// Wrap one difference into `[-period / 2, period / 2)`.
    fn wrap_difference(&self, difference: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let mut out = vector(difference, self.inner.dimensions(), "difference")?;
        self.inner.wrap_difference(&mut out).map_err(map_err)?;
        Ok(out)
    }

    /// Wrapped displacement from `p` to `q`.
    fn displacement(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let p = vector(p, self.inner.dimensions(), "p")?;
        let q = vector(q, self.inner.dimensions(), "q")?;
        let mut out = vec![0.0; self.inner.dimensions()];
        self.inner.displacement(&p, &q, &mut out).map_err(map_err)?;
        Ok(out)
    }

    /// Squared minimum-image distance between `p` and `q`.
    fn dist2(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.dist2(
            &vector(p, self.inner.dimensions(), "p")?,
            &vector(q, self.inner.dimensions(), "q")?,
        ))
    }

    /// Minimum-image distance between `p` and `q`.
    fn distance(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.distance(
            &vector(p, self.inner.dimensions(), "p")?,
            &vector(q, self.inner.dimensions(), "q")?,
        ))
    }

    /// Wrap contiguous logical rows of points.
    fn wrap_points(&self, points: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let (mut flat, count) = rows(points, self.inner.dimensions(), "points")?;
        self.inner.wrap_points(&mut flat).map_err(map_err)?;
        Ok(flat
            .chunks_exact(self.inner.dimensions())
            .take(count)
            .map(<[f64]>::to_vec)
            .collect())
    }

    /// Wrap contiguous logical rows of difference vectors.
    fn wrap_differences(&self, differences: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let (mut flat, count) = rows(differences, self.inner.dimensions(), "differences")?;
        self.inner.wrap_differences(&mut flat).map_err(map_err)?;
        Ok(flat
            .chunks_exact(self.inner.dimensions())
            .take(count)
            .map(<[f64]>::to_vec)
            .collect())
    }

    /// Squared distances from `point` to each row of `points`.
    fn dist2_many(
        &self,
        point: &Bound<'_, PyAny>,
        points: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<f64>> {
        let point = vector(point, self.inner.dimensions(), "point")?;
        let (points, count) = rows(points, self.inner.dimensions(), "points")?;
        let mut out = vec![0.0; count];
        self.inner
            .dist2_many(&point, &points, &mut out)
            .map_err(map_err)?;
        Ok(out)
    }

    /// Squared distances between corresponding rows of `left` and `right`.
    fn dist2_pairs(&self, left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let (left, count) = rows(left, self.inner.dimensions(), "left")?;
        let (right, right_count) = rows(right, self.inner.dimensions(), "right")?;
        if count != right_count {
            return Err(map_err(minimage::Error::BufferSize));
        }
        let mut out = vec![0.0; count];
        self.inner
            .dist2_pairs(&left, &right, &mut out)
            .map_err(map_err)?;
        Ok(out)
    }

    /// Squared lower-bound distance from `x` to `[lo, hi]` on one axis.
    fn axis_min_dist2(&self, axis: usize, x: f64, lo: f64, hi: f64) -> PyResult<f64> {
        if axis >= self.inner.dimensions() {
            return Err(PyValueError::new_err("axis is outside the space"));
        }
        let value = self.inner.axis_min_dist2(axis, x, lo, hi);
        if value.is_nan() {
            Err(PyValueError::new_err("invalid interval bounds"))
        } else {
            Ok(value)
        }
    }

    /// Squared lower-bound distance from `point` to `[lower, upper]`.
    fn aabb_min_dist2(
        &self,
        point: &Bound<'_, PyAny>,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let value = self.inner.aabb_min_dist2(
            &vector(point, self.inner.dimensions(), "point")?,
            &vector(lower, self.inner.dimensions(), "lower")?,
            &vector(upper, self.inner.dimensions(), "upper")?,
        );
        if value.is_nan() {
            Err(PyValueError::new_err("invalid AABB bounds"))
        } else {
            Ok(value)
        }
    }
}

#[pymethods]
impl PyCell {
    /// Construct an orthorhombic cell with side lengths `(lx, ly, lz)` and
    /// zero origin.
    ///
    /// Raises `ValueError` if `lx * ly * lz` is non-finite or smaller than
    /// `1e-18` in magnitude.
    #[staticmethod]
    fn ortho(lx: f64, ly: f64, lz: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::ortho(lx, ly, lz).map_err(map_err)?,
        })
    }

    /// Construct a cell from lattice vectors `a`, `b`, `c` (columns of `H`)
    /// and optional `origin` (defaults to `[0.0, 0.0, 0.0]`).
    ///
    /// Raises `ValueError` if any vector has length != 3 or `H` is singular
    /// or non-finite.
    #[staticmethod]
    #[pyo3(signature = (a, b, c, origin=None))]
    fn from_vectors(
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        origin: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let origin = match origin {
            Some(o) => triple(o, "origin")?,
            None => [0.0, 0.0, 0.0],
        };
        Ok(Self {
            inner: RustCell::from_vectors(
                triple(a, "a")?,
                triple(b, "b")?,
                triple(c, "c")?,
                origin,
            )
            .map_err(map_err)?,
        })
    }

    /// Construct a restricted-triclinic cell from LAMMPS dump bound spans
    /// `(xspan, yspan, zspan)`, tilts `(xy, xz, yz)`, and bound origins
    /// `(xlo_b, ylo_b, zlo_b)`.
    ///
    /// Raises `ValueError` if the recovered cell is singular or non-finite.
    #[staticmethod]
    fn from_lammps_bounds(
        xspan: f64,
        yspan: f64,
        zspan: f64,
        xy: f64,
        xz: f64,
        yz: f64,
        xlo_b: f64,
        ylo_b: f64,
        zlo_b: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_lammps_bounds(
                xspan, yspan, zspan, xy, xz, yz, xlo_b, ylo_b, zlo_b,
            )
            .map_err(map_err)?,
        })
    }

    /// Construct from an ASE-style `(3, 3)` row-major cell (`[a, b, c]`),
    /// with origin `[0.0, 0.0, 0.0]`.
    ///
    /// Raises `ValueError` if `cell` is not 3x3 or is singular / non-finite.
    #[staticmethod]
    fn from_ase(cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_ase(rows3(cell, "cell")?).map_err(map_err)?,
        })
    }

    /// Construct from a CON `(3, 3)` row-major lattice, zero origin.
    ///
    /// Raises `ValueError` if `cell` is not 3x3 or is singular / non-finite.
    #[staticmethod]
    fn from_con(cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_con(rows3(cell, "cell")?).map_err(map_err)?,
        })
    }

    /// Construct from a vesin `(3, 3)` row-major box, zero origin.
    ///
    /// Raises `ValueError` if `cell` is not 3x3 or is singular / non-finite.
    #[staticmethod]
    fn from_vesin(cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_vesin(rows3(cell, "cell")?).map_err(map_err)?,
        })
    }

    /// Squared engine minimum-image distance between length-3 vectors `p`
    /// and `q`, in squared cell length units. Raises `ValueError` if `p`
    /// or `q` does not have length 3.
    fn dist2(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.dist2(triple(p, "p")?, triple(q, "q")?))
    }

    /// Engine minimum-image displacement from `p` to `q` (`wrap(q - p)`) as
    /// `[dx, dy, dz]` in cell length units. Raises `ValueError` if `p` or
    /// `q` does not have length 3.
    fn displacement(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
        Ok(self.inner.displacement(triple(p, "p")?, triple(q, "q")?))
    }

    /// Squared engine MIC distances from `p` (length 3) to each row of
    /// `qs` (`(N, 3)`).
    ///
    /// Accepts any host CPU DLPack producer (`float64` or `float32`) or
    /// nested Python sequences. Returns a `float64` NumPy array of shape
    /// `(N,)` for array inputs (when NumPy is importable), or a Python list
    /// for list inputs. Raises `ValueError` on shape mismatch, `TypeError`
    /// on unsupported dtype or non-CPU device, and `BufferError` on
    /// unsupported DLPack capsule versions.
    fn dist2_many(
        &self,
        py: Python<'_>,
        p: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let p = triple(p, "p")?;
        if let Some(b) = dlpack::borrow(qs)? {
            let rows = b.rows_f64("qs")?;
            let mut out = vec![0.0; rows.len()];
            dist2_many(&self.inner, p, &rows, &mut out).map_err(map_err)?;
            drop(rows);
            drop(b);
            return out_f64(py, out);
        }
        let packed = triples(qs, "qs")?;
        let mut out = vec![0.0; packed.len()];
        dist2_many(&self.inner, p, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// Squared distances for paired rows of `ps` and `qs`.
    fn dist2_pairs(
        &self,
        py: Python<'_>,
        ps: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let bp = dlpack::borrow(ps)?;
        let bq = dlpack::borrow(qs)?;
        let array = bp.is_some() || bq.is_some();
        let rp = match &bp {
            Some(b) => b.rows_f64("ps")?,
            None => dlpack::Rows::Owned(triples(ps, "ps")?),
        };
        let rq = match &bq {
            Some(b) => b.rows_f64("qs")?,
            None => dlpack::Rows::Owned(triples(qs, "qs")?),
        };
        let mut out = vec![0.0; rp.len()];
        dist2_pairs(&self.inner, &rp, &rq, &mut out).map_err(map_err)?;
        if array {
            out_f64(py, out)
        } else {
            out.into_py_any(py)
        }
    }

    /// Euclidean nearest-image squared distances from `p` to each row of
    /// `qs`, one superbasis lookup per call.
    fn dist2_euclidean_many(
        &self,
        py: Python<'_>,
        p: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let p = triple(p, "p")?;
        if let Some(b) = dlpack::borrow(qs)? {
            let rows = b.rows_f64("qs")?;
            let mut out = vec![0.0; rows.len()];
            dist2_euclidean_many(&self.inner, p, &rows, &mut out).map_err(map_err)?;
            return out_f64(py, out);
        }
        let packed = triples(qs, "qs")?;
        let mut out = vec![0.0; packed.len()];
        dist2_euclidean_many(&self.inner, p, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// Euclidean nearest-image squared distances for paired rows.
    fn dist2_euclidean_pairs(
        &self,
        py: Python<'_>,
        ps: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let bp = dlpack::borrow(ps)?;
        let bq = dlpack::borrow(qs)?;
        let array = bp.is_some() || bq.is_some();
        let rp = match &bp {
            Some(b) => b.rows_f64("ps")?,
            None => dlpack::Rows::Owned(triples(ps, "ps")?),
        };
        let rq = match &bq {
            Some(b) => b.rows_f64("qs")?,
            None => dlpack::Rows::Owned(triples(qs, "qs")?),
        };
        let mut out = vec![0.0; rp.len()];
        dist2_euclidean_pairs(&self.inner, &rp, &rq, &mut out).map_err(map_err)?;
        if array {
            out_f64(py, out)
        } else {
            out.into_py_any(py)
        }
    }

    /// Euclidean nearest-image squared distances for pairs that persist
    /// across frames. `images` holds each pair's lattice shift from the
    /// previous call (zeros, or `None`, to start); returns the distances
    /// and the updated shifts.
    #[pyo3(signature = (ps, qs, images=None))]
    fn dist2_euclidean_pairs_warm(
        &self,
        py: Python<'_>,
        ps: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
        images: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let bp = dlpack::borrow(ps)?;
        let bq = dlpack::borrow(qs)?;
        let rp = match &bp {
            Some(b) => b.rows_f64("ps")?,
            None => dlpack::Rows::Owned(triples(ps, "ps")?),
        };
        let rq = match &bq {
            Some(b) => b.rows_f64("qs")?,
            None => dlpack::Rows::Owned(triples(qs, "qs")?),
        };
        let mut shifts: Vec<[i32; 3]> = match images {
            None => vec![[0; 3]; rp.len()],
            Some(obj) => match dlpack::borrow(obj)? {
                Some(b) => b
                    .rows_u32("images")?
                    .iter()
                    .map(|r| r.map(|v| v as i32))
                    .collect(),
                None => {
                    let v: Vec<Vec<i32>> = as_nested(obj)?
                        .extract()
                        .map_err(|_| PyValueError::new_err("images must be (N, 3) int32"))?;
                    v.iter()
                        .map(|r| {
                            if r.len() == 3 {
                                Ok([r[0], r[1], r[2]])
                            } else {
                                Err(PyValueError::new_err("images rows must have length 3"))
                            }
                        })
                        .collect::<PyResult<_>>()?
                }
            },
        };
        let mut out = vec![0.0; rp.len()];
        dist2_euclidean_pairs_warm(&self.inner, &rp, &rq, &mut shifts, &mut out)
            .map_err(map_err)?;
        if dlpack::have_numpy(py) {
            Ok((
                dlpack::array_f64(py, out)?,
                dlpack::array_rows_i32(py, shifts)?,
            ))
        } else {
            Ok((out.into_py_any(py)?, shifts.into_py_any(py)?))
        }
    }

    /// 32-bit fixed-point fractions of one position: half the bytes of
    /// `fixed`, within `2^-33` of the cell's fractions.
    fn fixed32(&self, r: &Bound<'_, PyAny>) -> PyResult<[u32; 3]> {
        Ok(self.inner.fixed32(triple(r, "r")?))
    }

    /// 32-bit fixed-point fractions of each row, as uint32.
    fn fixed32_many(&self, py: Python<'_>, rs: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Some(b) = dlpack::borrow(rs)? {
            let rows = b.rows_f64("rs")?;
            let mut out = vec![[0u32; 3]; rows.len()];
            fixed32_many(&self.inner, &rows, &mut out).map_err(map_err)?;
            return out_rows_u32(py, out);
        }
        let packed = triples(rs, "rs")?;
        let mut out = vec![[0u32; 3]; packed.len()];
        fixed32_many(&self.inner, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// Squared distance between two 32-bit fixed-point positions.
    fn dist2_fixed32(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self
            .inner
            .dist2_fixed32(fixed32_triple(a, "a")?, fixed32_triple(b, "b")?))
    }

    /// Squared distances from 32-bit fixed-point `p` to each row.
    fn dist2_many_fixed32(
        &self,
        py: Python<'_>,
        p: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let p = fixed32_triple(p, "p")?;
        if let Some(b) = dlpack::borrow(qs)? {
            let rows = b.rows_u32("qs")?;
            let mut out = vec![0.0; rows.len()];
            dist2_many_fixed32(&self.inner, p, &rows, &mut out).map_err(map_err)?;
            return out_f64(py, out);
        }
        let packed = fixed32_triples(qs, "qs")?;
        let mut out = vec![0.0; packed.len()];
        dist2_many_fixed32(&self.inner, p, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// Squared distances for paired 32-bit fixed-point rows.
    fn dist2_pairs_fixed32(
        &self,
        py: Python<'_>,
        ps: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let bp = dlpack::borrow(ps)?;
        let bq = dlpack::borrow(qs)?;
        let array = bp.is_some() || bq.is_some();
        let rp = match &bp {
            Some(b) => b.rows_u32("ps")?,
            None => dlpack::Rows::Owned(fixed32_triples(ps, "ps")?),
        };
        let rq = match &bq {
            Some(b) => b.rows_u32("qs")?,
            None => dlpack::Rows::Owned(fixed32_triples(qs, "qs")?),
        };
        let mut out = vec![0.0; rp.len()];
        dist2_pairs_fixed32(&self.inner, &rp, &rq, &mut out).map_err(map_err)?;
        if array {
            out_f64(py, out)
        } else {
            out.into_py_any(py)
        }
    }

    /// Fixed-point fractional coordinates of one position.
    fn fixed(&self, r: &Bound<'_, PyAny>) -> PyResult<[u64; 3]> {
        Ok(self.inner.fixed(triple(r, "r")?))
    }

    /// Fixed-point fractional coordinates of each row, as uint64.
    fn fixed_many(&self, py: Python<'_>, rs: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Some(b) = dlpack::borrow(rs)? {
            let rows = b.rows_f64("rs")?;
            let mut out = vec![[0u64; 3]; rows.len()];
            fixed_many(&self.inner, &rows, &mut out).map_err(map_err)?;
            return out_rows_u64(py, out);
        }
        let packed = triples(rs, "rs")?;
        let mut out = vec![[0u64; 3]; packed.len()];
        fixed_many(&self.inner, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// Squared distance between two fixed-point positions.
    fn dist2_fixed(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self
            .inner
            .dist2_fixed(fixed_triple(a, "a")?, fixed_triple(b, "b")?))
    }

    /// Squared distances from fixed-point `p` to each fixed-point row.
    fn dist2_many_fixed(
        &self,
        py: Python<'_>,
        p: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let p = fixed_triple(p, "p")?;
        if let Some(b) = dlpack::borrow(qs)? {
            let rows = b.rows_u64("qs")?;
            let mut out = vec![0.0; rows.len()];
            dist2_many_fixed(&self.inner, p, &rows, &mut out).map_err(map_err)?;
            return out_f64(py, out);
        }
        let packed = fixed_triples(qs, "qs")?;
        let mut out = vec![0.0; packed.len()];
        dist2_many_fixed(&self.inner, p, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// Squared distances for paired fixed-point rows.
    fn dist2_pairs_fixed(
        &self,
        py: Python<'_>,
        ps: &Bound<'_, PyAny>,
        qs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let bp = dlpack::borrow(ps)?;
        let bq = dlpack::borrow(qs)?;
        let array = bp.is_some() || bq.is_some();
        let rp = match &bp {
            Some(b) => b.rows_u64("ps")?,
            None => dlpack::Rows::Owned(fixed_triples(ps, "ps")?),
        };
        let rq = match &bq {
            Some(b) => b.rows_u64("qs")?,
            None => dlpack::Rows::Owned(fixed_triples(qs, "qs")?),
        };
        let mut out = vec![0.0; rp.len()];
        dist2_pairs_fixed(&self.inner, &rp, &rq, &mut out).map_err(map_err)?;
        if array {
            out_f64(py, out)
        } else {
            out.into_py_any(py)
        }
    }

    /// Engine wrap of a single difference vector `diff` (`q - p`) in cell
    /// length units. Equivalent to `displacement([0, 0, 0], diff)`.
    /// Raises `ValueError` if `diff` does not have length 3.
    fn wrap(&self, diff: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
        Ok(self
            .inner
            .displacement([0.0, 0.0, 0.0], triple(diff, "diff")?))
    }

    /// True when `a` lies along +x/-x and `b` lies in the xy plane
    /// (LAMMPS / HOOMD / GROMACS restricted-triclinic frame).
    fn is_restricted(&self) -> bool {
        self.inner.is_restricted()
    }

    /// True when the cell is restricted and `|xy| <= |lx|/2`,
    /// `|xz| <= |lx|/2`, and `|yz| <= |ly|/2`.
    fn tilts_reduced(&self) -> bool {
        self.inner.tilts_reduced()
    }

    /// True when the columns of `H` satisfy the 3D Minkowski inequalities.
    fn is_minkowski_reduced(&self) -> bool {
        minimage::is_minkowski_reduced(&self.inner)
    }

    /// Return a new `Cell` with unimodular tilt reduction (GROMACS
    /// `correct_box`), rotating into the restricted frame first if needed.
    /// Raises `ValueError` if reduction fails.
    fn reduce_tilts(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.reduce_tilts().map_err(map_err)?,
        })
    }

    /// Rotate a general parallelepiped onto the restricted-triclinic frame.
    /// Identity when already restricted. Raises `ValueError` on failure.
    fn to_restricted(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.to_restricted().map_err(map_err)?,
        })
    }

    /// Euclidean nearest-image displacement from `p` to `q` in cell length
    /// units (Smith test, else the Selling superbasis slicer; ties keep the
    /// engine vector). Raises `ValueError` if `p` or `q` lacks length 3.
    fn displacement_euclidean(
        &self,
        p: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<[f64; 3]> {
        Ok(self
            .inner
            .displacement_euclidean(triple(p, "p")?, triple(q, "q")?))
    }

    /// Squared Euclidean nearest-image distance between `p` and `q`, in
    /// squared cell length units. Raises `ValueError` if `p` or `q` lacks
    /// length 3.
    fn dist2_euclidean(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.dist2_euclidean(triple(p, "p")?, triple(q, "q")?))
    }

    /// Cartesian nearest-image displacement from `p` to `q` by checking
    /// the 27 nearest lattice shifts on the caller's `H`. Raises
    /// `ValueError` if `p` or `q` lacks length 3.
    fn displacement_cartesian(
        &self,
        p: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<[f64; 3]> {
        Ok(self
            .inner
            .displacement_cartesian(triple(p, "p")?, triple(q, "q")?))
    }

    /// True when fractional wrap matches the 27-image Cartesian check on a
    /// grid of short cutoff-scale probes.
    fn fractional_matches_cartesian(&self) -> bool {
        self.inner.fractional_matches_cartesian()
    }

    /// Engine wrap of each difference row in `diffs` (`(N, 3)`).
    ///
    /// Accepts any host CPU DLPack producer (`float64` or `float32`) or
    /// nested Python sequences. Returns an `(N, 3)` `float64` NumPy array
    /// for array inputs, or a Python list otherwise.
    fn wrap_many(&self, py: Python<'_>, diffs: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Some(b) = dlpack::borrow(diffs)? {
            let rows = b.rows_f64("diffs")?;
            let mut out = vec![[0.0, 0.0, 0.0]; rows.len()];
            wrap_many(&self.inner, &rows, &mut out).map_err(map_err)?;
            return out_rows_f64(py, out);
        }
        let packed = triples(diffs, "diffs")?;
        let mut out = vec![[0.0, 0.0, 0.0]; packed.len()];
        wrap_many(&self.inner, &packed, &mut out).map_err(map_err)?;
        out.into_py_any(py)
    }

    /// True when `H` is diagonal within numerical tolerance.
    fn is_ortho(&self) -> bool {
        self.inner.is_ortho()
    }
}

/// Drop self images (`i == j`) and collapse duplicate `(i, j)` pairs.
///
/// Takes a sequence of `(i, j)` integer pairs and returns sorted unique
/// pairs with `i != j`. Ordered pairs `(i, j)` and `(j, i)` are distinct.
#[pyfunction]
fn reduce_image_pairs(pairs: Vec<(i32, i32)>) -> PyResult<Vec<(i32, i32)>> {
    let rows: Vec<[i32; 2]> = pairs.into_iter().map(|(i, j)| [i, j]).collect();
    let mut out = Vec::new();
    reduce_pairs(&rows, &mut out).map_err(map_err)?;
    Ok(out.into_iter().map(|p| (p[0], p[1])).collect())
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyCell>()?;
    m.add_class::<PyPeriodicSpace>()?;
    m.add_function(wrap_pyfunction!(reduce_image_pairs, m)?)?;
    Ok(())
}
