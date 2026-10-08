//! Thin Python bindings for [`minimage::Cell`].

mod dlpack;

use minimage::{
    dist2_euclidean_many, dist2_euclidean_pairs, dist2_many, dist2_many_fixed, dist2_many_fixed32,
    dist2_pairs, dist2_pairs_fixed, dist2_pairs_fixed32, fixed32_many, fixed_many, reduce_pairs,
    wrap_many, Cell as RustCell,
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

/// Periodic parallelepiped: lattice vectors a, b, c and origin.
#[pyclass(name = "Cell", module = "minimage")]
struct PyCell {
    inner: RustCell,
}

#[pymethods]
impl PyCell {
    #[staticmethod]
    fn ortho(lx: f64, ly: f64, lz: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::ortho(lx, ly, lz).map_err(map_err)?,
        })
    }

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

    #[staticmethod]
    fn from_ase(cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_ase(rows3(cell, "cell")?).map_err(map_err)?,
        })
    }

    #[staticmethod]
    fn from_con(cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_con(rows3(cell, "cell")?).map_err(map_err)?,
        })
    }

    #[staticmethod]
    fn from_vesin(cell: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RustCell::from_vesin(rows3(cell, "cell")?).map_err(map_err)?,
        })
    }

    fn dist2(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.dist2(triple(p, "p")?, triple(q, "q")?))
    }

    fn displacement(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
        Ok(self.inner.displacement(triple(p, "p")?, triple(q, "q")?))
    }

    /// Squared distances from `p` to each row of `qs`. An array in (any
    /// DLPack producer) is read in place and a numpy array comes back; a
    /// list in gives a list.
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

    fn wrap(&self, diff: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
        Ok(self
            .inner
            .displacement([0.0, 0.0, 0.0], triple(diff, "diff")?))
    }

    fn is_restricted(&self) -> bool {
        self.inner.is_restricted()
    }

    fn tilts_reduced(&self) -> bool {
        self.inner.tilts_reduced()
    }

    fn is_minkowski_reduced(&self) -> bool {
        minimage::is_minkowski_reduced(&self.inner)
    }

    fn reduce_tilts(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.reduce_tilts().map_err(map_err)?,
        })
    }

    fn to_restricted(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.to_restricted().map_err(map_err)?,
        })
    }

    fn displacement_euclidean(
        &self,
        p: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<[f64; 3]> {
        Ok(self
            .inner
            .displacement_euclidean(triple(p, "p")?, triple(q, "q")?))
    }

    fn dist2_euclidean(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.dist2_euclidean(triple(p, "p")?, triple(q, "q")?))
    }

    fn displacement_cartesian(
        &self,
        p: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<[f64; 3]> {
        Ok(self
            .inner
            .displacement_cartesian(triple(p, "p")?, triple(q, "q")?))
    }

    fn fractional_matches_cartesian(&self) -> bool {
        self.inner.fractional_matches_cartesian()
    }

    /// Engine wrap of each difference row. Array in, numpy array out.
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

    fn is_ortho(&self) -> bool {
        self.inner.is_ortho()
    }
}

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
    m.add_function(wrap_pyfunction!(reduce_image_pairs, m)?)?;
    Ok(())
}
