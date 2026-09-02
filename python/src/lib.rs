//! Thin Python bindings for [`minimage::Cell`].

use minimage::{dist2_many, reduce_pairs, Cell as RustCell};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn triple(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 3]> {
    let v: Vec<f64> = obj.extract()?;
    if v.len() != 3 {
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    Ok([v[0], v[1], v[2]])
}

fn rows3(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[[f64; 3]; 3]> {
    let v: Vec<Vec<f64>> = obj.extract()?;
    if v.len() != 3 || v.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err(format!("{what} must be 3x3")));
    }
    Ok([
        [v[0][0], v[0][1], v[0][2]],
        [v[1][0], v[1][1], v[1][2]],
        [v[2][0], v[2][1], v[2][2]],
    ])
}

fn map_err(err: minimage::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
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

    fn dist2_many(&self, p: &Bound<'_, PyAny>, qs: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let qs: Vec<Vec<f64>> = qs.extract()?;
        let mut packed = Vec::with_capacity(qs.len());
        for (i, row) in qs.iter().enumerate() {
            if row.len() != 3 {
                return Err(PyValueError::new_err(format!("qs[{i}] must have length 3")));
            }
            packed.push([row[0], row[1], row[2]]);
        }
        let mut out = vec![0.0; packed.len()];
        dist2_many(&self.inner, triple(p, "p")?, &packed, &mut out).map_err(map_err)?;
        Ok(out)
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
