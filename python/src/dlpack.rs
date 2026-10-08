//! DLPack: read any array in place and hand results back without a copy.
//!
//! An input with `__dlpack__` (numpy, PyTorch, JAX, CuPy host memory) is
//! borrowed through its capsule for the length of one call. Contiguous
//! `float64` or `uint64` rows of three are read where they lie; strided
//! or `float32` input is copied once. Results go back through a small
//! exporter whose capsule owns the buffer, so `numpy.from_dlpack` (or
//! `torch.from_dlpack`) adopts it without copying.

use std::ffi::{c_char, c_void};

use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

const KDL_CPU: i32 = 1;
const KDL_CUDA_HOST: i32 = 3;
const KDL_INT: u8 = 0;
const KDL_UINT: u8 = 1;
const KDL_FLOAT: u8 = 2;

const DLTENSOR: &[u8] = b"dltensor\0";
const USED_DLTENSOR: &[u8] = b"used_dltensor\0";
const DLTENSOR_VERSIONED: &[u8] = b"dltensor_versioned\0";
const USED_DLTENSOR_VERSIONED: &[u8] = b"used_dltensor_versioned\0";

fn name(bytes: &'static [u8]) -> *const c_char {
    bytes.as_ptr().cast()
}

enum Release {
    Legacy(*mut DLManagedTensor),
    Versioned(*mut DLManagedTensorVersioned),
}

/// One array borrowed through DLPack. Dropping it calls the producer's
/// deleter, as the protocol asks of a consumer.
pub struct Borrowed {
    tensor: *const DLTensor,
    release: Release,
}

impl Drop for Borrowed {
    fn drop(&mut self) {
        // SAFETY: the capsule was renamed `used_*`, so only this guard
        // calls the deleter, once.
        unsafe {
            match self.release {
                Release::Legacy(p) => {
                    if let Some(d) = (*p).deleter {
                        d(p);
                    }
                }
                Release::Versioned(p) => {
                    if let Some(d) = (*p).deleter {
                        d(p);
                    }
                }
            }
        }
    }
}

/// Rows of three, read in place when the layout allows.
pub enum Rows<'a, T> {
    View(&'a [[T; 3]]),
    Owned(Vec<[T; 3]>),
}

impl<T> std::ops::Deref for Rows<'_, T> {
    type Target = [[T; 3]];

    fn deref(&self) -> &[[T; 3]] {
        match self {
            Rows::View(v) => v,
            Rows::Owned(v) => v,
        }
    }
}

/// Borrow `obj` through DLPack, or `None` when it is not a DLPack producer.
pub fn borrow(obj: &Bound<'_, PyAny>) -> PyResult<Option<Borrowed>> {
    if !obj.hasattr("__dlpack__")? {
        return Ok(None);
    }
    if let Ok(dev) = obj.call_method0("__dlpack_device__") {
        let (kind, _id): (i32, i32) = dev.extract()?;
        if kind != KDL_CPU && kind != KDL_CUDA_HOST {
            return Err(PyTypeError::new_err(
                "minimage reads host memory; move the array to the CPU first",
            ));
        }
    }
    let py = obj.py();
    // A read-only numpy array exports only as a versioned capsule.
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_version", (1u32, 0u32))?;
    let capsule = match obj.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(c) => c,
        Err(_) => obj.call_method0("__dlpack__")?,
    };
    let raw = capsule.as_ptr();
    // SAFETY: `raw` is a live object; the name checks guard each pointer.
    unsafe {
        if ffi::PyCapsule_IsValid(raw, name(DLTENSOR_VERSIONED)) == 1 {
            let p = ffi::PyCapsule_GetPointer(raw, name(DLTENSOR_VERSIONED))
                .cast::<DLManagedTensorVersioned>();
            if p.is_null() {
                return Err(PyErr::fetch(py));
            }
            if (*p).version.major != 1 {
                return Err(PyBufferError::new_err(format!(
                    "unsupported DLPack version {}.{}",
                    (*p).version.major,
                    (*p).version.minor
                )));
            }
            ffi::PyCapsule_SetName(raw, name(USED_DLTENSOR_VERSIONED));
            return Ok(Some(Borrowed {
                tensor: std::ptr::addr_of!((*p).dl_tensor),
                release: Release::Versioned(p),
            }));
        }
        if ffi::PyCapsule_IsValid(raw, name(DLTENSOR)) == 1 {
            let p = ffi::PyCapsule_GetPointer(raw, name(DLTENSOR)).cast::<DLManagedTensor>();
            if p.is_null() {
                return Err(PyErr::fetch(py));
            }
            ffi::PyCapsule_SetName(raw, name(USED_DLTENSOR));
            return Ok(Some(Borrowed {
                tensor: std::ptr::addr_of!((*p).dl_tensor),
                release: Release::Legacy(p),
            }));
        }
    }
    Err(PyTypeError::new_err(
        "__dlpack__ did not return a DLPack capsule",
    ))
}

impl Borrowed {
    fn tensor(&self) -> &DLTensor {
        // SAFETY: the producer keeps the tensor alive until the deleter runs.
        unsafe { &*self.tensor }
    }

    /// Rows count and element strides for an `(n, 3)` array, or a single
    /// row for a length-3 vector.
    fn layout(&self, what: &str) -> PyResult<(usize, [isize; 2])> {
        let t = self.tensor();
        let shape = |k: usize| -> i64 {
            // SAFETY: `k < ndim`, and `shape` holds `ndim` entries.
            unsafe { *t.shape.add(k) }
        };
        let stride = |k: usize, default: i64| -> i64 {
            if t.strides.is_null() {
                default
            } else {
                // SAFETY: `strides`, when present, holds `ndim` entries.
                unsafe { *t.strides.add(k) }
            }
        };
        if t.dtype.lanes != 1 {
            return Err(PyValueError::new_err(format!(
                "{what} must not be vector typed"
            )));
        }
        match t.ndim {
            2 if shape(1) == 3 => {
                let n = usize::try_from(shape(0))
                    .map_err(|_| PyValueError::new_err(format!("{what} has a negative length")))?;
                Ok((n, [stride(0, 3) as isize, stride(1, 1) as isize]))
            }
            1 if shape(0) == 3 => Ok((1, [3, stride(0, 1) as isize])),
            _ => Err(PyValueError::new_err(format!("{what} must be (N, 3)"))),
        }
    }

    fn base(&self) -> *const u8 {
        let t = self.tensor();
        // SAFETY: `byte_offset` lies inside the producer's buffer.
        unsafe { t.data.cast::<u8>().add(t.byte_offset as usize) }
    }

    /// `float64` rows read in place when contiguous and aligned; `float32`
    /// or strided rows copied once into doubles.
    pub fn rows_f64(&self, what: &str) -> PyResult<Rows<'_, f64>> {
        let (n, [s0, s1]) = self.layout(what)?;
        let dt = self.tensor().dtype;
        let base = self.base();
        match (dt.code, dt.bits) {
            (KDL_FLOAT, 64) => {
                if s0 == 3 && s1 == 1 && (base as usize) % std::mem::align_of::<f64>() == 0 {
                    // SAFETY: `n` contiguous, aligned rows of three doubles.
                    let rows = unsafe { std::slice::from_raw_parts(base.cast::<[f64; 3]>(), n) };
                    return Ok(Rows::View(rows));
                }
                let p = base.cast::<f64>();
                Ok(Rows::Owned(gather(n, s0, s1, |off| {
                    // SAFETY: each offset is inside the strided array.
                    unsafe { p.offset(off).read_unaligned() }
                })))
            }
            (KDL_FLOAT, 32) => {
                let p = base.cast::<f32>();
                Ok(Rows::Owned(gather(n, s0, s1, |off| {
                    // SAFETY: each offset is inside the strided array.
                    f64::from(unsafe { p.offset(off).read_unaligned() })
                })))
            }
            _ => Err(PyTypeError::new_err(format!(
                "{what} must be float64 or float32"
            ))),
        }
    }

    /// 32-bit integer rows, as from `fixed32_many`, read in place when
    /// contiguous and aligned.
    pub fn rows_u32(&self, what: &str) -> PyResult<Rows<'_, u32>> {
        let (n, [s0, s1]) = self.layout(what)?;
        let dt = self.tensor().dtype;
        if !((dt.code == KDL_UINT || dt.code == KDL_INT) && dt.bits == 32) {
            return Err(PyTypeError::new_err(format!("{what} must be uint32")));
        }
        let base = self.base();
        if s0 == 3 && s1 == 1 && (base as usize) % std::mem::align_of::<u32>() == 0 {
            // SAFETY: `n` contiguous, aligned rows of three integers.
            let rows = unsafe { std::slice::from_raw_parts(base.cast::<[u32; 3]>(), n) };
            return Ok(Rows::View(rows));
        }
        let p = base.cast::<u32>();
        Ok(Rows::Owned(gather(n, s0, s1, |off| {
            // SAFETY: each offset is inside the strided array.
            unsafe { p.offset(off).read_unaligned() }
        })))
    }

    /// 64-bit integer rows, as from `fixed_many`, read in place when
    /// contiguous and aligned.
    pub fn rows_u64(&self, what: &str) -> PyResult<Rows<'_, u64>> {
        let (n, [s0, s1]) = self.layout(what)?;
        let dt = self.tensor().dtype;
        if !((dt.code == KDL_UINT || dt.code == KDL_INT) && dt.bits == 64) {
            return Err(PyTypeError::new_err(format!("{what} must be uint64")));
        }
        let base = self.base();
        if s0 == 3 && s1 == 1 && (base as usize) % std::mem::align_of::<u64>() == 0 {
            // SAFETY: `n` contiguous, aligned rows of three integers.
            let rows = unsafe { std::slice::from_raw_parts(base.cast::<[u64; 3]>(), n) };
            return Ok(Rows::View(rows));
        }
        let p = base.cast::<u64>();
        Ok(Rows::Owned(gather(n, s0, s1, |off| {
            // SAFETY: each offset is inside the strided array.
            unsafe { p.offset(off).read_unaligned() }
        })))
    }
}

fn gather<T: Copy + Default>(
    n: usize,
    s0: isize,
    s1: isize,
    read: impl Fn(isize) -> T,
) -> Vec<[T; 3]> {
    (0..n as isize)
        .map(|i| [read(i * s0), read(i * s0 + s1), read(i * s0 + 2 * s1)])
        .collect()
}

enum Data {
    F64(Vec<f64>),
    U64(Vec<u64>),
    U32(Vec<u32>),
}

/// The buffer a capsule owns until the consumer's array lets go.
struct Holder {
    data: Data,
    shape: [i64; 2],
    ndim: i32,
}

unsafe extern "C" fn delete_managed(p: *mut DLManagedTensor) {
    // SAFETY: `p` and its context came from `Box::into_raw` in `__dlpack__`.
    unsafe {
        let managed = Box::from_raw(p);
        drop(Box::from_raw(managed.manager_ctx.cast::<Holder>()));
    }
}

unsafe extern "C" fn delete_versioned(p: *mut DLManagedTensorVersioned) {
    // SAFETY: `p` and its context came from `Box::into_raw` in `__dlpack__`.
    unsafe {
        let managed = Box::from_raw(p);
        drop(Box::from_raw(managed.manager_ctx.cast::<Holder>()));
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // A consumer renames the capsule; only an unclaimed one frees here.
    // SAFETY: the name checks guard each pointer.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, name(DLTENSOR)) == 1 {
            let p = ffi::PyCapsule_GetPointer(capsule, name(DLTENSOR)).cast::<DLManagedTensor>();
            if !p.is_null() {
                if let Some(d) = (*p).deleter {
                    d(p);
                }
            }
        } else if ffi::PyCapsule_IsValid(capsule, name(DLTENSOR_VERSIONED)) == 1 {
            let p = ffi::PyCapsule_GetPointer(capsule, name(DLTENSOR_VERSIONED))
                .cast::<DLManagedTensorVersioned>();
            if !p.is_null() {
                if let Some(d) = (*p).deleter {
                    d(p);
                }
            }
        }
    }
}

/// A result buffer that exports itself once through DLPack.
#[pyclass(module = "minimage", name = "_Exported")]
pub struct Exported {
    holder: Option<Holder>,
}

#[pymethods]
impl Exported {
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &mut self,
        py: Python<'_>,
        stream: Option<PyObject>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        let _ = (stream, copy);
        if let Some((kind, _)) = dl_device {
            if kind != KDL_CPU {
                return Err(PyBufferError::new_err("minimage results live on the CPU"));
            }
        }
        let holder = self
            .holder
            .take()
            .ok_or_else(|| PyBufferError::new_err("this result was already exported"))?;
        let ctx = Box::into_raw(Box::new(holder));
        // SAFETY: `ctx` is a live box until a deleter frees it.
        let (data, dtype, ndim, shape) = unsafe {
            let h = &mut *ctx;
            let (data, code, bits) = match &mut h.data {
                Data::F64(v) => (v.as_mut_ptr().cast::<c_void>(), KDL_FLOAT, 64),
                Data::U64(v) => (v.as_mut_ptr().cast::<c_void>(), KDL_UINT, 64),
                Data::U32(v) => (v.as_mut_ptr().cast::<c_void>(), KDL_UINT, 32),
            };
            let dtype = DLDataType {
                code,
                bits,
                lanes: 1,
            };
            (data, dtype, h.ndim, h.shape.as_mut_ptr())
        };
        let tensor = DLTensor {
            data,
            device: DLDevice {
                device_type: KDL_CPU,
                device_id: 0,
            },
            ndim,
            dtype,
            shape,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // A 1.x consumer gets the versioned capsule, whose flags say the
        // buffer is writable; numpy marks a legacy import read-only.
        if matches!(max_version, Some((major, _)) if major >= 1) {
            let managed = Box::into_raw(Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: ctx.cast::<c_void>(),
                deleter: Some(delete_versioned),
                flags: 0,
                dl_tensor: tensor,
            }));
            // SAFETY: the capsule takes the managed tensor; its destructor
            // frees it unless a consumer claimed it by renaming.
            unsafe {
                let capsule = ffi::PyCapsule_New(
                    managed.cast::<c_void>(),
                    name(DLTENSOR_VERSIONED),
                    Some(capsule_destructor),
                );
                if capsule.is_null() {
                    delete_versioned(managed);
                    return Err(PyErr::fetch(py));
                }
                return Ok(PyObject::from_owned_ptr(py, capsule));
            }
        }
        let managed = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: tensor,
            manager_ctx: ctx.cast::<c_void>(),
            deleter: Some(delete_managed),
        }));
        // SAFETY: as above, for the legacy capsule.
        unsafe {
            let capsule = ffi::PyCapsule_New(
                managed.cast::<c_void>(),
                name(DLTENSOR),
                Some(capsule_destructor),
            );
            if capsule.is_null() {
                delete_managed(managed);
                return Err(PyErr::fetch(py));
            }
            Ok(PyObject::from_owned_ptr(py, capsule))
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (KDL_CPU, 0)
    }
}

/// Whether numpy imports; an imported module is a dictionary lookup.
pub fn have_numpy(py: Python<'_>) -> bool {
    py.import("numpy").is_ok()
}

fn export(py: Python<'_>, holder: Holder) -> PyResult<PyObject> {
    let exported = Py::new(
        py,
        Exported {
            holder: Some(holder),
        },
    )?;
    Ok(py
        .import("numpy")?
        .call_method1("from_dlpack", (exported,))?
        .unbind())
}

/// The same allocation viewed as `3 n` scalars; no copy.
fn flatten<T>(rows: Vec<[T; 3]>) -> Vec<T> {
    let mut rows = std::mem::ManuallyDrop::new(rows);
    let (ptr, len, cap) = (rows.as_mut_ptr(), rows.len(), rows.capacity());
    // SAFETY: `[T; 3]` has the size of three `T` and the alignment of
    // one, so the allocation is a valid `Vec<T>` of `3 cap`.
    unsafe { Vec::from_raw_parts(ptr.cast::<T>(), len * 3, cap * 3) }
}

/// `values` as a numpy array that owns them.
pub fn array_f64(py: Python<'_>, values: Vec<f64>) -> PyResult<PyObject> {
    let n = values.len() as i64;
    export(
        py,
        Holder {
            data: Data::F64(values),
            shape: [n, 0],
            ndim: 1,
        },
    )
}

/// `rows` as an `(n, 3)` numpy array that owns them.
pub fn array_rows_f64(py: Python<'_>, rows: Vec<[f64; 3]>) -> PyResult<PyObject> {
    let n = rows.len() as i64;
    export(
        py,
        Holder {
            data: Data::F64(flatten(rows)),
            shape: [n, 3],
            ndim: 2,
        },
    )
}

/// `rows` as an `(n, 3)` uint32 numpy array that owns them.
pub fn array_rows_u32(py: Python<'_>, rows: Vec<[u32; 3]>) -> PyResult<PyObject> {
    let n = rows.len() as i64;
    export(
        py,
        Holder {
            data: Data::U32(flatten(rows)),
            shape: [n, 3],
            ndim: 2,
        },
    )
}

/// `rows` as an `(n, 3)` uint64 numpy array that owns them.
pub fn array_rows_u64(py: Python<'_>, rows: Vec<[u64; 3]>) -> PyResult<PyObject> {
    let n = rows.len() as i64;
    export(
        py,
        Holder {
            data: Data::U64(flatten(rows)),
            shape: [n, 3],
            ndim: 2,
        },
    )
}
