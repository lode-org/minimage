//! C ABI. Prefix `mi_`. Caller owns every buffer.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CString};
use std::ptr;
use std::slice;

use crate::{dist2_many, dist2_ortho_diffs, dist2_pairs, reduce_pairs_packed, Cell, Error};

/// Periodic parallelepiped. Lattice vectors are a, b, c (same as vesin rows).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct mi_cell {
    /// Lattice vector a, x.
    pub ax: f64,
    /// Lattice vector a, y.
    pub ay: f64,
    /// Lattice vector a, z.
    pub az: f64,
    /// Lattice vector b, x.
    pub bx: f64,
    /// Lattice vector b, y.
    pub by: f64,
    /// Lattice vector b, z.
    pub bz: f64,
    /// Lattice vector c, x.
    pub cx: f64,
    /// Lattice vector c, y.
    pub cy: f64,
    /// Lattice vector c, z.
    pub cz: f64,
    /// Origin x.
    pub ox: f64,
    /// Origin y.
    pub oy: f64,
    /// Origin z.
    pub oz: f64,
}

impl mi_cell {
    fn to_cell(self) -> Result<Cell, Error> {
        Cell::from_vectors(
            [self.ax, self.ay, self.az],
            [self.bx, self.by, self.bz],
            [self.cx, self.cy, self.cz],
            [self.ox, self.oy, self.oz],
        )
    }

    fn from_cell(cell: &Cell) -> Self {
        let a = cell.a();
        let b = cell.b();
        let c = cell.c();
        let o = cell.origin();
        Self {
            ax: a[0],
            ay: a[1],
            az: a[2],
            bx: b[0],
            by: b[1],
            bz: b[2],
            cx: c[0],
            cy: c[1],
            cz: c[2],
            ox: o[0],
            oy: o[1],
            oz: o[2],
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|slot| {
        let cstr = CString::new(msg).unwrap_or_else(|_| {
            CString::new("error message contained NUL").expect("fallback has no NUL")
        });
        *slot.borrow_mut() = Some(cstr);
    });
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn fail(err: Error) -> c_int {
    set_error(&err.to_string());
    1
}

fn fail_msg(msg: &str) -> c_int {
    set_error(msg);
    1
}

/// Thread-local last-error string from this thread's most recent `mi_*`
/// failure.
///
/// Returns a pointer to a NUL-terminated UTF-8 C string, or `NULL` if
/// the last call on this thread succeeded, or if none has failed yet.
/// [`mi_version`] does not read or write the slot. Distinct threads
/// have independent slots. The pointer is valid until the next `mi_*`
/// call on this thread. Do not free it.
#[no_mangle]
pub extern "C" fn mi_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// Library version string. Process-static, NUL-terminated. Do not free.
#[no_mangle]
pub extern "C" fn mi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

fn write_cell(cell: Result<Cell, Error>, out: *mut mi_cell) -> c_int {
    if out.is_null() {
        return fail_msg("null cell output");
    }
    match cell {
        Ok(c) => {
            clear_error();
            // SAFETY: `out` is a valid `mi_cell`.
            unsafe {
                *out = mi_cell::from_cell(&c);
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Fill `out` from lattice vectors a, b, c and an origin.
///
/// # Safety
///
/// `a`, `b`, `c`, `origin`, and `out` are non-null and point at three
/// doubles / one `mi_cell`.
#[no_mangle]
pub unsafe extern "C" fn mi_cell_from_vectors(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    origin: *const f64,
    out: *mut mi_cell,
) -> c_int {
    if a.is_null() || b.is_null() || c.is_null() || origin.is_null() {
        return fail_msg("null lattice vector");
    }
    // SAFETY: each pointer is three readable doubles.
    let a = unsafe { [*a, *a.add(1), *a.add(2)] };
    let b = unsafe { [*b, *b.add(1), *b.add(2)] };
    let c = unsafe { [*c, *c.add(1), *c.add(2)] };
    let origin = unsafe { [*origin, *origin.add(1), *origin.add(2)] };
    write_cell(Cell::from_vectors(a, b, c, origin), out)
}

/// Fill `out` from LAMMPS `xlo xhi ylo yhi zlo zhi` and tilts.
#[no_mangle]
pub extern "C" fn mi_cell_from_lammps(
    xlo: f64,
    xhi: f64,
    ylo: f64,
    yhi: f64,
    zlo: f64,
    zhi: f64,
    xy: f64,
    xz: f64,
    yz: f64,
    out: *mut mi_cell,
) -> c_int {
    write_cell(
        Cell::from_lammps(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz),
        out,
    )
}

/// Fill `out` from dump bound spans, tilts, and bound lo.
#[no_mangle]
pub extern "C" fn mi_cell_from_lammps_bounds(
    xspan: f64,
    yspan: f64,
    zspan: f64,
    xy: f64,
    xz: f64,
    yz: f64,
    xlo_b: f64,
    ylo_b: f64,
    zlo_b: f64,
    out: *mut mi_cell,
) -> c_int {
    write_cell(
        Cell::from_lammps_bounds(xspan, yspan, zspan, xy, xz, yz, xlo_b, ylo_b, zlo_b),
        out,
    )
}

/// Fill `out` from an ASE-style row-major 3x3 cell. `origin` may be NULL
/// (then zero).
///
/// # Safety
///
/// `rows` is nine readable doubles. `origin`, if non-null, is three
/// readable doubles. `out` is one writable `mi_cell`.
#[no_mangle]
pub unsafe extern "C" fn mi_cell_from_ase(
    rows: *const f64,
    origin: *const f64,
    out: *mut mi_cell,
) -> c_int {
    if rows.is_null() {
        return fail_msg("null ASE cell");
    }
    // SAFETY: nine readable doubles.
    let rows = unsafe {
        [
            [*rows, *rows.add(1), *rows.add(2)],
            [*rows.add(3), *rows.add(4), *rows.add(5)],
            [*rows.add(6), *rows.add(7), *rows.add(8)],
        ]
    };
    let cell = if origin.is_null() {
        Cell::from_ase(rows)
    } else {
        let origin = unsafe { [*origin, *origin.add(1), *origin.add(2)] };
        Cell::from_vectors(rows[0], rows[1], rows[2], origin)
    };
    write_cell(cell, out)
}

/// Fill `out` from a CON 3x3 lattice (rows a, b, c).
///
/// # Safety
///
/// Same contract as [`mi_cell_from_ase`] with a null origin.
#[no_mangle]
pub unsafe extern "C" fn mi_cell_from_con(rows: *const f64, out: *mut mi_cell) -> c_int {
    unsafe { mi_cell_from_ase(rows, ptr::null(), out) }
}

/// Fill `out` from CON lengths and angles in degrees.
///
/// # Safety
///
/// `boxl` and `angles_deg` are three readable doubles. `out` is one
/// writable `mi_cell`.
#[no_mangle]
pub unsafe extern "C" fn mi_cell_from_con_box(
    boxl: *const f64,
    angles_deg: *const f64,
    out: *mut mi_cell,
) -> c_int {
    if boxl.is_null() || angles_deg.is_null() {
        return fail_msg("null CON box");
    }
    let boxl = unsafe { [*boxl, *boxl.add(1), *boxl.add(2)] };
    let angles = unsafe { [*angles_deg, *angles_deg.add(1), *angles_deg.add(2)] };
    write_cell(Cell::from_con_box(boxl, angles), out)
}

/// Fill `out` from a vesin 3x3 box (rows a, b, c).
///
/// # Safety
///
/// Same contract as [`mi_cell_from_con`].
#[no_mangle]
pub unsafe extern "C" fn mi_cell_from_vesin(box_rows: *const f64, out: *mut mi_cell) -> c_int {
    unsafe { mi_cell_from_con(box_rows, out) }
}

fn read_cell(simbox: *const mi_cell) -> Result<Cell, c_int> {
    if simbox.is_null() {
        return Err(fail_msg("null cell"));
    }
    // SAFETY: one readable `mi_cell`.
    let raw = unsafe { *simbox };
    raw.to_cell().map_err(fail)
}

fn read3(p: *const f64, what: &str) -> Result<[f64; 3], c_int> {
    if p.is_null() {
        return Err(fail_msg(what));
    }
    Ok(unsafe { [*p, *p.add(1), *p.add(2)] })
}

/// Minimum-image displacement from `p` to `q` into `dr`.
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `p`, `q`, and `dr` are three doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_displacement(
    simbox: *const mi_cell,
    p: *const f64,
    q: *const f64,
    dr: *mut f64,
) -> c_int {
    let cell = match read_cell(simbox) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let p = match read3(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let q = match read3(q, "null q") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if dr.is_null() {
        return fail_msg("null dr");
    }
    let v = cell.displacement(p, q);
    unsafe {
        *dr = v[0];
        *dr.add(1) = v[1];
        *dr.add(2) = v[2];
    }
    clear_error();
    0
}

/// Squared minimum-image distance from `p` to `q`.
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `p` and `q` are three doubles. `out` is
/// one writable double.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2(
    simbox: *const mi_cell,
    p: *const f64,
    q: *const f64,
    out: *mut f64,
) -> c_int {
    let cell = match read_cell(simbox) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let p = match read3(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let q = match read3(q, "null q") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if out.is_null() {
        return fail_msg("null out");
    }
    unsafe {
        *out = cell.dist2(p, q);
    }
    clear_error();
    0
}

fn packed_triples<'a>(ptr: *const f64, n: usize, what: &str) -> Result<&'a [[f64; 3]], c_int> {
    if n == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(fail_msg(what));
    }
    // SAFETY: `n * 3` readable doubles, viewed as `n` triples.
    Ok(unsafe { slice::from_raw_parts(ptr as *const [f64; 3], n) })
}

/// Squared MIC distances from `p` to `n` packed candidates in `qs`.
///
/// # Safety
///
/// `qs` is `n * 3` doubles. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_many(
    simbox: *const mi_cell,
    p: *const f64,
    qs: *const f64,
    n: usize,
    out: *mut f64,
) -> c_int {
    let cell = match read_cell(simbox) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let p = match read3(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_triples(qs, n, "null qs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n == 0 {
        clear_error();
        return 0;
    }
    if out.is_null() {
        return fail_msg("null out");
    }
    let out = unsafe { slice::from_raw_parts_mut(out, n) };
    match dist2_many(&cell, p, qs, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(e) => fail(e),
    }
}

/// Squared MIC distances for `n` packed pairs `(ps[k], qs[k])`.
///
/// # Safety
///
/// `ps` and `qs` are `n * 3` doubles. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_pairs(
    simbox: *const mi_cell,
    ps: *const f64,
    qs: *const f64,
    n: usize,
    out: *mut f64,
) -> c_int {
    let cell = match read_cell(simbox) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let ps = match packed_triples(ps, n, "null ps") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_triples(qs, n, "null qs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n == 0 {
        clear_error();
        return 0;
    }
    if out.is_null() {
        return fail_msg("null out");
    }
    let out = unsafe { slice::from_raw_parts_mut(out, n) };
    match dist2_pairs(&cell, ps, qs, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(e) => fail(e),
    }
}

/// Orthorhombic wrap of precomputed differences (Highway kernel).
///
/// # Safety
///
/// `dx`, `dy`, `dz`, and `out` are `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_ortho_diffs(
    dx: *const f64,
    dy: *const f64,
    dz: *const f64,
    bx: f64,
    by: f64,
    bz: f64,
    out: *mut f64,
    n: usize,
) -> c_int {
    if n == 0 {
        clear_error();
        return 0;
    }
    if dx.is_null() || dy.is_null() || dz.is_null() || out.is_null() {
        return fail_msg("null diff buffer");
    }
    let dx = unsafe { slice::from_raw_parts(dx, n) };
    let dy = unsafe { slice::from_raw_parts(dy, n) };
    let dz = unsafe { slice::from_raw_parts(dz, n) };
    let out = unsafe { slice::from_raw_parts_mut(out, n) };
    match dist2_ortho_diffs(dx, dy, dz, bx, by, bz, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(e) => fail(e),
    }
}

/// Drop self images and collapse duplicate `(i, j)` rows.
///
/// `pairs` is `n * 2` ints. `out` has room for `n * 2` ints. `out_n`
/// receives the kept pair count.
///
/// # Safety
///
/// `pairs` is `2 * n` readable ints. `out` is `2 * n` writable ints.
/// `out_n` is one writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn mi_reduce_pairs(
    pairs: *const c_int,
    n: usize,
    out: *mut c_int,
    out_n: *mut usize,
) -> c_int {
    if out_n.is_null() {
        return fail_msg("null out_n");
    }
    if n == 0 {
        unsafe {
            *out_n = 0;
        }
        clear_error();
        return 0;
    }
    if pairs.is_null() || out.is_null() {
        return fail_msg("null pair buffer");
    }
    let pairs = unsafe { slice::from_raw_parts(pairs, n * 2) };
    let out = unsafe { slice::from_raw_parts_mut(out, n * 2) };
    let mut kept = 0usize;
    match reduce_pairs_packed(pairs, out, &mut kept) {
        Ok(()) => {
            unsafe {
                *out_n = kept;
            }
            clear_error();
            0
        }
        Err(e) => fail(e),
    }
}
