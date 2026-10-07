//! C ABI. Prefix `mi_`. Caller owns every buffer.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::{Cell as StdCell, RefCell};
use std::ffi::{c_char, c_int, CString};
use std::ptr;
use std::slice;

use crate::fused;
use crate::{dist2_ortho_diffs, reduce_pairs_packed, Cell, Error};

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
    fn columns(&self) -> [[f64; 3]; 3] {
        [
            [self.ax, self.ay, self.az],
            [self.bx, self.by, self.bz],
            [self.cx, self.cy, self.cz],
        ]
    }

    /// Engine cell. Selling is not part of construction.
    fn to_cell(self) -> Result<Cell, Error> {
        let h = self.columns();
        Cell::from_vectors(h[0], h[1], h[2], [self.ox, self.oy, self.oz])
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

/// One thread's ABI state: the last error and the last cell built from
/// an `mi_cell`. seams calls `mi_dist2` once per pair with the same
/// twelve doubles, so one lookup finds the inverse and the frame.
struct State {
    error: RefCell<Option<CString>>,
    /// `error` holds a message. A success clears it only when set.
    failed: StdCell<bool>,
    cell: RefCell<Option<([f64; 12], Cell)>>,
}

thread_local! {
    static STATE: State = const {
        State {
            error: RefCell::new(None),
            failed: StdCell::new(false),
            cell: RefCell::new(None),
        }
    };
}

fn cell_key(raw: &mi_cell) -> [f64; 12] {
    [
        raw.ax, raw.ay, raw.az, raw.bx, raw.by, raw.bz, raw.cx, raw.cy, raw.cz, raw.ox, raw.oy,
        raw.oz,
    ]
}

/// Twelve components equal, without an early exit, so the hit test is a
/// few packed compares.
#[inline(always)]
fn same_key(a: &[f64; 12], b: &[f64; 12]) -> bool {
    let mut same = true;
    for (x, y) in a.iter().zip(b) {
        same &= x == y;
    }
    same
}

fn set_error(msg: &str) {
    let cstr = CString::new(msg).unwrap_or_else(|_| {
        CString::new("error message contained NUL").expect("fallback has no NUL")
    });
    STATE.with(|st| {
        *st.error.borrow_mut() = Some(cstr);
        st.failed.set(true);
    });
}

#[inline]
fn clear_error() {
    STATE.with(clear_in);
}

#[inline(always)]
fn clear_in(st: &State) {
    if st.failed.get() {
        st.error.borrow_mut().take();
        st.failed.set(false);
    }
}

fn fail(err: Error) -> c_int {
    set_error(&err.to_string());
    1
}

fn fail_msg(msg: &str) -> c_int {
    set_error(msg);
    1
}

/// Run `f` on the engine cell for `simbox`, built once per distinct
/// twelve doubles on this thread, and clear the error slot on success.
#[inline(always)]
fn with_cell<R>(simbox: *const mi_cell, f: impl FnOnce(&Cell) -> R) -> Result<R, c_int> {
    if simbox.is_null() {
        return Err(fail_msg("null simbox"));
    }
    // SAFETY: `simbox` is a readable `mi_cell`.
    let raw = unsafe { &*simbox };
    let key = cell_key(raw);
    STATE.with(|st| {
        if let Some((cached, cell)) = st.cell.borrow().as_ref() {
            if same_key(cached, &key) {
                let out = f(cell);
                clear_in(st);
                return Ok(out);
            }
        }
        let cell = raw.to_cell().map_err(fail)?;
        let out = f(&cell);
        *st.cell.borrow_mut() = Some((key, cell));
        clear_in(st);
        Ok(out)
    })
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
    STATE.with(|st| {
        st.error
            .borrow()
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
#[allow(clippy::too_many_arguments)]
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
    let (h, origin) =
        crate::cell::dump_bounds_to_h(xhi - xlo, yhi - ylo, zhi - zlo, xy, xz, yz, xlo, ylo, zlo);
    write_cell(Cell::from_vectors(h[0], h[1], h[2], origin), out)
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
    let (h, origin) =
        crate::cell::dump_bounds_to_h(xspan, yspan, zspan, xy, xz, yz, xlo_b, ylo_b, zlo_b);
    write_cell(Cell::from_vectors(h[0], h[1], h[2], origin), out)
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
        Cell::from_vectors(rows[0], rows[1], rows[2], [0.0, 0.0, 0.0])
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
    write_cell(
        Cell::from_con_box(boxl, angles)
            .map(|c| Cell::from_vectors(c.a(), c.b(), c.c(), c.origin()))
            .and_then(|r| r),
        out,
    )
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
    match with_cell(simbox, |cell| cell.displacement(p, q)) {
        Ok(v) => {
            unsafe {
                *dr = v[0];
                *dr.add(1) = v[1];
                *dr.add(2) = v[2];
            }
            0
        }
        Err(e) => e,
    }
}

/// Euclidean MIC: Smith half-altitude test, else the slicer on the
/// Selling superbasis, into `dr`.
///
/// # Safety
///
/// Same contract as [`mi_displacement`].
#[no_mangle]
pub unsafe extern "C" fn mi_displacement_euclidean(
    simbox: *const mi_cell,
    p: *const f64,
    q: *const f64,
    dr: *mut f64,
) -> c_int {
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
    match with_cell(simbox, |cell| cell.displacement_euclidean(p, q)) {
        Ok(v) => {
            unsafe {
                *dr = v[0];
                *dr.add(1) = v[1];
                *dr.add(2) = v[2];
            }
            0
        }
        Err(e) => e,
    }
}

/// Engine wrap of `n` packed difference vectors into `out`.
///
/// # Safety
///
/// `diffs` and `out` are `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_wrap_many(
    simbox: *const mi_cell,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> c_int {
    let diffs = match packed_triples(diffs, n, "null diffs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [[f64; 3]] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out as *mut [f64; 3], n) }
    };
    match with_cell(simbox, |cell| fused::wrap_many(cell.frame(), diffs, out)) {
        Ok(()) => 0,
        Err(e) => e,
    }
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
    match with_cell(simbox, |cell| cell.dist2(p, q)) {
        Ok(d2) => {
            unsafe {
                *out = d2;
            }
            0
        }
        Err(e) => e,
    }
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
    let p = match read3(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_triples(qs, n, "null qs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [f64] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out, n) }
    };
    match with_cell(simbox, |cell| fused::dist2_many(cell.frame(), p, qs, out)) {
        Ok(()) => 0,
        Err(e) => e,
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
    let ps = match packed_triples(ps, n, "null ps") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_triples(qs, n, "null qs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [f64] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out, n) }
    };
    match with_cell(simbox, |cell| fused::dist2_pairs(cell.frame(), ps, qs, out)) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

fn read3_fixed(p: *const u64, what: &str) -> Result<[u64; 3], c_int> {
    if p.is_null() {
        return Err(fail_msg(what));
    }
    Ok(unsafe { [*p, *p.add(1), *p.add(2)] })
}

fn packed_fixed<'a>(ptr: *const u64, n: usize, what: &str) -> Result<&'a [[u64; 3]], c_int> {
    if n == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(fail_msg(what));
    }
    // SAFETY: `n * 3` readable integers, viewed as `n` triples.
    Ok(unsafe { slice::from_raw_parts(ptr as *const [u64; 3], n) })
}

/// Fixed-point fractional coordinates of `n` packed positions: three
/// `uint64_t` per position, each fraction as `round(s * 2^52) * 2^12`.
///
/// # Safety
///
/// `rs` is `n * 3` doubles. `out` is `n * 3` writable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn mi_fixed_many(
    simbox: *const mi_cell,
    rs: *const f64,
    n: usize,
    out: *mut u64,
) -> c_int {
    let rs = match packed_triples(rs, n, "null rs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [[u64; 3]] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out as *mut [u64; 3], n) }
    };
    match with_cell(simbox, |cell| {
        crate::fixed::fixed_many(&cell.fold(), rs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// Squared engine-wrap distance between two fixed-point positions.
///
/// # Safety
///
/// `a` and `b` are three `uint64_t`. `out` is one writable double.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_fixed(
    simbox: *const mi_cell,
    a: *const u64,
    b: *const u64,
    out: *mut f64,
) -> c_int {
    let a = match read3_fixed(a, "null a") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let b = match read3_fixed(b, "null b") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if out.is_null() {
        return fail_msg("null out");
    }
    match with_cell(simbox, |cell| cell.dist2_fixed(a, b)) {
        Ok(d2) => {
            unsafe {
                *out = d2;
            }
            0
        }
        Err(e) => e,
    }
}

/// Squared engine-wrap distances from fixed-point `p` to `n` packed
/// fixed-point candidates.
///
/// # Safety
///
/// `p` is three `uint64_t`. `qs` is `n * 3` `uint64_t`. `out` is `n`
/// doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_many_fixed(
    simbox: *const mi_cell,
    p: *const u64,
    qs: *const u64,
    n: usize,
    out: *mut f64,
) -> c_int {
    let p = match read3_fixed(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_fixed(qs, n, "null qs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [f64] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out, n) }
    };
    match with_cell(simbox, |cell| {
        crate::fixed::dist2_many(cell.fixed_lattice(), p, qs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// Squared engine-wrap distances for `n` packed fixed-point pairs.
///
/// # Safety
///
/// `ps` and `qs` are `n * 3` `uint64_t`. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_pairs_fixed(
    simbox: *const mi_cell,
    ps: *const u64,
    qs: *const u64,
    n: usize,
    out: *mut f64,
) -> c_int {
    let ps = match packed_fixed(ps, n, "null ps") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_fixed(qs, n, "null qs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [f64] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out, n) }
    };
    match with_cell(simbox, |cell| {
        crate::fixed::dist2_pairs(cell.fixed_lattice(), ps, qs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
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

/// Squared distances from `p` to `n` points in `qs`, plus one lattice
/// shift `(sx, sy, sz)` on every candidate. Rapaport's bin pair.
///
/// # Safety
///
/// `p` is three doubles. `qs` is `n * 3` doubles. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn mi_dist2_shifted_many(
    p: *const f64,
    qs: *const f64,
    shift: *const f64,
    n: usize,
    out: *mut f64,
) -> c_int {
    let p = match read3(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let shift = match read3(shift, "null shift") {
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
    crate::simd::dist2_shifted_many(p, qs, shift, out);
    clear_error();
    0
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
