//! C ABI. Function symbols use the crate-name prefix `minimage_`.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::{Cell as StdCell, RefCell};
use std::ffi::{c_char, c_int, CString};
use std::ops::ControlFlow;
use std::ptr;
use std::slice;

use crate::fused;
use crate::{dist2_ortho_diffs, reduce_pairs_packed, Cell, Error, PartialCell};

/// Periodic parallelepiped.
///
/// Stores lattice vectors `a`, `b`, `c` (columns of `H`) and origin
/// `(ox, oy, oz)` in consistent Cartesian length units.
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
/// an `mi_cell`. seams calls `minimage_dist2` once per pair with the same
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

/// Thread-local last-error string from this thread's most recent `minimage_*`
/// failure.
///
/// Returns a pointer to a NUL-terminated UTF-8 C string, or `NULL` if
/// the last call on this thread succeeded, or if none has failed yet.
/// [`minimage_version`] does not read or write the slot. Distinct threads
/// have independent slots. The pointer is valid until the next `minimage_*`
/// call on this thread. Do not free it.
#[no_mangle]
pub extern "C" fn minimage_last_error() -> *const c_char {
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
pub extern "C" fn minimage_version() -> *const c_char {
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

/// Fill `out` from lattice vectors `a`, `b`, `c` and `origin`, all in
/// consistent length units.
///
/// Returns `0` on success, or `1` on failure (null pointer or singular
/// / non-finite cell; inspect `minimage_last_error()` on this thread).
///
/// # Safety
///
/// `a`, `b`, `c`, `origin`, and `out` are non-null and point at three
/// doubles / one `mi_cell`.
#[no_mangle]
pub unsafe extern "C" fn minimage_cell_from_vectors(
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

/// Fill `out` from LAMMPS dump-bound coordinates `xlo xhi ylo yhi zlo zhi`
/// and tilt factors `xy, xz, yz` in consistent length units.
///
/// Expects tilt-padded dump bounds (`xlo_bound, xhi_bound, ...`).
/// Returns `0` on success, or `1` on failure (`out == NULL` or singular
/// / non-finite cell; inspect `minimage_last_error()` on this thread).
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn minimage_cell_from_lammps(
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

/// Fill `out` from LAMMPS dump bound spans, tilts, and bound origins,
/// all in consistent length units.
///
/// Returns `0` on success, or `1` on failure (`out == NULL` or singular
/// / non-finite cell; inspect `minimage_last_error()` on this thread).
#[no_mangle]
pub extern "C" fn minimage_cell_from_lammps_bounds(
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

/// Fill `out` from an ASE-style row-major 3x3 cell (`[a, b, c]`) in
/// consistent length units. `origin` defaults to zero when `NULL`.
///
/// Returns `0` on success, or `1` on failure (null `rows`/`out` or
/// singular / non-finite cell).
///
/// # Safety
///
/// `rows` is nine readable doubles. `origin`, if non-null, is three
/// readable doubles. `out` is one writable `mi_cell`.
#[no_mangle]
pub unsafe extern "C" fn minimage_cell_from_ase(
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

/// Fill `out` from a CON 3x3 lattice (rows `a`, `b`, `c`) with origin
/// defaulting to zero.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// Same contract as `minimage_cell_from_ase` with a null origin.
#[no_mangle]
pub unsafe extern "C" fn minimage_cell_from_con(rows: *const f64, out: *mut mi_cell) -> c_int {
    unsafe { minimage_cell_from_ase(rows, ptr::null(), out) }
}

/// Fill `out` from CON lengths (`boxl` in length units) and
/// crystallographic angles (`angles_deg` `[alpha, beta, gamma]` in
/// degrees). Origin defaults to zero.
///
/// Returns `0` on success, or `1` on failure (null pointer, non-positive
/// length, or degenerate angles).
///
/// # Safety
///
/// `boxl` and `angles_deg` are three readable doubles. `out` is one
/// writable `mi_cell`.
#[no_mangle]
pub unsafe extern "C" fn minimage_cell_from_con_box(
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

/// Fill `out` from a vesin 3x3 box (rows `a`, `b`, `c`) with origin
/// defaulting to zero.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// Same contract as `minimage_cell_from_con`.
#[no_mangle]
pub unsafe extern "C" fn minimage_cell_from_vesin(
    box_rows: *const f64,
    out: *mut mi_cell,
) -> c_int {
    unsafe { minimage_cell_from_con(box_rows, out) }
}

fn read3(p: *const f64, what: &str) -> Result<[f64; 3], c_int> {
    if p.is_null() {
        return Err(fail_msg(what));
    }
    Ok(unsafe { [*p, *p.add(1), *p.add(2)] })
}

/// Engine minimum-image displacement from `p` to `q` (`wrap(q - p)`)
/// into `dr`, in the cell's length units.
///
/// Returns `0` on success, or `1` on failure (null pointer or invalid
/// `simbox`; inspect `minimage_last_error()` on this thread).
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `p`, `q`, and `dr` are three doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_displacement(
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

/// Euclidean MIC displacement from `p` to `q` into `dr`, in the cell's
/// length units: Smith half-altitude test, else the slicer on the
/// Selling superbasis cached on the calling thread. A tie keeps the
/// engine vector.
///
/// Returns `0` on success, or `1` on failure (null pointer or invalid
/// `simbox`; inspect `minimage_last_error()` on this thread).
///
/// # Safety
///
/// Same contract as `minimage_displacement`.
#[no_mangle]
pub unsafe extern "C" fn minimage_displacement_euclidean(
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

/// Engine wrap of `n` packed difference vectors (`q - p`) into `out`,
/// in the cell's length units. When `n == 0`, returns `0` without
/// dereferencing `diffs` or `out` (after validating `simbox`).
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `diffs` and `out` are `n * 3` doubles when `n > 0`.
#[no_mangle]
pub unsafe extern "C" fn minimage_wrap_many(
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

/// Squared minimum-image distance from `p` to `q` into `out`, in the
/// cell's squared length units.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `p` and `q` are three doubles. `out` is
/// one writable double.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2(
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

/// Squared MIC distances from `p` to `n` packed candidates in `qs`, in
/// squared cell length units. When `n == 0`, returns `0` after
/// validating `simbox` and `p`.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `qs` is `n * 3` doubles. `out` is `n` doubles when `n > 0`.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_many(
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

/// Squared MIC distances for `n` packed pairs `(ps[k], qs[k])`, in
/// squared cell length units. When `n == 0`, returns `0` after
/// validating `simbox`.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `ps` and `qs` are `n * 3` doubles. `out` is `n` doubles when `n > 0`.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_pairs(
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

/// Euclidean nearest-image squared distances from `p` to `n` packed
/// candidates: [`minimage_displacement_euclidean`] per pair, bit for bit, with
/// one superbasis lookup per call and eight pairs per slicer pass.
///
/// # Safety
///
/// `qs` is `n * 3` doubles. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_euclidean_many(
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
    match with_cell(simbox, |cell| {
        crate::batch::euclidean_many(cell, p, qs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// [`minimage_dist2_euclidean_many`] for paired rows.
///
/// # Safety
///
/// `ps` and `qs` are `n * 3` doubles. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_euclidean_pairs(
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
    match with_cell(simbox, |cell| {
        crate::batch::euclidean_pairs(cell, ps, qs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// [`minimage_dist2_euclidean_pairs`] for pairs that persist across frames.
/// `images` is `n` rows of three `int32_t`: the cell-basis lattice shift
/// of each pair's image from the previous call, zeros to start, updated
/// in place. A pair whose stored image still lies in the Voronoi cell
/// skips the search; its distance equals the full search to rounding.
///
/// # Safety
///
/// `ps` and `qs` are `n * 3` doubles, `images` is `n * 3` writable
/// `int32_t`, and `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_euclidean_pairs_warm(
    simbox: *const mi_cell,
    ps: *const f64,
    qs: *const f64,
    n: usize,
    images: *mut i32,
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
    if n > 0 && (out.is_null() || images.is_null()) {
        return fail_msg(if out.is_null() {
            "null out"
        } else {
            "null images"
        });
    }
    let (images, out): (&mut [[i32; 3]], &mut [f64]) = if n == 0 {
        (&mut [], &mut [])
    } else {
        unsafe {
            (
                slice::from_raw_parts_mut(images as *mut [i32; 3], n),
                slice::from_raw_parts_mut(out, n),
            )
        }
    };
    match with_cell(simbox, |cell| {
        crate::batch::euclidean_pairs_warm(cell, ps, qs, images, out)
    }) {
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
pub unsafe extern "C" fn minimage_fixed_many(
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
pub unsafe extern "C" fn minimage_dist2_fixed(
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
pub unsafe extern "C" fn minimage_dist2_many_fixed(
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
pub unsafe extern "C" fn minimage_dist2_pairs_fixed(
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

fn read3_fixed32(p: *const u32, what: &str) -> Result<[u32; 3], c_int> {
    if p.is_null() {
        return Err(fail_msg(what));
    }
    Ok(unsafe { [*p, *p.add(1), *p.add(2)] })
}

fn packed_fixed32<'a>(ptr: *const u32, n: usize, what: &str) -> Result<&'a [[u32; 3]], c_int> {
    if n == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(fail_msg(what));
    }
    // SAFETY: `n * 3` readable integers, viewed as `n` triples.
    Ok(unsafe { slice::from_raw_parts(ptr as *const [u32; 3], n) })
}

/// 32-bit fixed-point fractional coordinates of `n` packed positions:
/// three `uint32_t` per position, each the 64-bit fraction of
/// [`minimage_fixed_many`] rounded to its top 32 bits.
///
/// # Safety
///
/// `rs` is `n * 3` doubles. `out` is `n * 3` writable `uint32_t`.
#[no_mangle]
pub unsafe extern "C" fn minimage_fixed32_many(
    simbox: *const mi_cell,
    rs: *const f64,
    n: usize,
    out: *mut u32,
) -> c_int {
    let rs = match packed_triples(rs, n, "null rs") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if n > 0 && out.is_null() {
        return fail_msg("null out");
    }
    let out: &mut [[u32; 3]] = if n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(out as *mut [u32; 3], n) }
    };
    match with_cell(simbox, |cell| {
        crate::fixed32::fixed32_many(&cell.fold(), rs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// Squared engine-wrap distance between two 32-bit fixed-point positions.
///
/// # Safety
///
/// `a` and `b` are three `uint32_t`. `out` is one writable double.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_fixed32(
    simbox: *const mi_cell,
    a: *const u32,
    b: *const u32,
    out: *mut f64,
) -> c_int {
    let a = match read3_fixed32(a, "null a") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let b = match read3_fixed32(b, "null b") {
        Ok(v) => v,
        Err(e) => return e,
    };
    if out.is_null() {
        return fail_msg("null out");
    }
    match with_cell(simbox, |cell| cell.dist2_fixed32(a, b)) {
        Ok(d2) => {
            unsafe {
                *out = d2;
            }
            0
        }
        Err(e) => e,
    }
}

/// Squared engine-wrap distances from 32-bit fixed-point `p` to `n` packed
/// fixed-point candidates.
///
/// # Safety
///
/// `p` is three `uint32_t`. `qs` is `n * 3` `uint32_t`. `out` is `n`
/// doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_many_fixed32(
    simbox: *const mi_cell,
    p: *const u32,
    qs: *const u32,
    n: usize,
    out: *mut f64,
) -> c_int {
    let p = match read3_fixed32(p, "null p") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_fixed32(qs, n, "null qs") {
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
        crate::fixed32::dist2_many(cell.fixed32_lattice(), p, qs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// Squared engine-wrap distances for `n` packed 32-bit fixed-point pairs.
///
/// # Safety
///
/// `ps` and `qs` are `n * 3` `uint32_t`. `out` is `n` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_pairs_fixed32(
    simbox: *const mi_cell,
    ps: *const u32,
    qs: *const u32,
    n: usize,
    out: *mut f64,
) -> c_int {
    let ps = match packed_fixed32(ps, n, "null ps") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qs = match packed_fixed32(qs, n, "null qs") {
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
        crate::fixed32::dist2_pairs(cell.fixed32_lattice(), ps, qs, out)
    }) {
        Ok(()) => 0,
        Err(e) => e,
    }
}

/// Orthorhombic wrap of precomputed differences (Highway kernel).
///
/// `dx`, `dy`, `dz`, and box lengths `bx`, `by`, `bz` use consistent
/// length units; `out` receives squared lengths in that unit squared.
/// When `n == 0`, returns `0` without inspecting box lengths or pointers.
/// Returns `1` if `n > 0` and any pointer is null or any box length is
/// not strictly positive (or `NaN`).
///
/// # Safety
///
/// `dx`, `dy`, `dz`, and `out` are `n` doubles when `n > 0`.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_ortho_diffs(
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
/// shift `(sx, sy, sz)` on every candidate, in squared length units.
/// Rapaport's bin pair. When `n == 0`, returns `0` after validating `p`
/// and `shift`.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `p` and `shift` are three doubles. `qs` is `n * 3` doubles. `out`
/// is `n` doubles when `n > 0`.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_shifted_many(
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

/// Drop self images (`i == j`) and collapse duplicate `(i, j)` rows.
///
/// `pairs` is `n * 2` ints. `out` has room for `n * 2` ints. `out_n`
/// receives the kept pair count (0 when `n == 0`). Ordered pairs
/// `(i, j)` and `(j, i)` are distinct.
///
/// Returns `0` on success, or `1` on failure (null pointer).
///
/// # Safety
///
/// `pairs` is `2 * n` readable ints and `out` is `2 * n` writable ints
/// when `n > 0`. `out_n` is one writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn minimage_reduce_pairs(
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

/// Cell periodic along the axes of a mask (a slab or a wire). The rows
/// of open axes are carried but ignored.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct mi_pbc_cell {
    /// Lattice rows and origin, as in an `mi_cell`.
    pub cell: mi_cell,
    /// Nonzero when axis a is periodic.
    pub pbc_a: c_int,
    /// Nonzero when axis b is periodic.
    pub pbc_b: c_int,
    /// Nonzero when axis c is periodic.
    pub pbc_c: c_int,
}

impl mi_pbc_cell {
    fn mask(&self) -> [bool; 3] {
        [self.pbc_a != 0, self.pbc_b != 0, self.pbc_c != 0]
    }

    fn to_partial(self) -> Result<PartialCell, Error> {
        PartialCell::from_ase(self.cell.columns(), self.mask())
    }
}

/// The last partial cell built on this thread, keyed like the full cell.
/// A separate slot: the fully periodic path keeps its own state untouched.
type PartialEntry = ([f64; 12], [bool; 3], PartialCell);

struct PartialState {
    cell: RefCell<Option<PartialEntry>>,
}

thread_local! {
    static PARTIAL: PartialState = const {
        PartialState {
            cell: RefCell::new(None),
        }
    };
}

fn with_partial<R>(
    simbox: *const mi_pbc_cell,
    f: impl FnOnce(&PartialCell) -> R,
) -> Result<R, c_int> {
    if simbox.is_null() {
        return Err(fail_msg("null simbox"));
    }
    // SAFETY: `simbox` is a readable `mi_pbc_cell`.
    let raw = unsafe { &*simbox };
    let key = cell_key(&raw.cell);
    let mask = raw.mask();
    PARTIAL.with(|st| {
        if let Some((cached, cached_mask, cell)) = st.cell.borrow().as_ref() {
            if same_key(cached, &key) && *cached_mask == mask {
                let out = f(cell);
                clear_error();
                return Ok(out);
            }
        }
        let cell = raw.to_partial().map_err(fail)?;
        let out = f(&cell);
        *st.cell.borrow_mut() = Some((key, mask, cell));
        clear_error();
        Ok(out)
    })
}

/// Fill `out` from ASE-style rows, a periodicity mask and an origin.
/// `pbc` holds three ints (`0` open, nonzero periodic); `origin`
/// defaults to zero when `NULL`. Rows use consistent length units;
/// rows of periodic axes must be finite and linearly independent, while
/// rows of open axes may be zero. All three periodic matches
/// `minimage_cell_from_ase`.
///
/// Returns `0` on success, or `1` on failure (null pointer or invalid
/// periodic rows; inspect `minimage_last_error()` on this thread).
///
/// # Safety
///
/// `rows` is nine readable doubles, `pbc` three readable ints, `origin`
/// NULL or three readable doubles, and `out` one writable `mi_pbc_cell`.
#[no_mangle]
pub unsafe extern "C" fn minimage_pbc_cell_from_ase(
    rows: *const f64,
    pbc: *const c_int,
    origin: *const f64,
    out: *mut mi_pbc_cell,
) -> c_int {
    if rows.is_null() {
        return fail_msg("null ASE cell");
    }
    if pbc.is_null() {
        return fail_msg("null periodicity mask");
    }
    if out.is_null() {
        return fail_msg("null cell output");
    }
    // SAFETY: nine readable doubles, three readable ints.
    let (r, mask) = unsafe {
        (
            [
                *rows,
                *rows.add(1),
                *rows.add(2),
                *rows.add(3),
                *rows.add(4),
                *rows.add(5),
                *rows.add(6),
                *rows.add(7),
                *rows.add(8),
            ],
            [*pbc, *pbc.add(1), *pbc.add(2)],
        )
    };
    let o = if origin.is_null() {
        [0.0; 3]
    } else {
        // SAFETY: three readable doubles.
        unsafe { [*origin, *origin.add(1), *origin.add(2)] }
    };
    let raw = mi_pbc_cell {
        cell: mi_cell {
            ax: r[0],
            ay: r[1],
            az: r[2],
            bx: r[3],
            by: r[4],
            bz: r[5],
            cx: r[6],
            cy: r[7],
            cz: r[8],
            ox: o[0],
            oy: o[1],
            oz: o[2],
        },
        pbc_a: c_int::from(mask[0] != 0),
        pbc_b: c_int::from(mask[1] != 0),
        pbc_c: c_int::from(mask[2] != 0),
    };
    match raw.to_partial() {
        Ok(_) => {
            clear_error();
            // SAFETY: `out` is a valid `mi_pbc_cell`.
            unsafe {
                *out = raw;
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Nearest-image displacement from `p` to `q` over the periodic axes of
/// `simbox` into `dr`, in cell length units. Open axes take no image.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `simbox` is one `mi_pbc_cell`. `p`, `q`, and `dr` are three doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_displacement_euclidean_pbc(
    simbox: *const mi_pbc_cell,
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
    match with_partial(simbox, |cell| cell.displacement_euclidean(p, q)) {
        Ok(v) => {
            // SAFETY: `dr` is three writable doubles.
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

/// Squared length of `minimage_displacement_euclidean_pbc` into `out`,
/// in squared cell length units.
///
/// Returns `0` on success, or `1` on failure.
///
/// # Safety
///
/// `simbox` is one `mi_pbc_cell`. `p` and `q` are three doubles; `out` is
/// one writable double.
#[no_mangle]
pub unsafe extern "C" fn minimage_dist2_euclidean_pbc(
    simbox: *const mi_pbc_cell,
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
    match with_partial(simbox, |cell| cell.dist2_euclidean(p, q)) {
        Ok(d2) => {
            // SAFETY: `out` is one writable double.
            unsafe {
                *out = d2;
            }
            0
        }
        Err(e) => e,
    }
}

/// Squared-distance filter of candidate pairs `(ii[k], jj[k], shift[k])`.
///
/// `xyz` uses the cell's length units and `cut2` is the squared cutoff
/// in those units squared. Keeps rows with squared distance strictly
/// below `cut2` into caller buffers and writes the count to `kept`. Each
/// output buffer holds at least `n` entries. `out_vector` may be `NULL`;
/// when non-null, it receives `q + H S - p` in cell length units.
///
/// Returns `0` on success, or `1` on failure (null required pointer,
/// invalid `simbox`, or an atom index `>= n_atoms`).
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `xyz` is `n_atoms * 3` doubles. `ii`, `jj`
/// and `shift` describe `n` candidates (`shift` is `n * 3` ints). The
/// output pointers are writable for `n` rows.
#[no_mangle]
pub unsafe extern "C" fn minimage_filter_shifted(
    simbox: *const mi_cell,
    xyz: *const f64,
    n_atoms: usize,
    ii: *const u32,
    jj: *const u32,
    shift: *const i32,
    n: usize,
    cut2: f64,
    out_i: *mut u32,
    out_j: *mut u32,
    out_shift: *mut i32,
    out_dist2: *mut f64,
    out_vector: *mut f64,
    kept: *mut usize,
) -> c_int {
    if xyz.is_null() || ii.is_null() || jj.is_null() || shift.is_null() || kept.is_null() {
        return fail_msg("null filter input");
    }
    if out_i.is_null() || out_j.is_null() || out_shift.is_null() || out_dist2.is_null() {
        return fail_msg("null filter output");
    }
    let cell = match with_cell(simbox, |c| *c) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let pts: Vec<[f64; 3]> =
        unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n_atoms).to_vec() };
    let ii = unsafe { std::slice::from_raw_parts(ii, n) };
    let jj = unsafe { std::slice::from_raw_parts(jj, n) };
    let shift_raw = unsafe { std::slice::from_raw_parts(shift, n.saturating_mul(3)) };
    if shift_raw.len() != n.saturating_mul(3) {
        return fail(Error::BufferSize);
    }
    let mut sh = Vec::with_capacity(n);
    for k in 0..n {
        sh.push([shift_raw[3 * k], shift_raw[3 * k + 1], shift_raw[3 * k + 2]]);
    }
    let want = !out_vector.is_null();
    let mut out = crate::FilterOut::default();
    if let Err(e) = crate::filter_shifted(&cell, &pts, ii, jj, &sh, cut2, want, &mut out) {
        return fail(e);
    }
    let m = out.i.len();
    unsafe {
        std::ptr::copy_nonoverlapping(out.i.as_ptr(), out_i, m);
        std::ptr::copy_nonoverlapping(out.j.as_ptr(), out_j, m);
        for (k, s) in out.shift.iter().enumerate() {
            *out_shift.add(3 * k) = s[0];
            *out_shift.add(3 * k + 1) = s[1];
            *out_shift.add(3 * k + 2) = s[2];
        }
        std::ptr::copy_nonoverlapping(out.dist2.as_ptr(), out_dist2, m);
        if want {
            for (k, v) in out.vector.iter().enumerate() {
                *out_vector.add(3 * k) = v[0];
                *out_vector.add(3 * k + 1) = v[1];
                *out_vector.add(3 * k + 2) = v[2];
            }
        }
        *kept = m;
    }
    0
}

/// Ring-polymer centroid in the cell's length units.
///
/// `xyz` is bead-major `(beads * atoms, 3)`; `out` receives `atoms`
/// positions wrapped into the primary cell. Returns `0` on success, or
/// `1` on failure (`beads == 0`, `atoms == 0`, null pointer, integer
/// overflow in `beads * atoms`, or invalid `simbox`).
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `xyz` holds `beads * atoms` points and
/// `out` holds `atoms` points.
#[no_mangle]
pub unsafe extern "C" fn minimage_ring_centroid(
    simbox: *const mi_cell,
    xyz: *const f64,
    beads: usize,
    atoms: usize,
    out: *mut f64,
) -> c_int {
    if xyz.is_null() || out.is_null() {
        return fail_msg("null centroid buffer");
    }
    let cell = match with_cell(simbox, |c| *c) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let n = match beads.checked_mul(atoms) {
        Some(n) => n,
        None => return fail(Error::BufferSize),
    };
    let pts = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) };
    match crate::ring_centroid(&cell, beads, pts) {
        Ok(rows) => {
            unsafe {
                for (k, p) in rows.iter().enumerate() {
                    *out.add(3 * k) = p[0];
                    *out.add(3 * k + 1) = p[1];
                    *out.add(3 * k + 2) = p[2];
                }
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// Perpendicular altitude of the face opposite lattice axis `axis`,
/// or `0.0` if `cell` is singular or non-finite.
///
/// # Safety
///
/// `cell` is nine readable doubles (ASE row-major `[a, b, c]`).
#[no_mangle]
pub unsafe extern "C" fn minimage_face_spacing(cell: *const f64, axis: c_int) -> f64 {
    if cell.is_null() || !(0..3).contains(&axis) {
        return 0.0;
    }
    let s = unsafe { slice::from_raw_parts(cell, 9) };
    let u = [
        s[3 * ((axis as usize + 1) % 3)],
        s[3 * ((axis as usize + 1) % 3) + 1],
        s[3 * ((axis as usize + 1) % 3) + 2],
    ];
    let v = [
        s[3 * ((axis as usize + 2) % 3)],
        s[3 * ((axis as usize + 2) % 3) + 1],
        s[3 * ((axis as usize + 2) % 3) + 2],
    ];
    let cx = u[1] * v[2] - u[2] * v[1];
    let cy = u[2] * v[0] - u[0] * v[2];
    let cz = u[0] * v[1] - u[1] * v[0];
    let area = (cx * cx + cy * cy + cz * cz).sqrt();
    let vol = (s[0] * (s[4] * s[8] - s[5] * s[7]) - s[1] * (s[3] * s[8] - s[5] * s[6])
        + s[2] * (s[3] * s[7] - s[4] * s[6]))
        .abs();
    if !area.is_finite() || !vol.is_finite() || area <= 0.0 || vol <= 0.0 {
        return 0.0;
    }
    vol / area
}

/// Nonzero when `2 * cutoff` is at least the perpendicular face spacing
/// on any periodic axis of the ASE row-major `cell`.
///
/// # Safety
///
/// `cell` is nine readable doubles. `pbc` is `NULL` or three readable ints.
#[no_mangle]
pub unsafe extern "C" fn minimage_cutoff_exceeds_half_width(
    cell: *const f64,
    pbc: *const c_int,
    cutoff: f64,
) -> c_int {
    if cell.is_null() {
        return 0;
    }
    let periodic = if pbc.is_null() {
        [true, true, true]
    } else {
        let s = unsafe { [*pbc, *pbc.add(1), *pbc.add(2)] };
        [s[0] != 0, s[1] != 0, s[2] != 0]
    };
    for (axis, &is_pbc) in periodic.iter().enumerate() {
        if !is_pbc {
            continue;
        }
        let spacing = unsafe { minimage_face_spacing(cell, axis as c_int) };
        if !cutoff.is_finite() || !spacing.is_finite() || 2.0 * cutoff >= spacing {
            return 1;
        }
    }
    0
}

/// One periodic image of pair `(i, j)` within a cutoff.
///
/// `d` is `(xyz[i] - xyz[j]) - H shift`, and `dist2` is `|d|^2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct mi_image_pair {
    /// Source atom index.
    pub i: i32,
    /// Target atom index.
    pub j: i32,
    /// Integer lattice shift `(sa, sb, sc)` subtracted from `xyz[i] - xyz[j]`.
    pub shift: [i32; 3],
    /// Cartesian separation `(xyz[i] - xyz[j]) - H shift`.
    pub d: [f64; 3],
    /// Squared distance `d[0]^2 + d[1]^2 + d[2]^2`.
    pub dist2: f64,
}

/// Enumerate every periodic image of `n` points strictly inside `cutoff`.
///
/// `pbc` is three ints (nonzero for a periodic axis) or `NULL` for
/// fully periodic. `half` nonzero visits `i <= j` and restricts
/// self-images (`i == j`) to the positive half-space.
///
/// Pass `out == NULL` to query the count into `out_count` (returns `0`).
/// Otherwise `out` holds `cap` rows. If `cap` is smaller than the
/// needed row count, writes the needed count to `out_count`, leaves
/// `out` untouched, and returns `1`.
///
/// Returns `0` on success, or `1` on failure (null required pointer,
/// invalid cell or cutoff, extent or shift count past the cap,
/// non-finite separation, or shift index overflow; inspect
/// `minimage_last_error()` on this thread).
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `pbc` is `NULL` or three ints. `xyz` is
/// `n * 3` doubles when `n > 0`. `out_count` is one writable `size_t`.
/// When `out` is non-null, it is writable for `cap` `mi_image_pair` rows.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn minimage_pairs_within_images(
    simbox: *const mi_cell,
    pbc: *const c_int,
    xyz: *const f64,
    n: usize,
    cutoff: f64,
    half: c_int,
    out: *mut mi_image_pair,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if out_count.is_null() {
        return fail_msg("null out_count");
    }
    let pts = match packed_triples(xyz, n, "null xyz") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let periodic = if pbc.is_null() {
        [true, true, true]
    } else {
        let s = unsafe { [*pbc, *pbc.add(1), *pbc.add(2)] };
        [s[0] != 0, s[1] != 0, s[2] != 0]
    };
    let cell = match with_cell(simbox, |c| *c) {
        Ok(c) => c,
        Err(e) => return e,
    };
    if out.is_null() {
        match crate::visit_pairs_within_images(&cell, periodic, pts, cutoff, half != 0, |_| {}) {
            Ok(count) => {
                unsafe {
                    *out_count = count;
                }
                clear_error();
                0
            }
            Err(e) => fail(e),
        }
    } else {
        // Rows are staged so that a short buffer stays untouched, as
        // documented above.
        let mut rows = Vec::new();
        match crate::visit_pairs_within_images(&cell, periodic, pts, cutoff, half != 0, |r| {
            if rows.len() < cap {
                rows.push(mi_image_pair {
                    i: r.i,
                    j: r.j,
                    shift: r.shift,
                    d: r.d,
                    dist2: r.dist2,
                });
            }
        }) {
            Ok(count) => {
                unsafe {
                    *out_count = count;
                }
                if count > cap {
                    return fail_msg("image pair buffer is shorter than the pair count");
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(rows.as_ptr(), out, count);
                }
                clear_error();
                0
            }
            Err(e) => fail(e),
        }
    }
}

/// Callback invoked for each periodic image pair by
/// [`minimage_visit_pairs_within_images`].
///
/// Return `0` to continue. A nonzero return stops the visit after the
/// current pair; the call still succeeds and `out_count` counts the pairs
/// delivered, including the one that stopped it. The callback must not
/// unwind (throw or `longjmp`) out of the call.
#[allow(non_camel_case_types)]
pub type mi_image_pair_visitor = Option<
    unsafe extern "C" fn(userdata: *mut std::ffi::c_void, pair: *const mi_image_pair) -> c_int,
>;

/// Visit every periodic image of `n` points strictly inside `cutoff` in a
/// single pass without allocating a pair buffer.
///
/// `pbc` is three ints (nonzero for a periodic axis) or `NULL` for
/// fully periodic. `half` nonzero visits `i <= j` and restricts
/// self-images (`i == j`) to the positive half-space. When `visit` is
/// non-null, it is called with `(userdata, &pair)` for each emitted image
/// and a nonzero return value stops the visit early; when `visit` is
/// `NULL`, only the total count is computed. `out_count`, if non-null,
/// receives the number of emitted pair images (the number delivered when
/// the visit was stopped early).
///
/// Returns `0` on success, or `1` on failure (null `simbox` or `xyz` when
/// `n > 0`, invalid cell or cutoff, extent or shift count past the cap,
/// non-finite separation, or shift index overflow; inspect
/// `minimage_last_error()` on this thread).
///
/// # Safety
///
/// `simbox` is one `mi_cell`. `pbc` is `NULL` or three ints. `xyz` is
/// `n * 3` doubles when `n > 0`. `out_count` is `NULL` or one writable
/// `size_t`. `visit`, if non-null, is safe to call with `userdata` and a
/// valid `*const mi_image_pair` for the duration of the call, and must
/// return normally rather than unwinding through this function.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn minimage_visit_pairs_within_images(
    simbox: *const mi_cell,
    pbc: *const c_int,
    xyz: *const f64,
    n: usize,
    cutoff: f64,
    half: c_int,
    visit: mi_image_pair_visitor,
    userdata: *mut std::ffi::c_void,
    out_count: *mut usize,
) -> c_int {
    let pts = match packed_triples(xyz, n, "null xyz") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let periodic = if pbc.is_null() {
        [true, true, true]
    } else {
        let s = unsafe { [*pbc, *pbc.add(1), *pbc.add(2)] };
        [s[0] != 0, s[1] != 0, s[2] != 0]
    };
    let cell = match with_cell(simbox, |c| *c) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let res = match visit {
        Some(cb) => crate::images::try_visit_pairs_within_images(
            &cell,
            periodic,
            pts,
            cutoff,
            half != 0,
            |r| {
                let raw = mi_image_pair {
                    i: r.i,
                    j: r.j,
                    shift: r.shift,
                    d: r.d,
                    dist2: r.dist2,
                };
                if unsafe { cb(userdata, &raw) } != 0 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        ),
        None => crate::visit_pairs_within_images(&cell, periodic, pts, cutoff, half != 0, |_| {}),
    };
    match res {
        Ok(count) => {
            if !out_count.is_null() {
                unsafe {
                    *out_count = count;
                }
            }
            clear_error();
            0
        }
        Err(e) => fail(e),
    }
}

#[cfg(feature = "capi-compat-names")]
mod compat_names {
    use super::*;

    macro_rules! safe_alias {
        ($alias:ident => $target:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
            #[deprecated(note = "use the corresponding minimage_* symbol")]
            #[no_mangle]
            extern "C" fn $alias($($arg: $ty),*) -> $ret {
                $target($($arg),*)
            }
        };
    }

    macro_rules! unsafe_alias {
        ($alias:ident => $target:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
            #[deprecated(note = "use the corresponding minimage_* symbol")]
            #[no_mangle]
            unsafe extern "C" fn $alias($($arg: $ty),*) -> $ret {
                unsafe { $target($($arg),*) }
            }
        };
    }

    safe_alias!(mi_last_error => minimage_last_error() -> *const c_char);
    safe_alias!(mi_version => minimage_version() -> *const c_char);
    unsafe_alias!(
        mi_cell_from_vectors => minimage_cell_from_vectors(
            a: *const f64,
            b: *const f64,
            c: *const f64,
            origin: *const f64,
            out: *mut mi_cell
        ) -> c_int
    );
    safe_alias!(
        mi_cell_from_lammps => minimage_cell_from_lammps(
            xlo: f64,
            xhi: f64,
            ylo: f64,
            yhi: f64,
            zlo: f64,
            zhi: f64,
            xy: f64,
            xz: f64,
            yz: f64,
            out: *mut mi_cell
        ) -> c_int
    );
    safe_alias!(
        mi_cell_from_lammps_bounds => minimage_cell_from_lammps_bounds(
            xspan: f64,
            yspan: f64,
            zspan: f64,
            xy: f64,
            xz: f64,
            yz: f64,
            xlo_b: f64,
            ylo_b: f64,
            zlo_b: f64,
            out: *mut mi_cell
        ) -> c_int
    );
    unsafe_alias!(
        mi_cell_from_ase => minimage_cell_from_ase(
            rows: *const f64,
            origin: *const f64,
            out: *mut mi_cell
        ) -> c_int
    );
    unsafe_alias!(
        mi_cell_from_con => minimage_cell_from_con(
            rows: *const f64,
            out: *mut mi_cell
        ) -> c_int
    );
    unsafe_alias!(
        mi_cell_from_con_box => minimage_cell_from_con_box(
            boxl: *const f64,
            angles_deg: *const f64,
            out: *mut mi_cell
        ) -> c_int
    );
    unsafe_alias!(
        mi_cell_from_vesin => minimage_cell_from_vesin(
            box_rows: *const f64,
            out: *mut mi_cell
        ) -> c_int
    );
    unsafe_alias!(
        mi_displacement => minimage_displacement(
            simbox: *const mi_cell,
            p: *const f64,
            q: *const f64,
            dr: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_displacement_euclidean => minimage_displacement_euclidean(
            simbox: *const mi_cell,
            p: *const f64,
            q: *const f64,
            dr: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_wrap_many => minimage_wrap_many(
            simbox: *const mi_cell,
            diffs: *const f64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2 => minimage_dist2(
            simbox: *const mi_cell,
            p: *const f64,
            q: *const f64,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_many => minimage_dist2_many(
            simbox: *const mi_cell,
            p: *const f64,
            qs: *const f64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_pairs => minimage_dist2_pairs(
            simbox: *const mi_cell,
            ps: *const f64,
            qs: *const f64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_euclidean_many => minimage_dist2_euclidean_many(
            simbox: *const mi_cell,
            p: *const f64,
            qs: *const f64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_euclidean_pairs => minimage_dist2_euclidean_pairs(
            simbox: *const mi_cell,
            ps: *const f64,
            qs: *const f64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_euclidean_pairs_warm => minimage_dist2_euclidean_pairs_warm(
            simbox: *const mi_cell,
            ps: *const f64,
            qs: *const f64,
            n: usize,
            images: *mut i32,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_fixed_many => minimage_fixed_many(
            simbox: *const mi_cell,
            rs: *const f64,
            n: usize,
            out: *mut u64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_fixed => minimage_dist2_fixed(
            simbox: *const mi_cell,
            a: *const u64,
            b: *const u64,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_many_fixed => minimage_dist2_many_fixed(
            simbox: *const mi_cell,
            p: *const u64,
            qs: *const u64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_pairs_fixed => minimage_dist2_pairs_fixed(
            simbox: *const mi_cell,
            ps: *const u64,
            qs: *const u64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_fixed32_many => minimage_fixed32_many(
            simbox: *const mi_cell,
            rs: *const f64,
            n: usize,
            out: *mut u32
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_fixed32 => minimage_dist2_fixed32(
            simbox: *const mi_cell,
            a: *const u32,
            b: *const u32,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_many_fixed32 => minimage_dist2_many_fixed32(
            simbox: *const mi_cell,
            p: *const u32,
            qs: *const u32,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_pairs_fixed32 => minimage_dist2_pairs_fixed32(
            simbox: *const mi_cell,
            ps: *const u32,
            qs: *const u32,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_ortho_diffs => minimage_dist2_ortho_diffs(
            dx: *const f64,
            dy: *const f64,
            dz: *const f64,
            bx: f64,
            by: f64,
            bz: f64,
            out: *mut f64,
            n: usize
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_shifted_many => minimage_dist2_shifted_many(
            p: *const f64,
            qs: *const f64,
            shift: *const f64,
            n: usize,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_reduce_pairs => minimage_reduce_pairs(
            pairs: *const c_int,
            n: usize,
            out: *mut c_int,
            out_n: *mut usize
        ) -> c_int
    );
    unsafe_alias!(
        mi_pbc_cell_from_ase => minimage_pbc_cell_from_ase(
            rows: *const f64,
            pbc: *const c_int,
            origin: *const f64,
            out: *mut mi_pbc_cell
        ) -> c_int
    );
    unsafe_alias!(
        mi_displacement_euclidean_pbc => minimage_displacement_euclidean_pbc(
            simbox: *const mi_pbc_cell,
            p: *const f64,
            q: *const f64,
            dr: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_dist2_euclidean_pbc => minimage_dist2_euclidean_pbc(
            simbox: *const mi_pbc_cell,
            p: *const f64,
            q: *const f64,
            out: *mut f64
        ) -> c_int
    );
    unsafe_alias!(
        mi_filter_shifted => minimage_filter_shifted(
            simbox: *const mi_cell,
            xyz: *const f64,
            n_atoms: usize,
            ii: *const u32,
            jj: *const u32,
            shift: *const i32,
            n: usize,
            cut2: f64,
            out_i: *mut u32,
            out_j: *mut u32,
            out_shift: *mut i32,
            out_dist2: *mut f64,
            out_vector: *mut f64,
            kept: *mut usize
        ) -> c_int
    );
    unsafe_alias!(
        mi_ring_centroid => minimage_ring_centroid(
            simbox: *const mi_cell,
            xyz: *const f64,
            beads: usize,
            atoms: usize,
            out: *mut f64
        ) -> c_int
    );
}
