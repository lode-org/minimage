//! The C ABI a host negotiates at run time.
//!
//! A host that loads the library by path resolves three symbols by name:
//! `minimage_abi_stamp`, `minimage_abi_compatible`, and
//! `minimage_ops_get`. Everything else comes from the operations table,
//! where a null member is an operation this library does not have.
//!
//! Status word, for every function here that returns `int32_t`: `0` is
//! success, a positive value is a failure the caller can recover from,
//! and a negative value is a misuse of the interface or a fault inside
//! the library. Each cause has its own stable `MINIMAGE_STATUS_*` code,
//! so a host branches on the code and shows `minimage_last_error()` to
//! a person.
//!
//! Layout: `minimage_abi` is frozen. Every other struct starts with a
//! `uint32_t size` and grows only at its end, so members keep their
//! offsets within one layout revision.
//!
//! Error string: `minimage_last_error()` is valid until the next
//! `minimage_*` call on the same thread.
//!
//! Threads: every function may be called from any thread at any time. A
//! lattice is immutable once built, so any number of threads may read
//! one handle at once; `minimage_lattice_free` must not run while
//! another thread still uses that handle.

use std::ffi::c_char;
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use super::{clear_error, minimage_last_error, minimage_version, set_error};
use crate::fused::{self, Frame};
use crate::kernel::{cross, dot, norm};
use crate::{Cell, Error, PartialCell};

/// Major version of the negotiated C ABI. A library serves a host
/// compiled against the same major.
pub const MINIMAGE_ABI_MAJOR: u32 = 1;

/// Minor version of the negotiated C ABI. It grows when a capability or
/// an operation is added, and is informational: a host learns what a
/// library has from `minimage_caps` and from the null members of
/// `minimage_ops`, not from this number.
pub const MINIMAGE_ABI_MINOR: u32 = 0;

/// Revision of the frozen layouts: `minimage_abi`, the leading
/// `uint32_t size` of every other struct, and the offsets of the
/// members those structs already have. A library serves a host compiled
/// against the same revision.
pub const MINIMAGE_ABI_LAYOUT_REVISION: u32 = 1;

/// Success.
pub const MINIMAGE_STATUS_OK: i32 = 0;

/// A failure with no code of its own. The entry points that predate the
/// negotiated ABI return this for every failure; the functions of the
/// operations table never do.
pub const MINIMAGE_STATUS_ERROR: i32 = 1;

/// The cell is not a lattice: a zero, infinite, or NaN edge on a
/// periodic axis, periodic vectors that are not linearly independent,
/// or a non-finite origin.
pub const MINIMAGE_STATUS_BAD_CELL: i32 = 2;

/// This library cannot do what was asked for this lattice. The matching
/// member of `minimage_caps` is zero. Nothing was written.
pub const MINIMAGE_STATUS_UNSUPPORTED: i32 = 3;

/// `minimage_abi_compatible`: the host was compiled against another
/// major version or another layout revision.
pub const MINIMAGE_STATUS_INCOMPATIBLE: i32 = 4;

// The negative codes are written as differences so that the generated C
// header parenthesizes them: `#define X (0 - 1)` stays one operand inside
// any expression, where `#define X -1` would not.

/// A required pointer is null.
pub const MINIMAGE_STATUS_NULL_POINTER: i32 = 0 - 1;

/// The `size` a host wrote into a struct is smaller than the `size`
/// member itself.
pub const MINIMAGE_STATUS_BAD_SIZE: i32 = 0 - 2;

/// A row count whose buffer would not fit in the address space.
pub const MINIMAGE_STATUS_BAD_COUNT: i32 = 0 - 3;

/// The output buffer overlaps an input without being that input.
pub const MINIMAGE_STATUS_BAD_OVERLAP: i32 = 0 - 4;

/// A fault inside the library was caught at the boundary. The call
/// wrote nothing a host may rely on.
pub const MINIMAGE_STATUS_PANIC: i32 = 0 - 5;

/// The C ABI a header or a library was built for. Frozen: three
/// `uint32_t` and nothing else, in every revision.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct minimage_abi {
    /// `MINIMAGE_ABI_MAJOR`.
    pub major: u32,
    /// `MINIMAGE_ABI_MINOR`.
    pub minor: u32,
    /// `MINIMAGE_ABI_LAYOUT_REVISION`.
    pub layout_revision: u32,
}

/// What this library can do, for a host to read before it calls.
///
/// The host sets `size` to `sizeof(minimage_caps)` as it compiled it.
/// The library zeroes that many bytes, writes the members that fit, and
/// sets `size` to the bytes it wrote, so a member the host knows and the
/// library does not reads as zero. Each flag is 1 when supported and 0
/// when not. This struct is outside the frozen layout revision: it grows
/// at its end.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct minimage_caps {
    /// Bytes of this struct: in, the host's; out, those written.
    pub size: u32,
    /// Lattice handles: `minimage_lattice_diagonal`,
    /// `minimage_lattice_general`, `minimage_lattice_get_info`, and
    /// `minimage_lattice_free`.
    pub lattice: u32,
    /// A lattice may be open along any of its axes.
    pub open_axes: u32,
    /// `minimage_lattice_wrap_diffs`.
    pub wrap_diffs: u32,
    /// `minimage_lattice_wrap_pairs`.
    pub wrap_pairs: u32,
    /// `minimage_lattice_nearest_diffs`.
    pub nearest_diffs: u32,
    /// `minimage_lattice_nearest_pairs`.
    pub nearest_pairs: u32,
    /// `minimage_lattice_wrap_points`.
    pub wrap_points: u32,
    /// The output of a batch call may be one of its inputs.
    pub in_place: u32,
    /// The engine wrap (`wrap_diffs`, `wrap_pairs`) of a sheared lattice
    /// with an open axis. Zero in this library: those calls return
    /// `MINIMAGE_STATUS_UNSUPPORTED` for such a lattice, and
    /// `nearest_diffs` and `nearest_pairs` serve it.
    pub wrap_open_sheared: u32,
}

/// What a lattice handle stores, read back with
/// `minimage_lattice_get_info`.
///
/// The host sets `size` to `sizeof(minimage_lattice_info)` as it
/// compiled it. The library zeroes that many bytes, writes the members
/// that fit, and sets `size` to the bytes it wrote. This struct grows at
/// its end.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct minimage_lattice_info {
    /// Bytes of this struct: in, the host's; out, those written.
    pub size: u32,
    /// 1 when every periodic vector lies along its own Cartesian axis.
    pub orthorhombic: u32,
    /// 1 for a periodic axis and 0 for an open one, for a, b, c.
    pub periodic: [u32; 3],
    /// Zero.
    pub reserved: u32,
    /// The stored lattice vectors a, b, c, row-major. An orthorhombic
    /// lattice stores each periodic vector with a positive diagonal
    /// entry, whatever sign it was built with. The vector of an open
    /// axis is carried as given and never read.
    pub rows: [f64; 9],
    /// The origin of the cell.
    pub origin: [f64; 3],
    /// The spacing of the lattice planes across each periodic vector,
    /// and zero for an open axis.
    pub widths: [f64; 3],
}

/// A lattice: three vectors, an origin, and which axes are periodic.
///
/// Opaque. Built by `minimage_lattice_diagonal` or
/// `minimage_lattice_general`, released by `minimage_lattice_free`, and
/// immutable in between.
#[allow(non_camel_case_types)]
pub struct minimage_lattice {
    rows: [[f64; 3]; 3],
    origin: [f64; 3],
    periodic: [bool; 3],
    widths: [f64; 3],
    /// `dual[i] . rows[j]` is the Kronecker delta over the periodic
    /// axes, and `dual[i]` lies in their span. Zero for an open axis.
    dual: [[f64; 3]; 3],
    orthorhombic: bool,
    /// The engine wrap, or `None` for a sheared lattice with an open axis.
    engine: Option<Frame>,
    nearest: PartialCell,
}

/// The operations of this library, as function pointers.
///
/// A host that loads the library at run time resolves
/// `minimage_abi_stamp`, `minimage_abi_compatible`, and
/// `minimage_ops_get` by name and takes every other entry from this
/// table. The host sets `size` to `sizeof(minimage_ops)` as it compiled
/// it. The library zeroes that many bytes, writes the members that fit,
/// and sets `size` to the bytes it wrote. A null member is an operation
/// this library does not have: one the host's header knows and an older
/// library lacks. Each member is the entry point of the same name with
/// the `minimage_` prefix. This struct is outside the frozen layout
/// revision: it grows at its end.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct minimage_ops {
    /// Bytes of this struct: in, the host's; out, those written.
    pub size: u32,
    /// Zero.
    pub reserved: u32,
    /// `minimage_last_error`.
    pub last_error: Option<extern "C" fn() -> *const c_char>,
    /// `minimage_version`.
    pub version: Option<extern "C" fn() -> *const c_char>,
    /// `minimage_capabilities`.
    pub capabilities: Option<unsafe extern "C" fn(*mut minimage_caps) -> i32>,
    /// `minimage_lattice_diagonal`.
    pub lattice_diagonal: Option<
        unsafe extern "C" fn(*const f64, *const u32, *const f64, *mut *mut minimage_lattice) -> i32,
    >,
    /// `minimage_lattice_general`.
    pub lattice_general: Option<
        unsafe extern "C" fn(*const f64, *const u32, *const f64, *mut *mut minimage_lattice) -> i32,
    >,
    /// `minimage_lattice_free`.
    pub lattice_free: Option<unsafe extern "C" fn(*mut minimage_lattice)>,
    /// `minimage_lattice_get_info`.
    pub lattice_get_info:
        Option<unsafe extern "C" fn(*const minimage_lattice, *mut minimage_lattice_info) -> i32>,
    /// `minimage_lattice_wrap_diffs`.
    pub lattice_wrap_diffs:
        Option<unsafe extern "C" fn(*const minimage_lattice, *const f64, usize, *mut f64) -> i32>,
    /// `minimage_lattice_wrap_pairs`.
    pub lattice_wrap_pairs: Option<
        unsafe extern "C" fn(
            *const minimage_lattice,
            *const f64,
            *const f64,
            usize,
            *mut f64,
        ) -> i32,
    >,
    /// `minimage_lattice_nearest_diffs`.
    pub lattice_nearest_diffs:
        Option<unsafe extern "C" fn(*const minimage_lattice, *const f64, usize, *mut f64) -> i32>,
    /// `minimage_lattice_nearest_pairs`.
    pub lattice_nearest_pairs: Option<
        unsafe extern "C" fn(
            *const minimage_lattice,
            *const f64,
            *const f64,
            usize,
            *mut f64,
        ) -> i32,
    >,
    /// `minimage_lattice_wrap_points`.
    pub lattice_wrap_points:
        Option<unsafe extern "C" fn(*const minimage_lattice, *const f64, usize, *mut f64) -> i32>,
}

/// A status other than success, with the text for `minimage_last_error`.
struct Failure {
    status: i32,
    message: String,
}

impl Failure {
    fn new(status: i32, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn null(what: &str) -> Self {
        Self::new(MINIMAGE_STATUS_NULL_POINTER, format!("null {what}"))
    }
}

impl From<Error> for Failure {
    /// The constructors refuse a cell with one error, `Error::BadBox`.
    fn from(err: Error) -> Self {
        Self::new(MINIMAGE_STATUS_BAD_CELL, err.to_string())
    }
}

/// Run `body` at the boundary: a panic becomes `MINIMAGE_STATUS_PANIC`,
/// a failure stores its message, and a success clears the error slot.
fn guard(body: impl FnOnce() -> Result<(), Failure>) -> i32 {
    let (status, message) = match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => (MINIMAGE_STATUS_OK, None),
        Ok(Err(failure)) => (failure.status, Some(failure.message)),
        Err(_) => (
            MINIMAGE_STATUS_PANIC,
            Some(String::from("fault inside minimage")),
        ),
    };
    // The slot is thread-local and may be gone while a thread exits.
    let stored = catch_unwind(AssertUnwindSafe(|| match &message {
        Some(text) => set_error(text),
        None => clear_error(),
    }));
    if stored.is_err() && status == MINIMAGE_STATUS_OK {
        return MINIMAGE_STATUS_PANIC;
    }
    status
}

/// The bytes of a `total`-byte struct that fit in `host` bytes without
/// splitting a member. Members before `split` are `small` bytes each
/// and those after it `large` bytes each.
fn whole_members(host: usize, total: usize, split: usize, small: usize, large: usize) -> usize {
    let fit = host.min(total);
    if fit <= split {
        fit - fit % small
    } else {
        split + (fit - split) / large * large
    }
}

/// Zero the `size` bytes the host declared at `out`, copy the whole
/// members of `value` that fit, and store the bytes written in `size`.
///
/// # Safety
///
/// `out` is writable for the `u32` it starts with and for as many bytes
/// as that `u32` says. `T` is `repr(C)`, starts with a `u32`, and has no
/// padding.
unsafe fn write_prefix<T>(
    out: *mut T,
    value: &T,
    what: &str,
    split: usize,
    small: usize,
    large: usize,
) -> Result<(), Failure> {
    if out.is_null() {
        return Err(Failure::null(what));
    }
    // SAFETY: `out` starts with a readable `u32`.
    let host = unsafe { ptr::read_unaligned(out as *const u32) } as usize;
    if host < size_of::<u32>() {
        return Err(Failure::new(
            MINIMAGE_STATUS_BAD_SIZE,
            format!("{what}: size {host} is smaller than the size member"),
        ));
    }
    let wrote = whole_members(host, size_of::<T>(), split, small, large);
    // SAFETY: `out` is writable for `host` bytes and `wrote <= host`;
    // `value` is `wrote` readable bytes that do not overlap `out`.
    unsafe {
        ptr::write_bytes(out as *mut u8, 0, host);
        ptr::copy_nonoverlapping(value as *const T as *const u8, out as *mut u8, wrote);
        ptr::write_unaligned(out as *mut u32, wrote as u32);
    }
    Ok(())
}

/// Write the ABI this library was built for into `out`: the
/// `MINIMAGE_ABI_*` constants of its own header.
///
/// Returns `MINIMAGE_STATUS_OK`, or `MINIMAGE_STATUS_NULL_POINTER` when
/// `out` is null.
///
/// Threads: any thread at any time; reads no shared state. Sets or clears
/// the calling thread's error slot.
///
/// # Safety
///
/// `out` is null or one writable `minimage_abi`.
#[no_mangle]
pub unsafe extern "C" fn minimage_abi_stamp(out: *mut minimage_abi) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(Failure::null("ABI stamp output"));
        }
        // SAFETY: `out` is one writable `minimage_abi`.
        unsafe {
            out.write(minimage_abi {
                major: MINIMAGE_ABI_MAJOR,
                minor: MINIMAGE_ABI_MINOR,
                layout_revision: MINIMAGE_ABI_LAYOUT_REVISION,
            });
        }
        Ok(())
    })
}

/// Whether this library serves a host compiled against `host`: the same
/// major and the same layout revision. The minor is not compared.
///
/// Returns `MINIMAGE_STATUS_OK` when it does,
/// `MINIMAGE_STATUS_INCOMPATIBLE` when it does not, and
/// `MINIMAGE_STATUS_NULL_POINTER` when `host` is null.
///
/// Threads: any thread at any time; reads no shared state. Sets or
/// clears the calling thread's error slot.
///
/// # Safety
///
/// `host` is null or one readable `minimage_abi`.
#[no_mangle]
pub unsafe extern "C" fn minimage_abi_compatible(host: *const minimage_abi) -> i32 {
    guard(|| {
        if host.is_null() {
            return Err(Failure::null("host ABI"));
        }
        // SAFETY: `host` is one readable `minimage_abi`.
        let host = unsafe { host.read() };
        if host.major != MINIMAGE_ABI_MAJOR || host.layout_revision != MINIMAGE_ABI_LAYOUT_REVISION
        {
            return Err(Failure::new(
                MINIMAGE_STATUS_INCOMPATIBLE,
                format!(
                    "host ABI {}.{} layout {} against library ABI {}.{} layout {}",
                    host.major,
                    host.minor,
                    host.layout_revision,
                    MINIMAGE_ABI_MAJOR,
                    MINIMAGE_ABI_MINOR,
                    MINIMAGE_ABI_LAYOUT_REVISION
                ),
            ));
        }
        Ok(())
    })
}

/// Fill `out` with the capabilities of this library.
///
/// Returns `MINIMAGE_STATUS_OK`, `MINIMAGE_STATUS_NULL_POINTER` when
/// `out` is null, or `MINIMAGE_STATUS_BAD_SIZE` when `out->size` is
/// smaller than the `size` member.
///
/// Threads: any thread at any time; reads no shared state. Sets or
/// clears the calling thread's error slot.
///
/// # Safety
///
/// `out` is null or writable for `out->size` bytes, which the host set.
#[no_mangle]
pub unsafe extern "C" fn minimage_capabilities(out: *mut minimage_caps) -> i32 {
    guard(|| {
        let caps = minimage_caps {
            size: 0,
            lattice: 1,
            open_axes: 1,
            wrap_diffs: 1,
            wrap_pairs: 1,
            nearest_diffs: 1,
            nearest_pairs: 1,
            wrap_points: 1,
            in_place: 1,
            wrap_open_sheared: 0,
        };
        let word = size_of::<u32>();
        // SAFETY: the caller's contract; the struct is ten `u32`.
        unsafe {
            write_prefix(
                out,
                &caps,
                "capabilities",
                size_of::<minimage_caps>(),
                word,
                word,
            )
        }
    })
}

/// Fill `out` with the operations of this library.
///
/// Returns `MINIMAGE_STATUS_OK`, `MINIMAGE_STATUS_NULL_POINTER` when
/// `out` is null, or `MINIMAGE_STATUS_BAD_SIZE` when `out->size` is
/// smaller than the `size` member.
///
/// Threads: any thread at any time; reads no shared state. Sets or clears
/// the calling thread's error slot.
///
/// # Safety
///
/// `out` is null or writable for `out->size` bytes, which the host set.
#[no_mangle]
pub unsafe extern "C" fn minimage_ops_get(out: *mut minimage_ops) -> i32 {
    guard(|| {
        let ops = minimage_ops {
            size: 0,
            reserved: 0,
            last_error: Some(minimage_last_error),
            version: Some(minimage_version),
            capabilities: Some(minimage_capabilities),
            lattice_diagonal: Some(minimage_lattice_diagonal),
            lattice_general: Some(minimage_lattice_general),
            lattice_free: Some(minimage_lattice_free),
            lattice_get_info: Some(minimage_lattice_get_info),
            lattice_wrap_diffs: Some(minimage_lattice_wrap_diffs),
            lattice_wrap_pairs: Some(minimage_lattice_wrap_pairs),
            lattice_nearest_diffs: Some(minimage_lattice_nearest_diffs),
            lattice_nearest_pairs: Some(minimage_lattice_nearest_pairs),
            lattice_wrap_points: Some(minimage_lattice_wrap_points),
        };
        let word = size_of::<u32>();
        // SAFETY: the caller's contract; the struct is two `u32` and
        // then function pointers.
        unsafe {
            write_prefix(
                out,
                &ops,
                "operations table",
                2 * word,
                word,
                size_of::<usize>(),
            )
        }
    })
}

fn finite3(v: [f64; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

fn scaled(s: f64, v: [f64; 3]) -> [f64; 3] {
    [s * v[0], s * v[1], s * v[2]]
}

/// The dual of each periodic vector within the span of the periodic
/// vectors, and zero for an open axis. The periodic vectors are
/// linearly independent.
fn duals(rows: [[f64; 3]; 3], periodic: [bool; 3]) -> [[f64; 3]; 3] {
    // Complete the periodic vectors to a basis with unit vectors normal
    // to all of them. The inverse of that basis then holds the in-span
    // duals in the rows of the periodic axes.
    fn reject(mut v: [f64; 3], frame: &[[f64; 3]]) -> [f64; 3] {
        for q in frame {
            let along = dot(v, *q);
            v = [
                v[0] - along * q[0],
                v[1] - along * q[1],
                v[2] - along * q[2],
            ];
        }
        v
    }
    let mut frame: Vec<[f64; 3]> = Vec::with_capacity(3);
    for axis in 0..3 {
        if periodic[axis] {
            let rest = reject(rows[axis], &frame);
            frame.push(scaled(1.0 / norm(rest), rest));
        }
    }
    let mut basis = rows;
    for axis in 0..3 {
        if periodic[axis] {
            continue;
        }
        let mut best = [0.0; 3];
        for k in 0..3 {
            let mut unit = [0.0; 3];
            unit[k] = 1.0;
            let rest = reject(unit, &frame);
            if norm(rest) > norm(best) {
                best = rest;
            }
        }
        let unit = scaled(1.0 / norm(best), best);
        frame.push(unit);
        basis[axis] = unit;
    }
    let [a, b, c] = basis;
    let bc = cross(b, c);
    let inv = 1.0 / dot(a, bc);
    let full = [
        scaled(inv, bc),
        scaled(inv, cross(c, a)),
        scaled(inv, cross(a, b)),
    ];
    let mut dual = [[0.0; 3]; 3];
    for axis in 0..3 {
        if periodic[axis] {
            dual[axis] = full[axis];
        }
    }
    dual
}

impl minimage_lattice {
    fn new(
        mut rows: [[f64; 3]; 3],
        periodic: [bool; 3],
        origin: [f64; 3],
    ) -> Result<Self, Failure> {
        if !finite3(origin) {
            return Err(Failure::new(
                MINIMAGE_STATUS_BAD_CELL,
                "non-finite cell origin",
            ));
        }
        let (engine, orthorhombic, widths);
        if periodic == [true; 3] {
            let cell = Cell::from_vectors(rows[0], rows[1], rows[2], origin)?;
            rows = [cell.a(), cell.b(), cell.c()];
            engine = Some(cell.frame());
            orthorhombic = cell.is_ortho();
            widths = Some(cell.widths());
        } else {
            let aligned = (0..3)
                .all(|axis| !periodic[axis] || (0..3).all(|k| k == axis || rows[axis][k] == 0.0));
            let mut edges = [1.0; 3];
            if aligned {
                // The edge -L spans the lattice the edge +L spans.
                for axis in 0..3 {
                    if periodic[axis] {
                        edges[axis] = rows[axis][axis].abs();
                        rows[axis] = [0.0; 3];
                        rows[axis][axis] = edges[axis];
                    }
                }
            }
            // An open axis takes the width one: its component is put
            // back after the wrap.
            engine = aligned.then_some(Frame::Ortho(edges));
            orthorhombic = aligned;
            // The widths of an orthorhombic lattice are its edges, to
            // the bit; a sheared one takes them from the duals below.
            widths = aligned.then(|| {
                let mut out = [0.0; 3];
                for axis in 0..3 {
                    if periodic[axis] {
                        out[axis] = edges[axis];
                    }
                }
                out
            });
        }
        let nearest = PartialCell::from_ase(rows, periodic)?;
        let dual = duals(rows, periodic);
        let widths = widths.unwrap_or_else(|| {
            let mut out = [0.0; 3];
            for axis in 0..3 {
                if periodic[axis] {
                    out[axis] = 1.0 / norm(dual[axis]);
                }
            }
            out
        });
        if !widths.iter().all(|w| w.is_finite()) || !dual.iter().all(|d| finite3(*d)) {
            return Err(Error::BadBox.into());
        }
        Ok(Self {
            rows,
            origin,
            periodic,
            widths,
            dual,
            orthorhombic,
            engine,
            nearest,
        })
    }

    /// The engine wrap of each difference; an open axis keeps its
    /// component.
    fn wrap(&self, frame: Frame, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) {
        fused::wrap_many(frame, diffs, out);
        for axis in 0..3 {
            if !self.periodic[axis] {
                for (o, d) in out.iter_mut().zip(diffs) {
                    o[axis] = d[axis];
                }
            }
        }
    }

    /// Fractional coordinate of `r` along the periodic vector `axis`.
    fn fraction(&self, axis: usize, r: [f64; 3]) -> f64 {
        let o = self.origin;
        dot(self.dual[axis], [r[0] - o[0], r[1] - o[1], r[2] - o[2]])
    }

    /// The image of `r` in the cell: an integer count of each periodic
    /// vector is subtracted, so a point already inside is returned as
    /// it is.
    fn fold(&self, mut r: [f64; 3]) -> [f64; 3] {
        let start = r;
        for axis in 0..3 {
            if self.periodic[axis] {
                let count = self.fraction(axis, start).floor();
                let row = self.rows[axis];
                r = [
                    r[0] - count * row[0],
                    r[1] - count * row[1],
                    r[2] - count * row[2],
                ];
            }
        }
        // The subtraction rounds, so a point a rounding error below the
        // lower face lands on the upper one. Take that one step back.
        let landed = r;
        for axis in 0..3 {
            if self.periodic[axis] {
                let s = self.fraction(axis, landed);
                let row = self.rows[axis];
                if s >= 1.0 {
                    r = [r[0] - row[0], r[1] - row[1], r[2] - row[2]];
                } else if s < 0.0 {
                    r = [r[0] + row[0], r[1] + row[1], r[2] + row[2]];
                }
            }
        }
        r
    }
}

/// Three doubles at `p`.
///
/// # Safety
///
/// `p` is null or three readable doubles.
unsafe fn read3(p: *const f64, what: &str) -> Result<[f64; 3], Failure> {
    if p.is_null() {
        return Err(Failure::null(what));
    }
    // SAFETY: three readable doubles.
    Ok(unsafe { [*p, *p.add(1), *p.add(2)] })
}

/// The periodicity mask at `p`, or every axis periodic when `p` is null.
///
/// # Safety
///
/// `p` is null or three readable `u32`.
unsafe fn read_mask(p: *const u32) -> [bool; 3] {
    if p.is_null() {
        return [true; 3];
    }
    // SAFETY: three readable `u32`.
    unsafe { [*p != 0, *p.add(1) != 0, *p.add(2) != 0] }
}

/// The origin at `p`, or zero when `p` is null.
///
/// # Safety
///
/// `p` is null or three readable doubles.
unsafe fn read_origin(p: *const f64) -> [f64; 3] {
    if p.is_null() {
        return [0.0; 3];
    }
    // SAFETY: three readable doubles.
    unsafe { [*p, *p.add(1), *p.add(2)] }
}

/// Build a lattice and hand its address to `out`, which is null after
/// any failure.
///
/// # Safety
///
/// `out` is null or one writable pointer.
unsafe fn emit(
    out: *mut *mut minimage_lattice,
    build: impl FnOnce() -> Result<minimage_lattice, Failure>,
) -> Result<(), Failure> {
    if out.is_null() {
        return Err(Failure::null("lattice output"));
    }
    // SAFETY: `out` is one writable pointer.
    unsafe { out.write(ptr::null_mut()) };
    let lattice = Box::into_raw(Box::new(build()?));
    // SAFETY: as above.
    unsafe { out.write(lattice) };
    Ok(())
}

/// Build a lattice from three edge lengths along x, y, z.
///
/// `edges` is three doubles. `periodic` is three `uint32_t`, nonzero for
/// a periodic axis, or null for all three periodic. `origin` is three
/// doubles, or null for zero. A negative edge is stored as its
/// magnitude: `-L` and `L` span one lattice. The edge of an open axis is
/// carried and never read.
///
/// Returns `MINIMAGE_STATUS_OK` and writes a handle to `*out`, to be
/// released with `minimage_lattice_free`. On failure `*out` is null:
/// `MINIMAGE_STATUS_BAD_CELL` for a zero, infinite, or NaN edge on a
/// periodic axis or a non-finite origin, and
/// `MINIMAGE_STATUS_NULL_POINTER` for a null `edges` or `out`.
///
/// Threads: any thread at any time. Two builds share no state; the new
/// handle belongs to the caller. Sets or clears the calling thread's
/// error slot.
///
/// # Safety
///
/// `edges` is null or three readable doubles, `periodic` null or three
/// readable `uint32_t`, `origin` null or three readable doubles, and
/// `out` null or one writable pointer.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_diagonal(
    edges: *const f64,
    periodic: *const u32,
    origin: *const f64,
    out: *mut *mut minimage_lattice,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            emit(out, || {
                let [lx, ly, lz] = read3(edges, "edges")?;
                let rows = [[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]];
                minimage_lattice::new(rows, read_mask(periodic), read_origin(origin))
            })
        }
    })
}

/// Build a lattice from three lattice vectors.
///
/// `rows` is nine doubles, the vectors a, b, c one after another.
/// `periodic` and `origin` are as in `minimage_lattice_diagonal`. The
/// vector of an open axis is carried and never read. When every
/// periodic vector lies along its own axis the lattice is orthorhombic
/// and stores positive edges; a sheared lattice keeps the vectors it
/// was given. Read the stored vectors back with `minimage_lattice_get_info`.
///
/// Returns `MINIMAGE_STATUS_OK` and writes a handle to `*out`, to be
/// released with `minimage_lattice_free`. On failure `*out` is null:
/// `MINIMAGE_STATUS_BAD_CELL` for a zero or non-finite periodic vector,
/// periodic vectors that are not linearly independent, or a non-finite
/// origin, and `MINIMAGE_STATUS_NULL_POINTER` for a null `rows` or
/// `out`.
///
/// Threads: as `minimage_lattice_diagonal`.
///
/// # Safety
///
/// `rows` is null or nine readable doubles, `periodic` null or three
/// readable `uint32_t`, `origin` null or three readable doubles, and
/// `out` null or one writable pointer.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_general(
    rows: *const f64,
    periodic: *const u32,
    origin: *const f64,
    out: *mut *mut minimage_lattice,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            emit(out, || {
                if rows.is_null() {
                    return Err(Failure::null("lattice vectors"));
                }
                let rows = [
                    read3(rows, "lattice vectors")?,
                    read3(rows.add(3), "lattice vectors")?,
                    read3(rows.add(6), "lattice vectors")?,
                ];
                minimage_lattice::new(rows, read_mask(periodic), read_origin(origin))
            })
        }
    })
}

/// Release a lattice. Null is accepted and ignored. Does not touch
/// `minimage_last_error`.
///
/// Threads: exclusive per handle. No other call may use `lattice` while
/// or after it is released.
///
/// # Safety
///
/// `lattice` is null or a handle from `minimage_lattice_diagonal` or
/// `minimage_lattice_general` that has not been released, and no other
/// thread is using it.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_free(lattice: *mut minimage_lattice) {
    if lattice.is_null() {
        return;
    }
    // The lattice holds plain numbers, so dropping it cannot fail.
    // SAFETY: the handle came from `Box::into_raw` and is released once.
    drop(unsafe { Box::from_raw(lattice) });
}

/// Fill `out` with what `lattice` stores: the vectors, the origin, the
/// periodicity, and the widths.
///
/// Returns `MINIMAGE_STATUS_OK`, `MINIMAGE_STATUS_NULL_POINTER` when
/// either pointer is null, or `MINIMAGE_STATUS_BAD_SIZE` when
/// `out->size` is smaller than the `size` member.
///
/// Threads: shared read of the handle; any number of threads may call
/// this and the batch calls on one handle at once. Sets or clears the
/// calling thread's error slot.
///
/// # Safety
///
/// `lattice` is null or a live handle. `out` is null or writable for
/// `out->size` bytes, which the host set.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_get_info(
    lattice: *const minimage_lattice,
    out: *mut minimage_lattice_info,
) -> i32 {
    guard(|| {
        // SAFETY: null or a live handle.
        let lattice = unsafe { lattice.as_ref() }.ok_or_else(|| Failure::null("lattice"))?;
        let [a, b, c] = lattice.rows;
        let info = minimage_lattice_info {
            size: 0,
            orthorhombic: u32::from(lattice.orthorhombic),
            periodic: lattice.periodic.map(u32::from),
            reserved: 0,
            rows: [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]],
            origin: lattice.origin,
            widths: lattice.widths,
        };
        let word = size_of::<u32>();
        // SAFETY: the caller's contract; the struct is six `u32` and
        // then doubles.
        unsafe { write_prefix(out, &info, "lattice info", 6 * word, word, size_of::<f64>()) }
    })
}

/// Rows a batch call copies at a time when its output is an input.
const CHUNK: usize = 256;

/// How an output of `bytes` bytes lies against an input of the same length.
#[derive(Clone, Copy, PartialEq)]
enum Lies {
    Apart,
    Same,
    Across,
}

fn lies(input: *const f64, out: *const f64, bytes: usize) -> Lies {
    let (a, b) = (input as usize, out as usize);
    if a == b {
        Lies::Same
    } else if a < b.saturating_add(bytes) && b < a.saturating_add(bytes) {
        Lies::Across
    } else {
        Lies::Apart
    }
}

/// The lattice behind a handle and the byte length of `n` packed rows.
///
/// # Safety
///
/// `lattice` is null or a live handle.
unsafe fn batch<'a>(
    lattice: *const minimage_lattice,
    n: usize,
) -> Result<(&'a minimage_lattice, usize), Failure> {
    // SAFETY: null or a live handle.
    let lattice = unsafe { lattice.as_ref() }.ok_or_else(|| Failure::null("lattice"))?;
    let bytes = n
        .checked_mul(3 * size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            Failure::new(
                MINIMAGE_STATUS_BAD_COUNT,
                format!("{n} rows of three doubles do not fit in the address space"),
            )
        })?;
    Ok((lattice, bytes))
}

/// `apply(input rows, output rows)` over `n` packed rows. The output may
/// be the input.
///
/// # Safety
///
/// `input` and `out` are null or `n * 3` doubles, readable and writable.
unsafe fn rows_from_one(
    input: *const f64,
    n: usize,
    bytes: usize,
    out: *mut f64,
    apply: impl Fn(&[[f64; 3]], &mut [[f64; 3]]),
) -> Result<(), Failure> {
    if n == 0 {
        return Ok(());
    }
    if input.is_null() {
        return Err(Failure::null("input rows"));
    }
    if out.is_null() {
        return Err(Failure::null("output rows"));
    }
    match lies(input, out, bytes) {
        Lies::Across => Err(Failure::new(
            MINIMAGE_STATUS_BAD_OVERLAP,
            "the output overlaps the input without being the input",
        )),
        Lies::Apart => {
            // SAFETY: `n` rows each, and the two do not overlap.
            let (rows, result) = unsafe {
                (
                    slice::from_raw_parts(input as *const [f64; 3], n),
                    slice::from_raw_parts_mut(out as *mut [f64; 3], n),
                )
            };
            apply(rows, result);
            Ok(())
        }
        Lies::Same => {
            let mut rows = [[0.0; 3]; CHUNK];
            let mut result = [[0.0; 3]; CHUNK];
            let mut start = 0;
            while start < n {
                let m = (n - start).min(CHUNK);
                // SAFETY: `m` rows from `start` are inside both buffers,
                // and the local arrays overlap neither.
                unsafe {
                    ptr::copy_nonoverlapping(
                        (input as *const [f64; 3]).add(start),
                        rows.as_mut_ptr(),
                        m,
                    );
                }
                apply(&rows[..m], &mut result[..m]);
                // SAFETY: as above.
                unsafe {
                    ptr::copy_nonoverlapping(result.as_ptr(), (out as *mut [f64; 3]).add(start), m);
                }
                start += m;
            }
            Ok(())
        }
    }
}

/// `apply(p rows, q rows, output rows)` over `n` packed rows. The output
/// may be `p` or `q`.
///
/// # Safety
///
/// `p`, `q`, and `out` are null or `n * 3` doubles, readable and
/// writable.
unsafe fn rows_from_two(
    p: *const f64,
    q: *const f64,
    n: usize,
    bytes: usize,
    out: *mut f64,
    apply: impl Fn(&[[f64; 3]], &[[f64; 3]], &mut [[f64; 3]]),
) -> Result<(), Failure> {
    if n == 0 {
        return Ok(());
    }
    if p.is_null() {
        return Err(Failure::null("p rows"));
    }
    if q.is_null() {
        return Err(Failure::null("q rows"));
    }
    if out.is_null() {
        return Err(Failure::null("output rows"));
    }
    if lies(p, out, bytes) == Lies::Across || lies(q, out, bytes) == Lies::Across {
        return Err(Failure::new(
            MINIMAGE_STATUS_BAD_OVERLAP,
            "the output overlaps an input without being that input",
        ));
    }
    let mut from = [[0.0; 3]; CHUNK];
    let mut to = [[0.0; 3]; CHUNK];
    let mut result = [[0.0; 3]; CHUNK];
    let mut start = 0;
    while start < n {
        let m = (n - start).min(CHUNK);
        // SAFETY: `m` rows from `start` are inside all three buffers,
        // and the local arrays overlap none of them.
        unsafe {
            ptr::copy_nonoverlapping((p as *const [f64; 3]).add(start), from.as_mut_ptr(), m);
            ptr::copy_nonoverlapping((q as *const [f64; 3]).add(start), to.as_mut_ptr(), m);
        }
        apply(&from[..m], &to[..m], &mut result[..m]);
        // SAFETY: as above.
        unsafe {
            ptr::copy_nonoverlapping(result.as_ptr(), (out as *mut [f64; 3]).add(start), m);
        }
        start += m;
    }
    Ok(())
}

fn engine(lattice: &minimage_lattice) -> Result<Frame, Failure> {
    lattice.engine.ok_or_else(|| {
        Failure::new(
            MINIMAGE_STATUS_UNSUPPORTED,
            "no engine wrap for a sheared lattice with an open axis; use the nearest-image calls",
        )
    })
}

/// Engine wrap of `n` packed difference vectors into `out`.
///
/// Each difference is wrapped in fractional coordinates: a component in
/// `[-L/2, L/2)` along every periodic axis of an orthorhombic lattice,
/// and the fractional wrap `H wrap(Hinv d)` of a sheared one. For a
/// fully periodic lattice this is `minimage_wrap_many`. The component
/// along an open axis is returned as given. `out` may be `diffs`.
///
/// Returns `MINIMAGE_STATUS_OK`; `MINIMAGE_STATUS_UNSUPPORTED` for a
/// sheared lattice with an open axis, which has no engine wrap
/// (`minimage_caps.wrap_open_sheared`; use
/// `minimage_lattice_nearest_diffs`); `MINIMAGE_STATUS_NULL_POINTER`,
/// `MINIMAGE_STATUS_BAD_COUNT`, or `MINIMAGE_STATUS_BAD_OVERLAP` for a
/// misuse. When `n` is zero the buffers are not read.
///
/// Threads: shared read of the handle, as `minimage_lattice_get_info`. Two
/// calls must not write one output buffer.
///
/// # Safety
///
/// `lattice` is null or a live handle. `diffs` and `out` are null or
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_wrap_diffs(
    lattice: *const minimage_lattice,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            let (lattice, bytes) = batch(lattice, n)?;
            let frame = engine(lattice)?;
            rows_from_one(diffs, n, bytes, out, |rows, result| {
                lattice.wrap(frame, rows, result)
            })
        }
    })
}

/// Engine wrap of `q[k] - p[k]` for `n` packed pairs into `out`: the
/// displacement from `p[k]` to `q[k]`, by the wrap of
/// `minimage_lattice_wrap_diffs`. `out` may be `p` or `q`.
///
/// Returns as `minimage_lattice_wrap_diffs` does.
///
/// Threads: as `minimage_lattice_wrap_diffs`.
///
/// # Safety
///
/// `lattice` is null or a live handle. `p`, `q`, and `out` are null or
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_wrap_pairs(
    lattice: *const minimage_lattice,
    p: *const f64,
    q: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            let (lattice, bytes) = batch(lattice, n)?;
            let frame = engine(lattice)?;
            rows_from_two(p, q, n, bytes, out, |from, to, result| {
                let mut diffs = [[0.0; 3]; CHUNK];
                for ((d, a), b) in diffs.iter_mut().zip(from).zip(to) {
                    *d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                }
                lattice.wrap(frame, &diffs[..from.len()], result)
            })
        }
    })
}

/// Nearest image of `n` packed difference vectors into `out`: the
/// shortest vector that differs from each by a combination of the
/// periodic lattice vectors. Defined for every lattice. For a fully
/// periodic one this is `minimage_displacement_euclidean`; with an open
/// axis it is `minimage_displacement_euclidean_pbc`. On an exact tie it
/// keeps the engine wrap. `out` may be `diffs`.
///
/// Returns `MINIMAGE_STATUS_OK`, or `MINIMAGE_STATUS_NULL_POINTER`,
/// `MINIMAGE_STATUS_BAD_COUNT`, or `MINIMAGE_STATUS_BAD_OVERLAP` for a
/// misuse. When `n` is zero the buffers are not read.
///
/// Threads: as `minimage_lattice_wrap_diffs`.
///
/// # Safety
///
/// `lattice` is null or a live handle. `diffs` and `out` are null or
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_nearest_diffs(
    lattice: *const minimage_lattice,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            let (lattice, bytes) = batch(lattice, n)?;
            rows_from_one(diffs, n, bytes, out, |rows, result| {
                for (o, d) in result.iter_mut().zip(rows) {
                    *o = lattice.nearest.displacement_euclidean([0.0; 3], *d);
                }
            })
        }
    })
}

/// Nearest image of `q[k] - p[k]` for `n` packed pairs into `out`: the
/// displacement from `p[k]` to `q[k]`, by the search of
/// `minimage_lattice_nearest_diffs`. `out` may be `p` or `q`.
///
/// Returns as `minimage_lattice_nearest_diffs` does.
///
/// Threads: as `minimage_lattice_wrap_diffs`.
///
/// # Safety
///
/// `lattice` is null or a live handle. `p`, `q`, and `out` are null or
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_nearest_pairs(
    lattice: *const minimage_lattice,
    p: *const f64,
    q: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            let (lattice, bytes) = batch(lattice, n)?;
            rows_from_two(p, q, n, bytes, out, |from, to, result| {
                for ((o, a), b) in result.iter_mut().zip(from).zip(to) {
                    *o = lattice.nearest.displacement_euclidean(*a, *b);
                }
            })
        }
    })
}

/// Wrap `n` packed points into the cell, into `out`.
///
/// Each point moves by an integer count of every periodic lattice
/// vector, so that its fractional coordinate along that vector, measured
/// from the origin, is in `[0, 1)` to within rounding. A point already
/// inside is returned bit for bit, and the coordinate along an open axis
/// is never changed. `out` may be `points`.
///
/// Returns `MINIMAGE_STATUS_OK`, or `MINIMAGE_STATUS_NULL_POINTER`,
/// `MINIMAGE_STATUS_BAD_COUNT`, or `MINIMAGE_STATUS_BAD_OVERLAP` for a
/// misuse. When `n` is zero the buffers are not read.
///
/// Threads: as `minimage_lattice_wrap_diffs`.
///
/// # Safety
///
/// `lattice` is null or a live handle. `points` and `out` are null or
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn minimage_lattice_wrap_points(
    lattice: *const minimage_lattice,
    points: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe {
            let (lattice, bytes) = batch(lattice, n)?;
            rows_from_one(points, n, bytes, out, |rows, result| {
                for (o, r) in result.iter_mut().zip(rows) {
                    *o = lattice.fold(*r);
                }
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    fn last_error() -> Option<String> {
        let text = minimage_last_error();
        // SAFETY: null or a NUL-terminated string valid until the next call.
        (!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_str().unwrap().to_owned())
    }

    #[test]
    fn a_panic_becomes_a_negative_status() {
        let status = guard(|| panic!("the boundary test panics on purpose"));
        assert_eq!(status, MINIMAGE_STATUS_PANIC);
        assert!(status < 0);
        assert_eq!(last_error().as_deref(), Some("fault inside minimage"));
        assert_eq!(guard(|| Ok(())), MINIMAGE_STATUS_OK);
        assert_eq!(last_error(), None);
    }

    #[test]
    fn a_failure_keeps_its_code_and_its_message() {
        let status = guard(|| Err(Failure::new(MINIMAGE_STATUS_UNSUPPORTED, "not here")));
        assert_eq!(status, MINIMAGE_STATUS_UNSUPPORTED);
        assert_eq!(last_error().as_deref(), Some("not here"));
    }

    #[test]
    fn a_prefix_never_splits_a_member() {
        // Ten four-byte members.
        assert_eq!(whole_members(40, 40, 40, 4, 4), 40);
        assert_eq!(whole_members(400, 40, 40, 4, 4), 40);
        assert_eq!(whole_members(11, 40, 40, 4, 4), 8);
        assert_eq!(whole_members(4, 40, 40, 4, 4), 4);
        // Two four-byte members, then eight-byte ones.
        assert_eq!(whole_members(7, 104, 8, 4, 8), 4);
        assert_eq!(whole_members(8, 104, 8, 4, 8), 8);
        assert_eq!(whole_members(23, 104, 8, 4, 8), 16);
        assert_eq!(whole_members(1000, 104, 8, 4, 8), 104);
    }

    #[test]
    fn the_growable_structs_have_no_padding() {
        assert_eq!(size_of::<minimage_abi>(), 12);
        assert_eq!(size_of::<minimage_caps>(), 40);
        assert_eq!(size_of::<minimage_lattice_info>(), 24 + 15 * 8);
        assert_eq!(size_of::<minimage_ops>(), 8 + 12 * size_of::<usize>());
    }

    #[test]
    fn duals_pair_with_the_periodic_vectors_only() {
        let rows = [[3.0, 1.0, 0.5], [-1.0, 4.0, 0.25], [0.5, -0.5, 5.0]];
        for mask in 0..8 {
            let periodic = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
            let dual = duals(rows, periodic);
            for i in 0..3 {
                if !periodic[i] {
                    assert_eq!(dual[i], [0.0; 3]);
                    continue;
                }
                for j in 0..3 {
                    if periodic[j] {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!((dot(dual[i], rows[j]) - want).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
