//! The negotiated C ABI: the stamp and the compatibility rule, the
//! size-prefixed capability and operations structs, the status codes,
//! and every lattice operation against the Rust API and a brute-force
//! scan over lattice images.

#![cfg(feature = "capi")]

use std::ffi::CStr;
use std::mem::{size_of, MaybeUninit};
use std::ptr;

use minimage::{
    mi_cell, minimage_abi, minimage_abi_compatible, minimage_abi_stamp, minimage_capabilities,
    minimage_caps, minimage_last_error, minimage_lattice, minimage_lattice_info, minimage_ops,
    minimage_ops_get, minimage_wrap_many, Cell, PartialCell, MINIMAGE_ABI_LAYOUT_REVISION,
    MINIMAGE_ABI_MAJOR, MINIMAGE_ABI_MINOR, MINIMAGE_STATUS_BAD_CELL, MINIMAGE_STATUS_BAD_COUNT,
    MINIMAGE_STATUS_BAD_OVERLAP, MINIMAGE_STATUS_BAD_SIZE, MINIMAGE_STATUS_INCOMPATIBLE,
    MINIMAGE_STATUS_NULL_POINTER, MINIMAGE_STATUS_OK, MINIMAGE_STATUS_UNSUPPORTED,
};

type Rows = [[f64; 3]; 3];

const ALL: [bool; 3] = [true; 3];
const TRIANGULAR: Rows = [[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]];

fn last_error() -> Option<String> {
    let text = minimage_last_error();
    // SAFETY: null or a NUL-terminated string valid until the next call.
    (!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_str().unwrap().to_owned())
}

/// The operations table as a host with this header receives it.
fn table() -> minimage_ops {
    // SAFETY: the all-zero table is every member null.
    let mut ops: minimage_ops = unsafe { MaybeUninit::zeroed().assume_init() };
    ops.size = size_of::<minimage_ops>() as u32;
    assert_eq!(unsafe { minimage_ops_get(&mut ops) }, MINIMAGE_STATUS_OK);
    assert_eq!(ops.size as usize, size_of::<minimage_ops>());
    ops
}

/// A lattice handle from the table, released on drop.
struct Lattice {
    ops: minimage_ops,
    raw: *mut minimage_lattice,
}

impl Lattice {
    fn general(rows: Rows, periodic: [bool; 3], origin: [f64; 3]) -> Result<Self, i32> {
        let ops = table();
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let mask = periodic.map(u32::from);
        let mut raw = ptr::NonNull::dangling().as_ptr();
        let status = unsafe {
            ops.lattice_general.unwrap()(flat.as_ptr(), mask.as_ptr(), origin.as_ptr(), &mut raw)
        };
        if status != MINIMAGE_STATUS_OK {
            assert!(raw.is_null(), "a failed build leaves the handle null");
            return Err(status);
        }
        Ok(Self { ops, raw })
    }

    fn diagonal(edges: [f64; 3], periodic: [bool; 3], origin: [f64; 3]) -> Result<Self, i32> {
        let ops = table();
        let mask = periodic.map(u32::from);
        let mut raw = ptr::NonNull::dangling().as_ptr();
        let status = unsafe {
            ops.lattice_diagonal.unwrap()(edges.as_ptr(), mask.as_ptr(), origin.as_ptr(), &mut raw)
        };
        if status != MINIMAGE_STATUS_OK {
            assert!(raw.is_null(), "a failed build leaves the handle null");
            return Err(status);
        }
        Ok(Self { ops, raw })
    }

    fn info(&self) -> minimage_lattice_info {
        // SAFETY: plain numbers.
        let mut info: minimage_lattice_info = unsafe { MaybeUninit::zeroed().assume_init() };
        info.size = size_of::<minimage_lattice_info>() as u32;
        assert_eq!(
            unsafe { self.ops.lattice_get_info.unwrap()(self.raw, &mut info) },
            MINIMAGE_STATUS_OK
        );
        assert_eq!(info.size as usize, size_of::<minimage_lattice_info>());
        info
    }

    fn rows(&self) -> Rows {
        let r = self.info().rows;
        [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]]
    }

    fn one(
        &self,
        call: unsafe extern "C" fn(*const minimage_lattice, *const f64, usize, *mut f64) -> i32,
        input: &[[f64; 3]],
    ) -> Result<Vec<[f64; 3]>, i32> {
        let mut out = vec![[f64::NAN; 3]; input.len()];
        let status = unsafe {
            call(
                self.raw,
                input.as_ptr().cast(),
                input.len(),
                out.as_mut_ptr().cast(),
            )
        };
        if status != MINIMAGE_STATUS_OK {
            assert!(
                out.iter().flatten().all(|v| v.is_nan()),
                "a failed call writes nothing"
            );
            return Err(status);
        }
        // The same call with the output on the input.
        let mut same = input.to_vec();
        let again = unsafe {
            call(
                self.raw,
                same.as_ptr().cast(),
                same.len(),
                same.as_mut_ptr().cast(),
            )
        };
        assert_eq!(again, MINIMAGE_STATUS_OK);
        for (a, b) in out.iter().zip(&same) {
            assert!(close3(*a, *b), "in place {b:?} but apart {a:?}");
        }
        Ok(out)
    }

    fn two(
        &self,
        call: unsafe extern "C" fn(
            *const minimage_lattice,
            *const f64,
            *const f64,
            usize,
            *mut f64,
        ) -> i32,
        p: &[[f64; 3]],
        q: &[[f64; 3]],
    ) -> Result<Vec<[f64; 3]>, i32> {
        let mut out = vec![[f64::NAN; 3]; p.len()];
        let status = unsafe {
            call(
                self.raw,
                p.as_ptr().cast(),
                q.as_ptr().cast(),
                p.len(),
                out.as_mut_ptr().cast(),
            )
        };
        if status != MINIMAGE_STATUS_OK {
            return Err(status);
        }
        // The output on `p`, then on `q`.
        for onto_p in [true, false] {
            let (mut a, mut b) = (p.to_vec(), q.to_vec());
            let target: *mut f64 = if onto_p {
                a.as_mut_ptr().cast()
            } else {
                b.as_mut_ptr().cast()
            };
            let again = unsafe {
                call(
                    self.raw,
                    a.as_ptr().cast(),
                    b.as_ptr().cast(),
                    p.len(),
                    target,
                )
            };
            assert_eq!(again, MINIMAGE_STATUS_OK);
            let same = if onto_p { &a } else { &b };
            for (x, y) in out.iter().zip(same) {
                assert!(close3(*x, *y), "in place {y:?} but apart {x:?}");
            }
        }
        Ok(out)
    }

    fn wrap_diffs(&self, diffs: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, i32> {
        self.one(self.ops.lattice_wrap_diffs.unwrap(), diffs)
    }

    fn nearest_diffs(&self, diffs: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, i32> {
        self.one(self.ops.lattice_nearest_diffs.unwrap(), diffs)
    }

    fn wrap_points(&self, points: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, i32> {
        self.one(self.ops.lattice_wrap_points.unwrap(), points)
    }

    fn wrap_pairs(&self, p: &[[f64; 3]], q: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, i32> {
        self.two(self.ops.lattice_wrap_pairs.unwrap(), p, q)
    }

    fn nearest_pairs(&self, p: &[[f64; 3]], q: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, i32> {
        self.two(self.ops.lattice_nearest_pairs.unwrap(), p, q)
    }
}

impl Drop for Lattice {
    fn drop(&mut self) {
        unsafe { self.ops.lattice_free.unwrap()(self.raw) };
    }
}

fn norm2(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

fn close3(a: [f64; 3], b: [f64; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Squared length of the shortest `d + n . rows` over the periodic rows.
fn brute_min2(rows: &Rows, periodic: [bool; 3], d: [f64; 3]) -> f64 {
    let reach = |axis: usize| if periodic[axis] { 4 } else { 0 };
    let mut best = f64::INFINITY;
    for na in -reach(0)..=reach(0) {
        for nb in -reach(1)..=reach(1) {
            for nc in -reach(2)..=reach(2) {
                let mut v = d;
                for (row, count) in rows.iter().zip([na, nb, nc]) {
                    for k in 0..3 {
                        v[k] += f64::from(count) * row[k];
                    }
                }
                best = best.min(norm2(v));
            }
        }
    }
    best
}

fn same_modulo(rows: &Rows, periodic: [bool; 3], v: [f64; 3], w: [f64; 3]) -> bool {
    brute_min2(rows, periodic, sub(v, w)) < 1e-18
}

fn points(n: usize, seed: u64, scale: f64) -> Vec<[f64; 3]> {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0) * scale
    };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

fn negate(rows: Rows, mask: usize) -> Rows {
    let mut out = rows;
    for (axis, row) in out.iter_mut().enumerate() {
        if mask & (1 << axis) != 0 {
            for value in row.iter_mut() {
                *value = -*value;
            }
        }
    }
    out
}

fn diagonal(edges: [f64; 3]) -> Rows {
    [
        [edges[0], 0.0, 0.0],
        [0.0, edges[1], 0.0],
        [0.0, 0.0, edges[2]],
    ]
}

/// Fractional coordinates of `r` along the periodic rows, by Cramer's
/// rule on the rows completed with the normals of the open axes.
fn fractions(rows: &Rows, periodic: [bool; 3], origin: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let mut basis = *rows;
    let open: Vec<usize> = (0..3).filter(|&axis| !periodic[axis]).collect();
    let kept: Vec<usize> = (0..3).filter(|&axis| periodic[axis]).collect();
    match (kept.len(), open.len()) {
        (2, 1) => basis[open[0]] = cross(rows[kept[0]], rows[kept[1]]),
        (1, 2) => {
            let a = rows[kept[0]];
            let seed = if a[0].abs() < a[1].abs() {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            basis[open[0]] = cross(a, seed);
            basis[open[1]] = cross(a, basis[open[0]]);
        }
        (0, 3) => return [0.0; 3],
        _ => {}
    }
    let [a, b, c] = basis;
    let d = sub(r, origin);
    let det = dot(a, cross(b, c));
    [
        dot(d, cross(b, c)) / det,
        dot(d, cross(c, a)) / det,
        dot(d, cross(a, b)) / det,
    ]
}

/// Every batch operation of a lattice against the Rust API and the
/// brute-force scan. `input` is what the lattice was built from.
fn check_lattice(input: Rows, periodic: [bool; 3]) {
    let origin = [0.3, -1.1, 2.0];
    let lattice = Lattice::general(input, periodic, origin)
        .unwrap_or_else(|status| panic!("{input:?} {periodic:?}: status {status}"));
    let info = lattice.info();
    let stored = lattice.rows();
    assert_eq!(info.periodic, periodic.map(u32::from));
    assert_eq!(info.origin, origin);
    assert_eq!(info.reserved, 0);
    for axis in 0..3 {
        if periodic[axis] {
            assert!(same_modulo(&input, periodic, stored[axis], [0.0; 3]));
            assert!(same_modulo(&stored, periodic, input[axis], [0.0; 3]));
            assert!(info.widths[axis] > 0.0 && info.widths[axis].is_finite());
            if info.orthorhombic == 1 {
                assert!(stored[axis][axis] > 0.0, "{stored:?}");
                assert_eq!(info.widths[axis], stored[axis][axis]);
            }
        } else {
            assert_eq!(stored[axis], input[axis], "an open vector is carried");
            assert_eq!(info.widths[axis], 0.0);
        }
    }

    let scale = 1.3
        * input
            .iter()
            .fold(0.0_f64, |m, row| m.max(norm2(*row).sqrt()));
    let ps = points(700, 0x9e3779b97f4a7c15, scale);
    let qs = points(700, 0xd1b54a32d192ed03, scale);
    let diffs: Vec<[f64; 3]> = ps.iter().zip(&qs).map(|(p, q)| sub(*q, *p)).collect();
    let partial = PartialCell::from_ase(stored, periodic).unwrap();

    // Nearest image: defined for every lattice.
    let nearest = lattice.nearest_diffs(&diffs).unwrap();
    let nearest_pairs = lattice.nearest_pairs(&ps, &qs).unwrap();
    for k in 0..diffs.len() {
        assert!(same_modulo(&input, periodic, nearest[k], diffs[k]));
        assert!(
            close(norm2(nearest[k]), brute_min2(&input, periodic, diffs[k])),
            "{input:?} {periodic:?}: {:?}",
            nearest[k]
        );
        assert_eq!(
            nearest[k],
            partial.displacement_euclidean([0.0; 3], diffs[k])
        );
        assert_eq!(
            nearest_pairs[k],
            partial.displacement_euclidean(ps[k], qs[k])
        );
        for axis in 0..3 {
            if !periodic[axis] && info.orthorhombic == 1 {
                assert_eq!(nearest[k][axis], diffs[k][axis]);
            }
        }
    }

    // Engine wrap: every lattice but a sheared one with an open axis.
    let sheared_open = periodic != ALL && info.orthorhombic == 0;
    match lattice.wrap_diffs(&diffs) {
        Err(status) => {
            assert!(sheared_open, "{input:?} {periodic:?}");
            assert_eq!(status, MINIMAGE_STATUS_UNSUPPORTED);
            assert!(last_error().unwrap().contains("nearest-image"));
            assert_eq!(
                lattice.wrap_pairs(&ps, &qs).err(),
                Some(MINIMAGE_STATUS_UNSUPPORTED)
            );
        }
        Ok(wrapped) => {
            assert!(!sheared_open, "{input:?} {periodic:?}");
            let pairs = lattice.wrap_pairs(&ps, &qs).unwrap();
            for k in 0..diffs.len() {
                assert!(same_modulo(&input, periodic, wrapped[k], diffs[k]));
                assert!(close3(pairs[k], wrapped[k]));
                for axis in 0..3 {
                    if !periodic[axis] {
                        assert_eq!(wrapped[k][axis], diffs[k][axis], "open axis {axis}");
                    }
                }
                if info.orthorhombic == 1 {
                    assert!(close(
                        norm2(wrapped[k]),
                        brute_min2(&input, periodic, diffs[k])
                    ));
                }
            }
            if periodic == ALL {
                // The direct entry point, on the cell the lattice stores.
                let cell = Cell::from_vectors(stored[0], stored[1], stored[2], origin).unwrap();
                let (a, b, c) = (cell.a(), cell.b(), cell.c());
                let raw = mi_cell {
                    ax: a[0],
                    ay: a[1],
                    az: a[2],
                    bx: b[0],
                    by: b[1],
                    bz: b[2],
                    cx: c[0],
                    cy: c[1],
                    cz: c[2],
                    ox: origin[0],
                    oy: origin[1],
                    oz: origin[2],
                };
                let mut direct = vec![[0.0; 3]; diffs.len()];
                assert_eq!(
                    unsafe {
                        minimage_wrap_many(
                            &raw,
                            diffs.as_ptr().cast(),
                            diffs.len(),
                            direct.as_mut_ptr().cast(),
                        )
                    },
                    0
                );
                assert_eq!(wrapped, direct, "{input:?}: not minimage_wrap_many");
                for k in 0..diffs.len() {
                    assert!(close3(wrapped[k], cell.displacement(ps[k], qs[k])));
                }
            }
        }
    }

    // Points into the cell.
    let folded = lattice.wrap_points(&ps).unwrap();
    for (p, w) in ps.iter().zip(&folded) {
        assert!(
            same_modulo(&input, periodic, *w, *p),
            "{input:?} {periodic:?}: {p:?} wrapped to {w:?}"
        );
        let s = fractions(&stored, periodic, origin, *w);
        for axis in 0..3 {
            if periodic[axis] {
                assert!((-1e-12..1.0 + 1e-12).contains(&s[axis]), "{s:?}");
            }
        }
        if info.orthorhombic == 1 {
            for axis in 0..3 {
                if periodic[axis] {
                    let x = w[axis] - origin[axis];
                    assert!((0.0..stored[axis][axis]).contains(&x), "{x}");
                } else {
                    assert_eq!(w[axis], p[axis], "open axis {axis}");
                }
            }
        }
    }
    // A point inside the cell is returned bit for bit.
    assert_eq!(lattice.wrap_points(&folded).unwrap(), folded);
}

#[test]
fn the_stamp_is_the_header_constants() {
    let mut stamp = minimage_abi {
        major: 99,
        minor: 99,
        layout_revision: 99,
    };
    assert_eq!(
        unsafe { minimage_abi_stamp(&mut stamp) },
        MINIMAGE_STATUS_OK
    );
    assert_eq!(
        stamp,
        minimage_abi {
            major: MINIMAGE_ABI_MAJOR,
            minor: MINIMAGE_ABI_MINOR,
            layout_revision: MINIMAGE_ABI_LAYOUT_REVISION,
        }
    );
    assert_eq!(size_of::<minimage_abi>(), 12);
    assert_eq!(
        unsafe { minimage_abi_stamp(ptr::null_mut()) },
        MINIMAGE_STATUS_NULL_POINTER
    );
}

#[test]
fn compatibility_is_the_major_and_the_layout_revision() {
    let host = |major, minor, layout_revision| {
        let abi = minimage_abi {
            major,
            minor,
            layout_revision,
        };
        unsafe { minimage_abi_compatible(&abi) }
    };
    let (major, minor, layout) = (
        MINIMAGE_ABI_MAJOR,
        MINIMAGE_ABI_MINOR,
        MINIMAGE_ABI_LAYOUT_REVISION,
    );
    assert_eq!(host(major, minor, layout), MINIMAGE_STATUS_OK);
    assert_eq!(last_error(), None);
    // The minor is informational, in both directions.
    assert_eq!(host(major, minor + 7, layout), MINIMAGE_STATUS_OK);
    assert_eq!(host(major, 0, layout), MINIMAGE_STATUS_OK);
    assert_eq!(host(major + 1, minor, layout), MINIMAGE_STATUS_INCOMPATIBLE);
    assert!(last_error().unwrap().contains("host ABI"));
    assert_eq!(host(major, minor, layout + 1), MINIMAGE_STATUS_INCOMPATIBLE);
    assert_eq!(host(0, minor, layout), MINIMAGE_STATUS_INCOMPATIBLE);
    assert_eq!(
        unsafe { minimage_abi_compatible(ptr::null()) },
        MINIMAGE_STATUS_NULL_POINTER
    );
}

#[test]
fn status_codes_are_distinct_and_signed_by_kind() {
    use minimage::{MINIMAGE_STATUS_ERROR, MINIMAGE_STATUS_PANIC};
    let recoverable = [
        MINIMAGE_STATUS_ERROR,
        MINIMAGE_STATUS_BAD_CELL,
        MINIMAGE_STATUS_UNSUPPORTED,
        MINIMAGE_STATUS_INCOMPATIBLE,
    ];
    let fatal = [
        MINIMAGE_STATUS_NULL_POINTER,
        MINIMAGE_STATUS_BAD_SIZE,
        MINIMAGE_STATUS_BAD_COUNT,
        MINIMAGE_STATUS_BAD_OVERLAP,
        MINIMAGE_STATUS_PANIC,
    ];
    assert_eq!(MINIMAGE_STATUS_OK, 0);
    assert!(recoverable.iter().all(|code| *code > 0));
    assert!(fatal.iter().all(|code| *code < 0));
    let mut all: Vec<i32> = recoverable.iter().chain(&fatal).copied().collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), recoverable.len() + fatal.len());
}

/// `words` four-byte words of 0xff with the leading one set to `size`,
/// passed to `fill` as a struct of type `T`.
fn prefix<T>(
    words: usize,
    size: u32,
    fill: unsafe extern "C" fn(*mut T) -> i32,
) -> (i32, Vec<u32>) {
    let mut buffer = vec![u64::MAX; (words + 1) / 2];
    let raw = buffer.as_mut_ptr().cast::<u32>();
    unsafe { raw.write(size) };
    let status = unsafe { fill(raw.cast()) };
    let out = (0..words).map(|k| unsafe { raw.add(k).read() }).collect();
    (status, out)
}

#[test]
fn capabilities_write_the_prefix_that_fits() {
    let full = size_of::<minimage_caps>() as u32;
    assert_eq!(full, 40);

    let (status, words) = prefix(10, full, minimage_capabilities);
    assert_eq!(status, MINIMAGE_STATUS_OK);
    // size, lattice, open_axes, wrap_diffs, wrap_pairs, nearest_diffs,
    // nearest_pairs, wrap_points, in_place, wrap_open_sheared.
    assert_eq!(words, [40, 1, 1, 1, 1, 1, 1, 1, 1, 0]);

    // An older host: three members, and nothing past them is touched.
    let (status, words) = prefix(10, 12, minimage_capabilities);
    assert_eq!(status, MINIMAGE_STATUS_OK);
    assert_eq!(&words[..3], [12, 1, 1]);
    assert!(words[3..].iter().all(|w| *w == u32::MAX));

    // A size inside a member: the whole members before it, the rest zero.
    let (status, words) = prefix(10, 11, minimage_capabilities);
    assert_eq!(status, MINIMAGE_STATUS_OK);
    assert_eq!(&words[..2], [8, 1]);
    assert_eq!(words[2] & 0x00ff_ffff, 0, "the stated bytes are zeroed");
    assert_eq!(words[2] & 0xff00_0000, 0xff00_0000, "and no byte past them");

    // A newer host: the members this library lacks read as zero.
    let (status, words) = prefix(16, 64, minimage_capabilities);
    assert_eq!(status, MINIMAGE_STATUS_OK);
    assert_eq!(words[0], 40);
    assert!(words[10..].iter().all(|w| *w == 0));

    let (status, words) = prefix(10, 3, minimage_capabilities);
    assert_eq!(status, MINIMAGE_STATUS_BAD_SIZE);
    assert!(words[1..].iter().all(|w| *w == u32::MAX));
    assert!(last_error().unwrap().contains("size member"));
    assert_eq!(
        unsafe { minimage_capabilities(ptr::null_mut()) },
        MINIMAGE_STATUS_NULL_POINTER
    );
}

#[test]
fn the_operations_table_writes_the_prefix_that_fits() {
    let ops = table();
    assert_eq!(ops.reserved, 0);
    assert!(ops.last_error.is_some() && ops.version.is_some() && ops.capabilities.is_some());
    assert!(ops.lattice_diagonal.is_some() && ops.lattice_general.is_some());
    assert!(ops.lattice_free.is_some() && ops.lattice_get_info.is_some());
    assert!(ops.lattice_wrap_diffs.is_some() && ops.lattice_wrap_pairs.is_some());
    assert!(ops.lattice_nearest_diffs.is_some() && ops.lattice_nearest_pairs.is_some());
    assert!(ops.lattice_wrap_points.is_some());
    assert_eq!(
        ops.last_error.unwrap() as *const (),
        minimage_last_error as *const ()
    );

    let pointer = size_of::<usize>();
    let total = size_of::<minimage_ops>();
    let words = total / 4;
    // An older host that knows the first two operations.
    let known = 8 + 2 * pointer;
    let (status, got) = prefix(words, known as u32, minimage_ops_get);
    assert_eq!(status, MINIMAGE_STATUS_OK);
    assert_eq!(got[0] as usize, known);
    assert!(got[known / 4..].iter().all(|w| *w == u32::MAX));
    // A size inside a pointer keeps the whole pointers before it.
    let (_, got) = prefix(words, (known + pointer - 1) as u32, minimage_ops_get);
    assert_eq!(got[0] as usize, known);
    // A newer host with four more operations: they read as null.
    let (status, got) = prefix(
        words + 4 * pointer / 4,
        (total + 4 * pointer) as u32,
        minimage_ops_get,
    );
    assert_eq!(status, MINIMAGE_STATUS_OK);
    assert_eq!(got[0] as usize, total);
    assert!(got[words..].iter().all(|w| *w == 0));

    let (status, _) = prefix(words, 2, minimage_ops_get);
    assert_eq!(status, MINIMAGE_STATUS_BAD_SIZE);
    assert_eq!(
        unsafe { minimage_ops_get(ptr::null_mut()) },
        MINIMAGE_STATUS_NULL_POINTER
    );
}

#[test]
fn a_negative_edge_is_stored_as_its_magnitude() {
    for mask in 0..8 {
        let edges = negate(diagonal([15.0, 12.0, 9.0]), mask);
        let edges = [edges[0][0], edges[1][1], edges[2][2]];
        for periodic in [ALL, [true, true, false], [false, true, false]] {
            let lattice = Lattice::diagonal(edges, periodic, [0.0; 3]).unwrap();
            let info = lattice.info();
            assert_eq!(info.orthorhombic, 1);
            let stored = lattice.rows();
            for axis in 0..3 {
                if periodic[axis] {
                    assert_eq!(stored[axis][axis], edges[axis].abs());
                    assert_eq!(info.widths[axis], edges[axis].abs());
                } else {
                    assert_eq!(stored[axis][axis], edges[axis]);
                }
            }
        }
    }
    // The reported cell, against the positive one.
    let negative = Lattice::diagonal([-15.0, -15.0, 15.0], ALL, [0.0; 3]).unwrap();
    let positive = Lattice::diagonal([15.0, 15.0, 15.0], ALL, [0.0; 3]).unwrap();
    assert_eq!(negative.rows(), positive.rows());
    let ps = points(300, 7, 40.0);
    assert_eq!(negative.wrap_points(&ps), positive.wrap_points(&ps));
    assert_eq!(negative.wrap_diffs(&ps), positive.wrap_diffs(&ps));
    assert_eq!(negative.nearest_diffs(&ps), positive.nearest_diffs(&ps));
}

#[test]
fn fully_periodic_lattices_match_the_direct_calls() {
    for mask in 0..8 {
        check_lattice(negate(diagonal([15.0, 12.0, 9.0]), mask), ALL);
        check_lattice(negate(TRIANGULAR, mask), ALL);
    }
    let [a, b, c] = TRIANGULAR;
    check_lattice([b, a, c], ALL);
    check_lattice(
        [[4.0, 1.0, -2.0], [-1.5, 5.0, 0.5], [0.25, -0.75, 6.0]],
        ALL,
    );
}

#[test]
fn mixed_periodicity_keeps_the_open_axes() {
    for bits in 0..7 {
        let periodic = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
        for mask in 0..8 {
            check_lattice(negate(diagonal([15.0, 12.0, 9.0]), mask), periodic);
        }
        check_lattice(TRIANGULAR, periodic);
        check_lattice(negate(TRIANGULAR, 5), periodic);
        check_lattice(
            [[4.0, 1.0, -2.0], [-1.5, 5.0, 0.5], [0.25, -0.75, 6.0]],
            periodic,
        );
    }
    // An open axis may carry a zero or a sheared vector.
    check_lattice(
        [[15.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 0.0]],
        [true, true, false],
    );
    check_lattice(
        [[15.0, 0.0, 0.0], [3.0, 4.0, 5.0], [0.0, 0.0, -9.0]],
        [true, false, true],
    );
}

#[test]
fn points_on_a_face_wrap_into_the_half_open_cell() {
    let lattice = Lattice::diagonal([10.0, 10.0, 10.0], ALL, [0.0; 3]).unwrap();
    let on_faces = [
        [0.0, 10.0, -10.0],
        [-1e-20, 20.0, 9.999999999999998],
        [-0.0, 1e-300, -1e-300],
        [30.0, -30.0, 5.0],
    ];
    let wrapped = lattice.wrap_points(&on_faces).unwrap();
    for (p, w) in on_faces.iter().zip(&wrapped) {
        for axis in 0..3 {
            assert!((0.0..10.0).contains(&w[axis]), "{p:?} wrapped to {w:?}");
        }
    }
    assert_eq!(wrapped[0], [0.0, 0.0, 0.0]);
    assert_eq!(wrapped[1], [0.0, 0.0, 9.999999999999998]);
    assert_eq!(wrapped[3], [0.0, 0.0, 5.0]);
}

#[test]
fn a_bad_cell_has_its_own_code() {
    for bad in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for axis in 0..3 {
            let mut edges = [15.0, -12.0, 9.0];
            edges[axis] = bad;
            for bits in 0..8 {
                let periodic = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
                let built = Lattice::diagonal(edges, periodic, [0.0; 3]);
                let general = Lattice::general(diagonal(edges), periodic, [0.0; 3]);
                if periodic[axis] {
                    assert_eq!(built.err(), Some(MINIMAGE_STATUS_BAD_CELL), "{edges:?}");
                    assert!(last_error().is_some());
                    assert_eq!(general.err(), Some(MINIMAGE_STATUS_BAD_CELL), "{edges:?}");
                } else {
                    assert!(built.is_ok() && general.is_ok(), "{edges:?} {periodic:?}");
                    assert_eq!(last_error(), None);
                }
            }
        }
    }
    // Dependent periodic vectors, and a non-finite origin.
    let flat = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
    assert_eq!(
        Lattice::general(flat, ALL, [0.0; 3]).err(),
        Some(MINIMAGE_STATUS_BAD_CELL)
    );
    assert_eq!(
        Lattice::general(flat, [true, true, false], [0.0; 3]).err(),
        Some(MINIMAGE_STATUS_BAD_CELL)
    );
    assert!(Lattice::general(flat, [true, false, true], [0.0; 3]).is_ok());
    assert_eq!(
        Lattice::diagonal([1.0, 1.0, 1.0], ALL, [0.0, f64::NAN, 0.0]).err(),
        Some(MINIMAGE_STATUS_BAD_CELL)
    );
}

#[test]
fn misuse_returns_a_negative_code() {
    let ops = table();
    let lattice = Lattice::diagonal([10.0, 10.0, 10.0], ALL, [0.0; 3]).unwrap();
    let rows = [[1.0, 2.0, 3.0]; 4];
    let mut out = [[0.0; 3]; 4];
    let input: *const f64 = rows.as_ptr().cast();
    let output: *mut f64 = out.as_mut_ptr().cast();
    let batches = [
        ops.lattice_wrap_diffs.unwrap(),
        ops.lattice_nearest_diffs.unwrap(),
        ops.lattice_wrap_points.unwrap(),
    ];
    for call in batches {
        unsafe {
            assert_eq!(
                call(ptr::null(), input, 4, output),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(last_error().as_deref(), Some("null lattice"));
            assert_eq!(
                call(lattice.raw, ptr::null(), 4, output),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(
                call(lattice.raw, input, 4, ptr::null_mut()),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(
                call(lattice.raw, input, usize::MAX / 8, output),
                MINIMAGE_STATUS_BAD_COUNT
            );
            // No rows: the buffers are not read.
            assert_eq!(
                call(lattice.raw, ptr::null(), 0, ptr::null_mut()),
                MINIMAGE_STATUS_OK
            );
            assert_eq!(last_error(), None);
        }
        // An output one row into its input.
        let mut shared = [[1.0, 2.0, 3.0]; 5];
        let base: *mut f64 = shared.as_mut_ptr().cast();
        unsafe {
            assert_eq!(
                call(lattice.raw, base, 4, base.add(3)),
                MINIMAGE_STATUS_BAD_OVERLAP
            );
        }
        assert_eq!(shared, [[1.0, 2.0, 3.0]; 5]);
    }
    for call in [
        ops.lattice_wrap_pairs.unwrap(),
        ops.lattice_nearest_pairs.unwrap(),
    ] {
        unsafe {
            assert_eq!(
                call(ptr::null(), input, input, 4, output),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(
                call(lattice.raw, ptr::null(), input, 4, output),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(
                call(lattice.raw, input, ptr::null(), 4, output),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(
                call(lattice.raw, input, input, 4, ptr::null_mut()),
                MINIMAGE_STATUS_NULL_POINTER
            );
            assert_eq!(
                call(lattice.raw, input, input, usize::MAX / 8, output),
                MINIMAGE_STATUS_BAD_COUNT
            );
            assert_eq!(
                call(lattice.raw, ptr::null(), ptr::null(), 0, ptr::null_mut()),
                MINIMAGE_STATUS_OK
            );
        }
        let mut shared = [[1.0, 2.0, 3.0]; 5];
        let base: *mut f64 = shared.as_mut_ptr().cast();
        unsafe {
            assert_eq!(
                call(lattice.raw, input, base, 4, base.add(3)),
                MINIMAGE_STATUS_BAD_OVERLAP
            );
        }
    }
    unsafe {
        let edges = [1.0, 1.0, 1.0];
        let mut raw = ptr::null_mut();
        assert_eq!(
            ops.lattice_diagonal.unwrap()(ptr::null(), ptr::null(), ptr::null(), &mut raw),
            MINIMAGE_STATUS_NULL_POINTER
        );
        assert_eq!(
            ops.lattice_general.unwrap()(ptr::null(), ptr::null(), ptr::null(), &mut raw),
            MINIMAGE_STATUS_NULL_POINTER
        );
        assert_eq!(
            ops.lattice_diagonal.unwrap()(
                edges.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut()
            ),
            MINIMAGE_STATUS_NULL_POINTER
        );
        // A null mask is every axis periodic and a null origin is zero.
        assert_eq!(
            ops.lattice_diagonal.unwrap()(edges.as_ptr(), ptr::null(), ptr::null(), &mut raw),
            MINIMAGE_STATUS_OK
        );
        let mut info: minimage_lattice_info = MaybeUninit::zeroed().assume_init();
        info.size = size_of::<minimage_lattice_info>() as u32;
        assert_eq!(
            ops.lattice_get_info.unwrap()(raw, &mut info),
            MINIMAGE_STATUS_OK
        );
        assert_eq!(info.periodic, [1, 1, 1]);
        assert_eq!(info.origin, [0.0; 3]);
        assert_eq!(
            ops.lattice_get_info.unwrap()(ptr::null(), &mut info),
            MINIMAGE_STATUS_NULL_POINTER
        );
        assert_eq!(
            ops.lattice_get_info.unwrap()(raw, ptr::null_mut()),
            MINIMAGE_STATUS_NULL_POINTER
        );
        info.size = 1;
        assert_eq!(
            ops.lattice_get_info.unwrap()(raw, &mut info),
            MINIMAGE_STATUS_BAD_SIZE
        );
        // An older host that stops after the periodicity flags.
        info = MaybeUninit::zeroed().assume_init();
        info.rows = [7.0; 9];
        info.size = 24;
        assert_eq!(
            ops.lattice_get_info.unwrap()(raw, &mut info),
            MINIMAGE_STATUS_OK
        );
        assert_eq!(
            (info.size, info.orthorhombic, info.periodic),
            (24, 1, [1, 1, 1])
        );
        assert_eq!(info.rows, [7.0; 9]);
        ops.lattice_free.unwrap()(raw);
        ops.lattice_free.unwrap()(ptr::null_mut());
    }
}

#[test]
fn the_capabilities_say_what_the_calls_do() {
    let ops = table();
    let mut caps: minimage_caps = unsafe { MaybeUninit::zeroed().assume_init() };
    caps.size = size_of::<minimage_caps>() as u32;
    assert_eq!(
        unsafe { ops.capabilities.unwrap()(&mut caps) },
        MINIMAGE_STATUS_OK
    );
    assert_eq!(caps.wrap_open_sheared, 0);
    assert_eq!(caps.open_axes, 1);
    assert_eq!(caps.in_place, 1);
    // The absent capability is a status, and the output is untouched.
    let lattice = Lattice::general(TRIANGULAR, [true, true, false], [0.0; 3]).unwrap();
    let diffs = points(8, 3, 20.0);
    assert_eq!(
        lattice.wrap_diffs(&diffs).err(),
        Some(MINIMAGE_STATUS_UNSUPPORTED)
    );
    assert!(lattice.nearest_diffs(&diffs).is_ok());
    let version = unsafe { CStr::from_ptr(ops.version.unwrap()()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}
