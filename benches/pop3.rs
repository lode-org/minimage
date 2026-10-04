//! POP3 profile of the minimum-image hot path.
//!
//! Performance Optimisation and Productivity 3 (EuroHPC, 2024–2026)
//! judges a region by a hierarchy of efficiencies in `[0, 1]`. Global
//! efficiency is parallel efficiency times computational scaling.
//! Parallel efficiency splits into load balance and communication, and
//! communication splits into serialisation and transfer. A leaf below
//! 0.8 is the one to open.
//!
//! This library is one thread, so those parallel leaves are not a
//! process count. The regions are the orthorhombic batch against the
//! per-pair wrap, Rapaport's shifted bin, a Smith-hit Euclidean query,
//! the Selling closest point, and an index-list bin. Cycles and
//! instructions come from `perf_event_open` when the kernel allows the
//! call.

use std::hint::black_box;
use std::time::Instant;

use minimage::Cell;

const PAIRS: usize = 16_384;

fn main() {
    let ortho = Cell::ortho(10.0, 11.0, 12.0).expect("ortho");
    let skew = Cell::from_vectors(
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    )
    .expect("skew");
    let near = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [1.0, 10.0, 0.0],
        [0.5, 0.2, 10.0],
        [0.0, 0.0, 0.0],
    )
    .expect("near");
    let qs = positions(PAIRS, 40.0);
    let skew_qs = positions(PAIRS, 8.0);
    let near_qs = positions(PAIRS, 2.0);
    let indices: Vec<usize> = (0..PAIRS).collect();
    let mut out = vec![0.0; PAIRS];
    check_ortho(&ortho, &qs, &mut out);

    let counters = Group::open();
    eprintln!(
        "counters={}",
        if counters.is_some() {
            "perf_event_open"
        } else {
            "refused"
        }
    );

    let p = [0.2, 0.3, 0.4];
    let shift = [10.0, 0.0, 0.0];
    let regions = [
        region("ortho_batch", PAIRS, &counters, || {
            ortho.dist2_many_unchecked(&qs, p, &mut out);
        }),
        region("ortho_pair", PAIRS, &counters, || {
            for (q, slot) in qs.iter().zip(out.iter_mut()) {
                *slot = ortho.dist2(p, *q);
            }
            black_box(&out);
        }),
        region("shifted_batch", PAIRS, &counters, || {
            ortho
                .dist2_shifted_many(p, &qs, shift, &mut out)
                .expect("shifted");
            black_box(&out);
        }),
        region("shifted_pair", PAIRS, &counters, || {
            for (q, slot) in qs.iter().zip(out.iter_mut()) {
                *slot = ortho.dist2_shifted(p, *q, shift);
            }
            black_box(&out);
        }),
        region("skew_engine", PAIRS, &counters, || {
            for (q, slot) in skew_qs.iter().zip(out.iter_mut()) {
                let d = skew.displacement(p, *q);
                *slot = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            }
            black_box(&out);
        }),
        region("skew_euclidean", PAIRS, &counters, || {
            for (q, slot) in skew_qs.iter().zip(out.iter_mut()) {
                let d = skew.displacement_euclidean(p, *q);
                *slot = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            }
            black_box(&out);
        }),
        region("skew_near", PAIRS, &counters, || {
            for (q, slot) in near_qs.iter().zip(out.iter_mut()) {
                let d = near.displacement_euclidean(p, *q);
                *slot = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            }
            black_box(&out);
        }),
        region("indexed_batch", PAIRS, &counters, || {
            ortho
                .dist2_shifted_indexed(p, &qs, &indices, shift, &mut out)
                .expect("indexed");
            black_box(&out);
        }),
        region("indexed_scalar", PAIRS, &counters, || {
            for (idx, slot) in indices.iter().zip(out.iter_mut()) {
                *slot = ortho.dist2_shifted(p, qs[*idx], shift);
            }
            black_box(&out);
        }),
        region("indexed_occ1", PAIRS, &counters, || {
            for (idx, slot) in indices.iter().zip(out.iter_mut()) {
                ortho
                    .dist2_shifted_indexed(p, &qs, &[*idx], shift, std::slice::from_mut(slot))
                    .expect("one");
            }
            black_box(&out);
        }),
    ];

    println!(
        "{:<16} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "region", "pairs", "ns/pair", "instr/pair", "ipc", "time_scale", "instr_scale"
    );
    let ortho_pair = &regions[1];
    let shifted_pair = &regions[3];
    let skew_engine = &regions[4];
    let indexed_scalar = &regions[8];
    for row in &regions {
        let (time_ref, instr_ref) = match row.name.as_str() {
            "ortho_batch" => (Some(ortho_pair), Some(ortho_pair)),
            "shifted_batch" => (Some(shifted_pair), Some(shifted_pair)),
            "skew_euclidean" | "skew_near" => (Some(skew_engine), Some(skew_engine)),
            "indexed_batch" | "indexed_occ1" => (Some(indexed_scalar), Some(indexed_scalar)),
            _ => (None, None),
        };
        let time_scale = time_ref.map(|r| r.ns_per / row.ns_per).unwrap_or(1.0);
        let instr_scale = match (
            instr_ref,
            row.instr_per,
            instr_ref.and_then(|r| r.instr_per),
        ) {
            (Some(_), Some(here), Some(there)) if here > 0.0 => there / here,
            _ => f64::NAN,
        };
        println!(
            "{:<16} {:>10} {:>12.3} {:>12} {:>10} {:>10.3} {:>10}",
            row.name,
            row.pairs,
            row.ns_per,
            fmt_opt(row.instr_per),
            fmt_opt(row.ipc),
            time_scale,
            fmt_opt(if instr_scale.is_nan() {
                None
            } else {
                Some(instr_scale)
            }),
        );
        println!(
            "{} ns_per_pair={:.6} instr_per_pair={} ipc={} time_scale={:.6} instr_scale={}",
            row.name,
            row.ns_per,
            fmt_opt(row.instr_per),
            fmt_opt(row.ipc),
            time_scale,
            fmt_opt(if instr_scale.is_nan() {
                None
            } else {
                Some(instr_scale)
            }),
        );
    }

    let sweep_n = [4usize, 16, 64, 256, 1024, 4096, 16_384];
    let mut sweep = Vec::new();
    for n in sweep_n {
        let sample = region(&format!("sweep_{n}"), n, &counters, || {
            ortho.dist2_many_unchecked(&qs[..n], p, &mut out[..n]);
        });
        sweep.push(sample);
    }
    let best = sweep.iter().map(|s| s.ns_per).fold(f64::INFINITY, f64::min);
    println!(
        "{:<16} {:>10} {:>12} {:>10}",
        "batch", "pairs", "ns/pair", "vs_best"
    );
    for row in &sweep {
        let eff = best / row.ns_per;
        println!(
            "{:<16} {:>10} {:>12.3} {:>10.3}{}",
            row.name,
            row.pairs,
            row.ns_per,
            eff,
            if eff < 0.8 { "  below 0.8" } else { "" }
        );
        println!(
            "{} ns_per_pair={:.6} efficiency={:.6}",
            row.name, row.ns_per, eff
        );
    }
}

struct Row {
    name: String,
    pairs: usize,
    ns_per: f64,
    instr_per: Option<f64>,
    ipc: Option<f64>,
}

fn region(name: &str, pairs: usize, counters: &Option<Group>, mut body: impl FnMut()) -> Row {
    let repeats = calibrate(pairs, &mut body);
    let (ns, counts) = measure(repeats, counters, &mut body);
    let total_pairs = (pairs as u64) * (repeats as u64);
    let ns_per = ns / total_pairs as f64;
    let (instr_per, ipc) = match counts {
        Some((cycles, instr)) if cycles > 0 => (
            Some(instr as f64 / total_pairs as f64),
            Some(instr as f64 / cycles as f64),
        ),
        _ => (None, None),
    };
    Row {
        name: name.to_string(),
        pairs,
        ns_per,
        instr_per,
        ipc,
    }
}

fn calibrate(pairs: usize, body: &mut impl FnMut()) -> u32 {
    let start = Instant::now();
    let mut n = 0u32;
    while start.elapsed().as_millis() < 30 && n < 10_000 {
        body();
        n += 1;
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let target = if pairs >= 4096 { 0.08 } else { 0.04 };
    ((target / elapsed) * n as f64).clamp(1.0, 10_000.0) as u32
}

fn measure(
    repeats: u32,
    counters: &Option<Group>,
    body: &mut impl FnMut(),
) -> (f64, Option<(u64, u64)>) {
    let start = Instant::now();
    let counts = if let Some(group) = counters {
        group.with(|| {
            for _ in 0..repeats {
                body();
            }
        })
    } else {
        for _ in 0..repeats {
            body();
        }
        None
    };
    (start.elapsed().as_secs_f64() * 1e9, counts)
}

fn check_ortho(cell: &Cell, qs: &[[f64; 3]], out: &mut [f64]) {
    let p = [0.0, 0.0, 0.0];
    let q = [25.0, -18.0, 3.0];
    let got = cell.dist2(p, q);
    if (got - 50.0).abs() > 1e-9 {
        eprintln!("ortho pair d2 {got} wanted 50");
        std::process::exit(1);
    }
    cell.dist2_many_unchecked(qs, p, out);
    for (q, slot) in qs.iter().zip(out.iter()).take(32) {
        let one = cell.dist2(p, *q);
        if (one - *slot).abs() > 1e-8 {
            eprintln!("batch disagrees with the pair wrap");
            std::process::exit(1);
        }
    }
}

fn positions(n: usize, span: f64) -> Vec<[f64; 3]> {
    let mut s = 0x1234_5678_9abc_u64;
    (0..n)
        .map(|_| {
            let mut p = [0.0; 3];
            for slot in &mut p {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = ((s >> 33) as f64) / (1u64 << 31) as f64;
                *slot = u * span - span / 2.0;
            }
            p
        })
        .collect()
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.3}"),
        None => "-".to_string(),
    }
}

trait BatchOrtho {
    fn dist2_many_unchecked(&self, qs: &[[f64; 3]], p: [f64; 3], out: &mut [f64]);
}

impl BatchOrtho for Cell {
    fn dist2_many_unchecked(&self, qs: &[[f64; 3]], p: [f64; 3], out: &mut [f64]) {
        minimage::dist2_many(self, p, qs, out).expect("dist2_many");
        black_box(out);
    }
}

struct Group {
    cycles: i32,
    instr: i32,
}

impl Group {
    fn open() -> Option<Self> {
        let cycles = open_event(0, -1)?;
        let instr = open_event(1, cycles)?;
        Some(Self { cycles, instr })
    }

    fn with(&self, work: impl FnOnce()) -> Option<(u64, u64)> {
        unsafe {
            if ioctl(self.cycles, IOC_RESET) < 0 || ioctl(self.instr, IOC_RESET) < 0 {
                return None;
            }
            if ioctl(self.cycles, IOC_ENABLE) < 0 {
                return None;
            }
        }
        work();
        unsafe {
            if ioctl(self.cycles, IOC_DISABLE) < 0 {
                return None;
            }
        }
        let cycles = read_count(self.cycles)?;
        let instr = read_count(self.instr)?;
        Some((cycles, instr))
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        unsafe {
            close(self.cycles);
            close(self.instr);
        }
    }
}

fn open_event(config: u64, group: i32) -> Option<i32> {
    let mut buf = [0u8; 128];
    let size: u32 = 128;
    buf[0..4].copy_from_slice(&0u32.to_ne_bytes());
    buf[4..8].copy_from_slice(&size.to_ne_bytes());
    buf[8..16].copy_from_slice(&config.to_ne_bytes());
    // disabled, exclude kernel, exclude hypervisor
    let flags: u64 = 1 | (1 << 5) | (1 << 6);
    buf[40..48].copy_from_slice(&flags.to_ne_bytes());
    let fd = unsafe { syscall(SYS_PERF_EVENT_OPEN, buf.as_ptr(), 0i32, -1i32, group, 8u64) };
    if fd < 0 {
        None
    } else {
        Some(fd as i32)
    }
}

fn read_count(fd: i32) -> Option<u64> {
    let mut buf = [0u8; 8];
    unsafe {
        if lseek(fd, 0, 0) < 0 {
            return None;
        }
        if pread_bytes(fd, buf.as_mut_ptr().cast::<std::ffi::c_void>(), 8) != 8 {
            return None;
        }
    }
    Some(u64::from_ne_bytes(buf))
}

const SYS_PERF_EVENT_OPEN: i64 = 298;
const IOC_ENABLE: u64 = 0x2400;
const IOC_DISABLE: u64 = 0x2401;
const IOC_RESET: u64 = 0x2403;

unsafe extern "C" {
    fn syscall(n: i64, ...) -> i64;
    fn ioctl(fd: i32, req: u64, ...) -> i32;
    fn close(fd: i32) -> i32;
    fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
    #[link_name = "read"]
    fn pread_bytes(fd: i32, buf: *mut std::ffi::c_void, count: usize) -> isize;
}
