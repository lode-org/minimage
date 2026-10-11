//! Bit identity of every usable SIMD path against the scalar path.
//!
//! The golden file records SHA-256 of the little-endian output bytes.
//! x86_64 and aarch64 runners must produce the same digest.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::simd::{self, Path};
use crate::Cell;

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/isa-hashes.toml");

fn sha(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let dig = hasher.finalize();
    let mut out = String::with_capacity(dig.len() * 2);
    for b in dig {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn push_f64(buf: &mut Vec<u8>, xs: &[f64]) {
    for x in xs {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn push_f64_rows(buf: &mut Vec<u8>, rows: &[[f64; 3]]) {
    for row in rows {
        push_f64(buf, row);
    }
}

fn push_u64_rows(buf: &mut Vec<u8>, rows: &[[u64; 3]]) {
    for row in rows {
        for lane in row {
            buf.extend_from_slice(&lane.to_le_bytes());
        }
    }
}

fn scalars() -> Vec<f64> {
    let one_ulp = f64::from_bits(1.0_f64.to_bits() - 1);
    vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        one_ulp,
        -one_ulp,
        4_503_599_627_370_496.0,
        -4_503_599_627_370_496.0,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
        0.5,
        -0.5,
        5.0,
        -5.0,
        9.2,
        -15.0,
        1e-320,
        -1e-320,
        10.0,
        -11.0,
        0.1,
        -0.1,
    ]
}

fn points() -> Vec<[f64; 3]> {
    let s = scalars();
    (0..67)
        .map(|i| {
            [
                s[i % s.len()],
                s[(i * 3 + 1) % s.len()],
                s[(i * 5 + 2) % s.len()],
            ]
        })
        .collect()
}

fn cells() -> Vec<(&'static str, Cell)> {
    vec![
        ("ortho", Cell::ortho(10.0, 11.0, 12.0).unwrap()),
        (
            "tri",
            Cell::from_vectors(
                [10.0, 0.0, 0.0],
                [5.0, 8.660254037844386, 0.0],
                [0.0, 0.0, 10.0],
                [0.0; 3],
            )
            .unwrap(),
        ),
        (
            "general",
            Cell::from_vectors(
                [9.0, 3.0, 2.0],
                [1.0, 9.0, -2.0],
                [-1.5, 2.0, 9.5],
                [0.25, -0.5, 0.0],
            )
            .unwrap(),
        ),
    ]
}

fn digest() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let s = scalars();
    let mut round = Vec::new();
    let mut floor = Vec::new();
    for x in &s {
        round.push(crate::kernel::round_away(*x));
        floor.push(crate::kernel::floor_fast(*x));
    }
    let mut buf = Vec::new();
    push_f64(&mut buf, &round);
    out.insert("round_away".to_string(), sha(&buf));
    buf.clear();
    push_f64(&mut buf, &floor);
    out.insert("floor_fast".to_string(), sha(&buf));

    let ps = points();
    let qs = points().into_iter().rev().collect::<Vec<_>>();
    for (name, cell) in cells() {
        let n = ps.len();
        let mut dist = vec![0.0; n];
        crate::dist2_many(&cell, ps[0], &qs, &mut dist).unwrap();
        buf.clear();
        push_f64(&mut buf, &dist);
        out.insert(format!("dist2_many_{name}"), sha(&buf));

        crate::dist2_pairs(&cell, &ps, &qs, &mut dist).unwrap();
        buf.clear();
        push_f64(&mut buf, &dist);
        out.insert(format!("dist2_pairs_{name}"), sha(&buf));

        let mut wrapped = vec![[0.0; 3]; n];
        let diffs: Vec<[f64; 3]> = ps
            .iter()
            .zip(&qs)
            .map(|(p, q)| [q[0] - p[0], q[1] - p[1], q[2] - p[2]])
            .collect();
        crate::wrap_many(&cell, &diffs, &mut wrapped).unwrap();
        buf.clear();
        push_f64_rows(&mut buf, &wrapped);
        out.insert(format!("wrap_many_{name}"), sha(&buf));

        let mut fixed = vec![[0u64; 3]; n];
        crate::fixed_many(&cell, &ps, &mut fixed).unwrap();
        buf.clear();
        push_u64_rows(&mut buf, &fixed);
        out.insert(format!("fixed_many_{name}"), sha(&buf));

        let mut fixed_q = vec![[0u64; 3]; n];
        crate::fixed_many(&cell, &qs, &mut fixed_q).unwrap();
        crate::dist2_many_fixed(&cell, fixed[0], &fixed_q, &mut dist).unwrap();
        buf.clear();
        push_f64(&mut buf, &dist);
        out.insert(format!("dist2_many_fixed_{name}"), sha(&buf));

        crate::dist2_pairs_fixed(&cell, &fixed, &fixed_q, &mut dist).unwrap();
        buf.clear();
        push_f64(&mut buf, &dist);
        out.insert(format!("dist2_pairs_fixed_{name}"), sha(&buf));

        let mut f32p = vec![[0u32; 3]; n];
        let mut f32q = vec![[0u32; 3]; n];
        crate::fixed32_many(&cell, &ps, &mut f32p).unwrap();
        crate::fixed32_many(&cell, &qs, &mut f32q).unwrap();
        crate::dist2_many_fixed32(&cell, f32p[0], &f32q, &mut dist).unwrap();
        buf.clear();
        push_f64(&mut buf, &dist);
        out.insert(format!("dist2_many_fixed32_{name}"), sha(&buf));

        crate::dist2_pairs_fixed32(&cell, &f32p, &f32q, &mut dist).unwrap();
        buf.clear();
        push_f64(&mut buf, &dist);
        out.insert(format!("dist2_pairs_fixed32_{name}"), sha(&buf));
    }

    let ortho = Cell::ortho(10.0, 11.0, 12.0).unwrap();
    let dx: Vec<f64> = ps.iter().map(|p| p[0]).collect();
    let dy: Vec<f64> = ps.iter().map(|p| p[1]).collect();
    let dz: Vec<f64> = ps.iter().map(|p| p[2]).collect();
    let mut dist = vec![0.0; ps.len()];
    crate::dist2_ortho_diffs(&dx, &dy, &dz, 10.0, 11.0, 12.0, &mut dist).unwrap();
    buf.clear();
    push_f64(&mut buf, &dist);
    out.insert("dist2_ortho_diffs".to_string(), sha(&buf));

    ortho
        .dist2_shifted_many(ps[0], &qs, [10.0, -11.0, 0.0], &mut dist)
        .unwrap();
    buf.clear();
    push_f64(&mut buf, &dist);
    out.insert("dist2_shifted_many".to_string(), sha(&buf));
    out
}

fn render(map: &BTreeMap<String, String>) -> String {
    let mut body = String::from(
        "# SHA-256 of little-endian output bytes on the scalar path.\n\
         # The same digest is required on x86_64 and aarch64.\n\
         # Recipe: 24 awkward scalars (signed zero, subnormals, 1-ulp below 1,\n\
         # 2^52, a quiet NaN payload, halves) cycled into 67 packed triples.\n\
         # Cells: ortho 10x11x12, restricted triclinic, general triclinic with origin.\n\
         # round_away and floor_fast hash those scalars directly.\n\
         # Set MINIMAGE_WRITE_GOLDEN=1 to regenerate from the scalar path.\n",
    );
    for (k, v) in map {
        body.push_str(k);
        body.push_str(" = \"");
        body.push_str(v);
        body.push_str("\"\n");
    }
    body
}

fn load_golden() -> String {
    std::fs::read_to_string(GOLDEN).unwrap_or_default()
}

#[test]
fn every_usable_path_matches_the_scalar_hash() {
    let scalar = {
        let _guard = simd::force(Path::Scalar);
        digest()
    };
    if std::env::var_os("MINIMAGE_WRITE_GOLDEN").is_some() {
        let dir = std::path::Path::new(GOLDEN).parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(GOLDEN, render(&scalar)).unwrap();
    }
    let golden = load_golden();
    for (k, v) in &scalar {
        let line = format!("{k} = \"{v}\"");
        assert!(
            golden.contains(&line),
            "scalar digest for {k} is not in tests/golden/isa-hashes.toml ({v})"
        );
    }

    let mut exercised = Vec::new();
    let mut skipped = Vec::new();
    let compare = simd::paths_to_compare();
    for path in [
        Path::Scalar,
        Path::Avx,
        Path::Avx2,
        Path::Avx512,
        Path::Neon,
    ] {
        if compare.contains(&path) {
            exercised.push(path.name());
        } else {
            skipped.push(path.name());
            continue;
        }
        if path == Path::Scalar {
            continue;
        }
        for threads in [1usize, 2, 3, 8] {
            let mut hashes = Vec::new();
            std::thread::scope(|scope| {
                let mut joins = Vec::new();
                for _ in 0..threads {
                    joins.push(scope.spawn(|| {
                        let _guard = simd::force(path);
                        digest()
                    }));
                }
                for join in joins {
                    hashes.push(join.join().expect("worker"));
                }
            });
            for got in hashes {
                for (k, v) in &scalar {
                    assert_eq!(
                        got.get(k).map(String::as_str),
                        Some(v.as_str()),
                        "{path:?} {threads} threads diverged on {k}"
                    );
                }
            }
        }
    }
    println!("simd paths exercised: {}", exercised.join(", "));
    println!("simd paths not exercised: {}", skipped.join(", "));
}

/// Interleaved per-path timings. Ignored in the default suite.
///
/// `cargo test --release --lib bench_dispatch -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_dispatch() {
    use std::hint::black_box;

    let ortho = Cell::ortho(10.0, 11.0, 12.0).unwrap();
    let sizes = [1_000usize, 100_000];
    let paths = simd::paths_to_compare();
    println!("cpu_path_native {}", simd::native().name());
    for n in sizes {
        let ps = (0..n)
            .map(|i| {
                let t = i as f64 * 0.17;
                [t.sin() * 9.0, t.cos() * 8.0, (t * 0.3).sin() * 7.0]
            })
            .collect::<Vec<_>>();
        let qs = (0..n)
            .map(|i| {
                let t = i as f64 * 0.13 + 1.0;
                [t.cos() * 9.0, t.sin() * 8.0, (t * 0.2).cos() * 7.0]
            })
            .collect::<Vec<_>>();
        let mut out = vec![0.0; n];
        let mut wrapped = vec![[0.0; 3]; n];
        let diffs: Vec<[f64; 3]> = ps
            .iter()
            .zip(&qs)
            .map(|(p, q)| [q[0] - p[0], q[1] - p[1], q[2] - p[2]])
            .collect();
        let dx: Vec<f64> = ps.iter().map(|p| p[0]).collect();
        let dy: Vec<f64> = ps.iter().map(|p| p[1]).collect();
        let dz: Vec<f64> = ps.iter().map(|p| p[2]).collect();
        let reps = if n >= 100_000 { 8u32 } else { 40u32 };
        for round in 0..5 {
            for path in &paths {
                let _guard = simd::force(*path);
                let time = |body: &mut dyn FnMut()| {
                    let start = std::time::Instant::now();
                    for _ in 0..reps {
                        body();
                    }
                    start.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
                };
                let pairs = time(&mut || {
                    crate::dist2_pairs(&ortho, &ps, &qs, &mut out).unwrap();
                    black_box(&out);
                });
                let many = time(&mut || {
                    crate::dist2_many(&ortho, ps[0], &qs, &mut out).unwrap();
                    black_box(&out);
                });
                let wrap = time(&mut || {
                    crate::wrap_many(&ortho, &diffs, &mut wrapped).unwrap();
                    black_box(&wrapped);
                });
                let diffs_ms = time(&mut || {
                    crate::dist2_ortho_diffs(&dx, &dy, &dz, 10.0, 11.0, 12.0, &mut out).unwrap();
                    black_box(&out);
                });
                println!(
                    "bench n={n} path={} round={round} dist2_pairs={pairs:.4} dist2_many={many:.4} wrap_many={wrap:.4} dist2_ortho_diffs={diffs_ms:.4}",
                    path.name()
                );
            }
        }
    }
}
