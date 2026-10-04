//! Same driver against this tree and against published 0.1.2.
//!
//! Only calls that exist on both. Prints `name ns_per_pair=<f64>`.
//! Construction is reported and is not a speed gate.

use std::hint::black_box;
use std::time::Instant;

use minimage::Cell;

const N: usize = 4096;

fn main() {
    let ortho = Cell::ortho(10.0, 11.0, 12.0).expect("ortho");
    let inbox = ortho.dist2([0.2, 0.0, 0.0], [9.4, 0.0, 0.0]);
    if (inbox - 0.64).abs() > 1e-9 {
        eprintln!("in-box ortho dist2 {inbox} wanted 0.64");
        std::process::exit(1);
    }
    let skew = Cell::from_vectors(
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    )
    .expect("skew");
    let lattice = skew.dist2_euclidean([0.0, 0.0, 0.0], [0.02, -0.02, 0.0]);
    if lattice > 1e-12 {
        eprintln!("skew lattice residual {lattice}");
        std::process::exit(1);
    }

    let (ps, qs) = inbox_pairs(N);
    let mut out = vec![0.0; N];
    minimage::dist2_many(&ortho, ps[0], &qs, &mut out).expect("many");
    for i in 0..32 {
        let one = ortho.dist2(ps[0], qs[i]);
        if (one - out[i]).abs() > 1e-8 {
            eprintln!("dist2_many disagrees with dist2");
            std::process::exit(1);
        }
    }
    minimage::dist2_pairs(&ortho, &ps, &qs, &mut out).expect("pairs");
    for i in 0..32 {
        let one = ortho.dist2(ps[i], qs[i]);
        if (one - out[i]).abs() > 1e-8 {
            eprintln!("dist2_pairs disagrees with dist2");
            std::process::exit(1);
        }
    }

    let near = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [1.0, 10.0, 0.0],
        [0.5, 0.2, 10.0],
        [0.0, 0.0, 0.0],
    )
    .expect("near");
    let near_q = span_points(N, 2.0);
    let far_q = span_points(N, 8.0);
    let origin = [0.0, 0.0, 0.0];

    report("ortho_pair", N, || {
        for i in 0..N {
            black_box(ortho.dist2(ps[i], qs[i]));
        }
    });
    report("ortho_many", N, || {
        minimage::dist2_many(&ortho, ps[0], &qs, &mut out).expect("many");
        black_box(&out);
    });
    report("ortho_pairs", N, || {
        minimage::dist2_pairs(&ortho, &ps, &qs, &mut out).expect("pairs");
        black_box(&out);
    });
    report("eucl_near", N, || {
        for q in &near_q {
            black_box(near.displacement_euclidean(origin, *q));
        }
    });
    report("eucl_far", N, || {
        for q in &far_q {
            black_box(skew.displacement_euclidean(origin, *q));
        }
    });
    report("construct_skew", N, || {
        for _ in 0..N {
            black_box(
                Cell::from_vectors(
                    [1.0, 0.0, 0.0],
                    [0.99, 0.01, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                )
                .expect("skew"),
            );
        }
    });
}

fn report(name: &str, pairs: usize, mut body: impl FnMut()) {
    let start = Instant::now();
    let mut warm = 0u32;
    while start.elapsed().as_millis() < 30 && warm < 2_000 {
        body();
        warm += 1;
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let repeats = ((0.10 / elapsed) * f64::from(warm)).clamp(1.0, 4_000.0) as u32;
    let start = Instant::now();
    for _ in 0..repeats {
        body();
    }
    let ns = start.elapsed().as_secs_f64() * 1e9;
    let ns_per = ns / (f64::from(repeats) * pairs as f64);
    println!("{name} ns_per_pair={ns_per:.6}");
}

fn inbox_pairs(n: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut s = 0x1234_5678_9abc_u64;
    let mut next = |span: f64| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((s >> 33) as f64) / (1u64 << 31) as f64;
        u * span
    };
    let mut ps = Vec::with_capacity(n);
    let mut qs = Vec::with_capacity(n);
    let lens = [10.0, 11.0, 12.0];
    for _ in 0..n {
        let mut p = [0.0; 3];
        let mut q = [0.0; 3];
        for a in 0..3 {
            p[a] = next(lens[a]);
            q[a] = next(lens[a]);
        }
        ps.push(p);
        qs.push(q);
    }
    (ps, qs)
}

fn span_points(n: usize, span: f64) -> Vec<[f64; 3]> {
    let mut s = 0x00c0_ffee_u64;
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
