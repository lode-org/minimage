//! Bit-compare `wrap_ortho_half_strict` with the fmod body from main.

use minimage::wrap_ortho_half_strict;

const SMALL_LENGTH: f64 = 9.332_636_185_032_189e-302;
const SMALL_SCALE: f64 = 18_446_744_073_709_551_616.0;

fn bits(x: f64) -> u64 {
    x.to_bits()
}

fn nan_class(x: f64) -> u32 {
    if !x.is_nan() {
        return 0;
    }
    if bits(x) & 0x0008_0000_0000_0000 != 0 {
        2
    } else {
        1
    }
}

fn fold_remainder(rem: f64, length: f64) -> f64 {
    let half = 0.5 * length;
    if rem > half {
        rem - length
    } else if rem < -half {
        rem + length
    } else {
        rem
    }
}

fn wrap_main_frozen(delta: f64, length: f64) -> f64 {
    if !delta.is_finite() || !length.is_finite() || length <= 0.0 {
        return f64::NAN;
    }
    let rem = delta % length;
    if length < SMALL_LENGTH {
        fold_remainder(rem * SMALL_SCALE, length * SMALL_SCALE) / SMALL_SCALE
    } else {
        fold_remainder(rem, length)
    }
}

fn next_after(x: f64, dir: f64) -> f64 {
    if x.is_nan() || dir.is_nan() {
        return f64::NAN;
    }
    if x == dir {
        return x;
    }
    if x == 0.0 {
        let tiny = f64::from_bits(1);
        return if dir > 0.0 { tiny } else { -tiny };
    }
    let mut u = x.to_bits();
    if (x > 0.0) == (dir > x) {
        u = u.wrapping_add(1);
    } else {
        u = u.wrapping_sub(1);
    }
    f64::from_bits(u)
}

struct Failures(u64);

fn check(delta: f64, length: f64, fails: &mut Failures) {
    let got = wrap_ortho_half_strict(delta, length);
    let exp = wrap_main_frozen(delta, length);
    if bits(got) != bits(exp) || nan_class(got) != nan_class(exp) {
        fails.0 += 1;
        if fails.0 <= 8 {
            eprintln!(
                "DIFF wrap({delta:?}, {length:?}) got={got:?} ref={exp:?} class {} vs {}",
                nan_class(got),
                nan_class(exp)
            );
        }
    }
}

fn check_frozen(delta: f64, length: f64, expected: f64, fails: &mut Failures) {
    let got = wrap_main_frozen(delta, length);
    if bits(got) != bits(expected) {
        fails.0 += 1;
        eprintln!("FROZEN wrap({delta:?}, {length:?}) = {got:?}, expected {expected:?}");
    }
}

fn around(base: f64, length: f64, fails: &mut Failures, edges: &mut u64) {
    if !base.is_finite() {
        check(base, length, fails);
        *edges += 1;
        return;
    }
    let mut x = base;
    for _ in 0..4 {
        check(x, length, fails);
        check(-x, length, fails);
        *edges += 2;
        x = next_after(x, f64::INFINITY);
    }
    x = base;
    for _ in 0..4 {
        x = next_after(x, f64::NEG_INFINITY);
        check(x, length, fails);
        check(-x, length, fails);
        *edges += 2;
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

#[test]
fn differential_matches_main_body() {
    assert_eq!(
        next_after(0.5, f64::INFINITY).to_bits(),
        0.5000000000000001f64.to_bits()
    );
    assert_eq!(
        next_after(0.5, f64::NEG_INFINITY).to_bits(),
        0.49999999999999994f64.to_bits()
    );

    let mut fails = Failures(0);
    let mut edges = 0u64;
    check_frozen(15.0, 10.0, 5.0, &mut fails);
    check_frozen(-15.0, 10.0, -5.0, &mut fails);
    check_frozen(6.0, 10.0, -4.0, &mut fails);
    check_frozen(0.0, 10.0, 0.0, &mut fails);
    check_frozen(-0.0, 10.0, -0.0, &mut fails);
    check_frozen(20.0, 10.0, 0.0, &mut fails);
    check_frozen(-20.0, 10.0, -0.0, &mut fails);
    check_frozen(0.45, 0.3, -0.14999999999999997, &mut fails);
    check_frozen(1e16, 3.0, 1.0, &mut fails);

    let sub = f64::from_bits(1);
    let maxv = f64::MAX;
    let lengths = [
        1.0,
        2.0,
        3.0,
        10.0,
        12.0,
        0.1,
        0.3,
        f64::from_bits(0x3f50000000000000),
        f64::from_bits(0x4090000000000000),
        SMALL_LENGTH,
        next_after(SMALL_LENGTH, 0.0),
        next_after(SMALL_LENGTH, f64::INFINITY),
        f64::from_bits(0x0020000000000000),
        f64::from_bits(0x0010000000000000),
        sub * 3.0,
        sub,
        f64::from_bits(0x3ff0000000000001),
        f64::from_bits(0x7fdfffffffffffff),
        next_after(f64::from_bits(0x7fdfffffffffffff), f64::INFINITY),
        maxv,
        1e-200,
        1e200,
        1.7e308,
    ];

    for &length in &lengths {
        if !(length > 0.0) || !length.is_finite() {
            check(1.0, length, &mut fails);
            check(0.0, length, &mut fails);
            check(-0.0, length, &mut fails);
            edges += 3;
            continue;
        }
        let half = 0.5 * length;
        check(0.0, length, &mut fails);
        check(-0.0, length, &mut fails);
        check(sub, length, &mut fails);
        check(-sub, length, &mut fails);
        check(length, length, &mut fails);
        check(-length, length, &mut fails);
        edges += 6;
        let two = length + length;
        if two.is_finite() {
            check(two, length, &mut fails);
            check(-two, length, &mut fails);
            check(next_after(two, f64::INFINITY), length, &mut fails);
            check(next_after(-two, f64::NEG_INFINITY), length, &mut fails);
            edges += 4;
        }
        around(half, length, &mut fails, &mut edges);
        around(-half, length, &mut fails, &mut edges);
        for k in 0..=64 {
            check(half * (k as f64), length, &mut fails);
            check(half * (-(k as f64)), length, &mut fails);
            edges += 2;
        }
        for e in 0..=60 {
            let k = (1u64 << e) as f64;
            check(half * k, length, &mut fails);
            check(half * -k, length, &mut fails);
            edges += 2;
            if e >= 1 {
                check(half * (k - 1.0), length, &mut fails);
                check(half * -(k - 1.0), length, &mut fails);
                check(half * (k + 1.0), length, &mut fails);
                check(half * -(k + 1.0), length, &mut fails);
                edges += 4;
            }
        }
    }

    let specials = [
        (f64::NAN, 10.0),
        (f64::INFINITY, 10.0),
        (f64::NEG_INFINITY, 10.0),
        (6.0, 0.0),
        (6.0, -0.0),
        (6.0, -10.0),
        (6.0, -sub),
        (6.0, f64::NAN),
        (6.0, f64::INFINITY),
        (6.0, f64::NEG_INFINITY),
        (f64::NAN, -1.0),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
    ];
    for (delta, length) in specials {
        check(delta, length, &mut fails);
        edges += 1;
    }
    let payloads = [0x1u64, 0x7ffff, 0x80000, 0x123456789, 0x000f_ffff_ffff_ffff];
    for payload in payloads {
        let quiet = 0x7ff8_0000_0000_0000 | (payload & 0x7_ffff);
        let signal = 0x7ff0_0000_0000_0000 | (payload & 0x7_ffff_ffff_ffff);
        if signal & 0x000f_ffff_ffff_ffff == 0 {
            continue;
        }
        for word in [
            quiet,
            quiet | 0x8000_0000_0000_0000,
            signal,
            signal | 0x8000_0000_0000_0000,
        ] {
            let value = f64::from_bits(word);
            check(value, 10.0, &mut fails);
            check(1.0, value, &mut fails);
            edges += 2;
        }
    }

    let mut rng = Rng(0x1234_5678_9abc_def);
    let randoms = 10_000_000u64;
    for i in 0..randoms {
        let kind = i % 5;
        let (delta, length) = if kind == 0 {
            let exp = (rng.next() % 2200) as i32 - 1100;
            let eb = exp + 1023;
            let mant = rng.next() & ((1u64 << 52) - 1);
            let length = if !(1..2047).contains(&eb) {
                f64::from_bits(rng.next() & 0x000f_ffff_ffff_ffff)
            } else {
                f64::from_bits(((eb as u64) << 52) | mant)
            };
            (f64::from_bits(rng.next()), length)
        } else if kind == 1 {
            let exp = (rng.next() % 2000) as i32 - 1000;
            let mut eb = exp + 1023;
            if !(1..2047).contains(&eb) {
                eb = 1023;
            }
            let length = f64::from_bits(((eb as u64) << 52) | (rng.next() & ((1u64 << 52) - 1)));
            let frac = ((rng.next() >> 11) as f64) / ((1u64 << 53) as f64);
            ((frac * 8.0 - 4.0) * length, length)
        } else if kind == 2 {
            let mut length = f64::from_bits((1023u64 << 52) | (rng.next() & ((1u64 << 52) - 1)));
            let shift = (rng.next() % 400) as i32 - 200;
            length = length * (2.0f64).powi(shift);
            let k = (rng.next() % 17) as i32 - 8;
            let mut delta = (k as f64 + 0.5) * length;
            let steps = rng.next() % 5;
            let dir = if rng.next() & 1 == 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            for _ in 0..steps {
                delta = next_after(delta, dir);
            }
            (delta, length)
        } else if kind == 3 {
            let length =
                f64::from_bits(((900 + 1023) as u64) << 52 | (rng.next() & ((1u64 << 52) - 1)));
            let mut delta = f64::from_bits(rng.next() & 0x000f_ffff_ffff_ffff);
            if rng.next() & 1 == 1 {
                delta = -delta;
            }
            if i & 1 == 0 {
                (length, delta.abs())
            } else {
                (delta, length)
            }
        } else {
            let length = if rng.next() & 3 == 0 {
                if rng.next() & 1 == 0 {
                    0.0
                } else {
                    -f64::from_bits(rng.next()).abs()
                }
            } else {
                f64::from_bits(rng.next()).abs()
            };
            let special = rng.next() % 5;
            let delta = if special == 0 {
                f64::INFINITY
            } else if special == 1 {
                f64::NEG_INFINITY
            } else if special == 2 {
                f64::from_bits(0x7ff8_0000_0000_0001 | (rng.next() & 0xffff))
            } else {
                f64::from_bits(rng.next())
            };
            (delta, length)
        };
        check(delta, length, &mut fails);
    }

    eprintln!(
        "diff_edges={edges} diff_random={randoms} diff_fail={}",
        fails.0
    );
    assert_eq!(fails.0, 0);
}
