#include "minimage.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

int failures = 0;
long long edges = 0;
long long randoms = 0;

std::uint64_t bits(double x) {
  std::uint64_t out = 0;
  std::memcpy(&out, &x, sizeof out);
  return out;
}

int nan_class(double x) {
  if (!std::isnan(x)) {
    return 0;
  }
  const std::uint64_t quiet = 0x0008000000000000ULL;
  return (bits(x) & quiet) != 0 ? 2 : 1;
}

double fold_remainder(double rem, double length) {
  const double half = 0.5 * length;
  if (rem > half) {
    return rem - length;
  }
  if (rem < -half) {
    return rem + length;
  }
  return rem;
}

// Frozen body of wrap_ortho_half_strict at 2c58f45af19a.
double wrap_main_frozen(double delta, double length) {
  if (!std::isfinite(delta) || !std::isfinite(length) || length <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double rem = std::fmod(delta, length);
  constexpr double small_length = 0x1p-1000;
  constexpr double small_scale = 0x1p64;
  if (length < small_length) {
    return fold_remainder(rem * small_scale, length * small_scale) /
           small_scale;
  }
  return fold_remainder(rem, length);
}

void differ(double delta, double length, const char *kind) {
  if (failures < 8) {
    const double got = minimage::wrap_ortho_half_strict(delta, length);
    const double exp = wrap_main_frozen(delta, length);
    std::cerr.precision(17);
    std::cerr << "DIFF " << kind << " wrap(" << delta << ", " << length
              << ") got " << got << " ref " << exp << " class " << nan_class(got)
              << " vs " << nan_class(exp) << "\n";
  }
  ++failures;
}

void check(double delta, double length, bool edge) {
  const double got = minimage::wrap_ortho_half_strict(delta, length);
  const double exp = wrap_main_frozen(delta, length);
  if (edge) {
    ++edges;
  } else {
    ++randoms;
  }
  if (bits(got) != bits(exp) || nan_class(got) != nan_class(exp)) {
    differ(delta, length, edge ? "edge" : "random");
  }
}

void check_frozen(double delta, double length, double expected) {
  const double got = wrap_main_frozen(delta, length);
  if (bits(got) != bits(expected)) {
    std::cerr.precision(17);
    std::cerr << "FROZEN wrap(" << delta << ", " << length << ") = " << got
              << ", expected " << expected << "\n";
    ++failures;
  }
}

std::uint64_t rng = 0x123456789abcdefULL;

std::uint64_t next_rng() {
  std::uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

double from_bits(std::uint64_t u) {
  double x = 0;
  std::memcpy(&x, &u, sizeof x);
  return x;
}

void around(double base, double length) {
  if (!std::isfinite(base)) {
    check(base, length, true);
    return;
  }
  double x = base;
  for (int i = 0; i < 4; ++i) {
    check(x, length, true);
    check(-x, length, true);
    x = std::nextafter(x, std::numeric_limits<double>::infinity());
  }
  x = base;
  for (int i = 0; i < 4; ++i) {
    x = std::nextafter(x, -std::numeric_limits<double>::infinity());
    check(x, length, true);
    check(-x, length, true);
  }
}

void edge_suite() {
  const double sub = std::numeric_limits<double>::denorm_min();
  const double maxv = std::numeric_limits<double>::max();
  const double inf = std::numeric_limits<double>::infinity();
  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const double lengths[] = {
      1.0,
      2.0,
      3.0,
      10.0,
      12.0,
      0.1,
      0.3,
      0x1p-10,
      0x1p10,
      0x1p-100,
      0x1p100,
      0x1p-1000,
      std::nextafter(0x1p-1000, 0.0),
      std::nextafter(0x1p-1000, inf),
      0x1p-1021,
      0x1p-1022,
      sub * 3.0,
      sub,
      0x1.0000000000001p0,
      0x1.fffffffffffffp1022,
      std::nextafter(0x1.fffffffffffffp1022, inf),
      maxv,
      1e-200,
      1e200,
      1.7e308,
  };

  check_frozen(15.0, 10.0, 5.0);
  check_frozen(-15.0, 10.0, -5.0);
  check_frozen(6.0, 10.0, -4.0);
  check_frozen(0.0, 10.0, 0.0);
  check_frozen(-0.0, 10.0, -0.0);
  check_frozen(20.0, 10.0, 0.0);
  check_frozen(-20.0, 10.0, -0.0);
  check_frozen(0.45, 0.3, -0.14999999999999997);
  check_frozen(1e16, 3.0, 1.0);

  for (double length : lengths) {
    if (!(length > 0.0) || !std::isfinite(length)) {
      check(1.0, length, true);
      check(0.0, length, true);
      check(-0.0, length, true);
      continue;
    }
    const double half = 0.5 * length;
    check(0.0, length, true);
    check(-0.0, length, true);
    check(sub, length, true);
    check(-sub, length, true);
    check(length, length, true);
    check(-length, length, true);
    if (std::isfinite(length + length)) {
      check(length + length, length, true);
      check(-(length + length), length, true);
      check(std::nextafter(length + length, inf), length, true);
      check(std::nextafter(-(length + length), -inf), length, true);
    }
    around(half, length);
    around(-half, length);
    for (int k = 0; k <= 64; ++k) {
      check(half * static_cast<double>(k), length, true);
      check(half * static_cast<double>(-k), length, true);
    }
    for (int e = 0; e <= 60; ++e) {
      const double k = std::ldexp(1.0, e);
      check(half * k, length, true);
      check(half * -k, length, true);
      if (e >= 1 && std::isfinite(k - 1.0)) {
        check(half * (k - 1.0), length, true);
        check(half * -(k - 1.0), length, true);
        check(half * (k + 1.0), length, true);
        check(half * -(k + 1.0), length, true);
      }
    }
  }

  check(qnan, 10.0, true);
  check(inf, 10.0, true);
  check(-inf, 10.0, true);
  check(6.0, 0.0, true);
  check(6.0, -0.0, true);
  check(6.0, -10.0, true);
  check(6.0, -sub, true);
  check(6.0, qnan, true);
  check(6.0, inf, true);
  check(6.0, -inf, true);
  check(qnan, -1.0, true);
  check(inf, inf, true);
  check(-inf, 0.0, true);

  const std::uint64_t payloads[] = {0x1ULL,
                                    0x7ffffULL,
                                    0x80000ULL,
                                    0x123456789ULL,
                                    0x000fffffffffffffULL};
  for (std::uint64_t payload : payloads) {
    const std::uint64_t quiet = 0x7ff8000000000000ULL | (payload & 0x7ffffULL);
    const std::uint64_t signal =
        0x7ff0000000000000ULL | (payload & 0x7ffffffffffffULL);
    if ((signal & 0x000fffffffffffffULL) == 0) {
      continue;
    }
    check(from_bits(quiet), 10.0, true);
    check(from_bits(quiet | 0x8000000000000000ULL), 10.0, true);
    check(1.0, from_bits(quiet), true);
    check(from_bits(signal), 10.0, true);
    check(from_bits(signal | 0x8000000000000000ULL), 3.0, true);
    check(6.0, from_bits(signal), true);
  }
}

void random_suite() {
  constexpr long long nrandom = 10000000LL;
  const double inf = std::numeric_limits<double>::infinity();
  for (long long i = 0; i < nrandom; ++i) {
    const int kind = static_cast<int>(i % 5);
    double length = 1.0;
    double delta = 0.0;
    if (kind == 0) {
      int exp = static_cast<int>(next_rng() % 2200) - 1100;
      int eb = exp + 1023;
      std::uint64_t mant = next_rng() & ((1ULL << 52) - 1);
      if (eb <= 0 || eb >= 2047) {
        length = from_bits(next_rng() & 0x000fffffffffffffULL);
      } else {
        length = from_bits((static_cast<std::uint64_t>(eb) << 52) | mant);
      }
      delta = from_bits(next_rng());
    } else if (kind == 1) {
      int exp = static_cast<int>(next_rng() % 2000) - 1000;
      int eb = exp + 1023;
      if (eb <= 0 || eb >= 2047) {
        eb = 1023;
      }
      length = from_bits((static_cast<std::uint64_t>(eb) << 52) |
                         (next_rng() & ((1ULL << 52) - 1)));
      double frac = static_cast<double>(next_rng() >> 11) /
                    static_cast<double>(1ULL << 53);
      delta = (frac * 8.0 - 4.0) * length;
    } else if (kind == 2) {
      length = from_bits((1023ULL << 52) | (next_rng() & ((1ULL << 52) - 1)));
      length = std::ldexp(length, static_cast<int>(next_rng() % 400) - 200);
      int k = static_cast<int>(next_rng() % 17) - 8;
      delta = (static_cast<double>(k) + 0.5) * length;
      int steps = static_cast<int>(next_rng() % 5);
      double dir = (next_rng() & 1ULL) ? inf : -inf;
      for (int s = 0; s < steps; ++s) {
        delta = std::nextafter(delta, dir);
      }
    } else if (kind == 3) {
      length = from_bits((static_cast<std::uint64_t>(900 + 1023) << 52) |
                         (next_rng() & ((1ULL << 52) - 1)));
      delta = from_bits(next_rng() & 0x000fffffffffffffULL);
      if (next_rng() & 1ULL) {
        delta = -delta;
      }
      if ((i & 1LL) == 0) {
        double tmp = delta;
        delta = length;
        length = std::fabs(tmp) + 0.0;
      }
    } else {
      if ((next_rng() & 3ULL) == 0) {
        length = (next_rng() & 1ULL) ? 0.0 : -std::fabs(from_bits(next_rng()));
      } else {
        length = std::fabs(from_bits(next_rng()));
      }
      const std::uint64_t special = next_rng() % 5ULL;
      if (special == 0) {
        delta = inf;
      } else if (special == 1) {
        delta = -inf;
      } else if (special == 2) {
        delta = from_bits(0x7ff8000000000001ULL | (next_rng() & 0xffffULL));
      } else {
        delta = from_bits(next_rng());
      }
    }
    check(delta, length, false);
  }
}

} // namespace

int main() {
  edge_suite();
  random_suite();
  std::cout << "diff_edges=" << edges << " diff_random=" << randoms
            << " diff_fail=" << failures << "\n";
  return failures == 0 ? 0 : 1;
}
