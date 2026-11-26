// Microbench of wrap_ortho_half_strict.
// Three bodies, one input stream: |x| < 2L and half the samples are
// exact half-integer ties (+-L/2 and +-3L/2).

#ifndef BASE_SHA
#define BASE_SHA "unknown"
#endif
#ifndef CC_ID
#define CC_ID __VERSION__
#endif
#ifndef BENCH_FLAGS
#define BENCH_FLAGS "unspecified"
#endif

#include "minimage.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr double k_length = 12.0;

inline double fold_remainder(double rem, double length) {
  const double half = 0.5 * length;
  if (rem > half) {
    return rem - length;
  }
  if (rem < -half) {
    return rem + length;
  }
  return rem;
}

// fmod body at 2c58f45af19adfb0b69e045412fab9ff9b1fa140.
extern "C" [[gnu::noinline]] double sum_base(const double *x, std::size_t n,
                                             double length) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double delta = x[i];
    double wrapped;
    if (!std::isfinite(delta) || !std::isfinite(length) || length <= 0.0) {
      wrapped = std::numeric_limits<double>::quiet_NaN();
    } else {
      const double rem = std::fmod(delta, length);
      constexpr double small_length = 0x1p-1000;
      constexpr double small_scale = 0x1p64;
      if (length < small_length) {
        wrapped = fold_remainder(rem * small_scale, length * small_scale) /
                  small_scale;
      } else {
        wrapped = fold_remainder(rem, length);
      }
    }
    sum += wrapped;
  }
  return sum;
}

// Floor body at 2e52b9e067ccccb29d767a0d33df4a32d03cb11a.
extern "C" [[gnu::noinline]] double sum_old(const double *x, std::size_t n,
                                            double length) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double delta = x[i];
    const double half = 0.5 * length;
    double wrapped = delta - length * std::floor(delta / length + 0.5);
    if (wrapped == -half && delta > 0.0) {
      wrapped = half;
    }
    sum += wrapped;
  }
  return sum;
}

extern "C" [[gnu::noinline]] double sum_new(const double *x, std::size_t n,
                                            double length) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += minimage::wrap_ortho_half_strict(x[i], length);
  }
  return sum;
}

std::uint64_t bits_of(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::vector<double> make_inputs(std::size_t n, double length) {
  std::vector<double> x(n);
  const double half = 0.5 * length;
  const double ties[4] = {half, -half, 3.0 * half, -3.0 * half};
  const double span = std::nextafter(length + length, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if ((i & 1u) == 0u) {
      x[i] = ties[(i >> 1) & 3u];
    } else {
      std::uint64_t z = (i + 1) * 0x9e3779b97f4a7c15ull;
      z ^= z >> 30;
      z *= 0xbf58476d1ce4e5b9ull;
      z ^= z >> 27;
      const double unit =
          static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
      x[i] = (unit * 2.0 - 1.0) * span;
    }
  }
  return x;
}

double median(double *samples, int count) {
  std::sort(samples, samples + count);
  return 0.5 * (samples[count / 2 - 1] + samples[count / 2]);
}

double time_call(double (*fn)(const double *, std::size_t, double),
                 const double *x, std::size_t n, double length) {
  const auto t0 = std::chrono::steady_clock::now();
  const double sum = fn(x, n, length);
  const auto t1 = std::chrono::steady_clock::now();
  asm volatile("" ::"x"(sum));
  return std::chrono::duration<double, std::nano>(t1 - t0).count() /
         static_cast<double>(n);
}

} // namespace

int main(int argc, char **argv) {
  const bool grind = argc >= 2 && std::strcmp(argv[1], "grind") == 0;
  const std::size_t n = grind ? (std::size_t{1} << 16) : (std::size_t{1} << 20);
  const auto inputs = make_inputs(n, k_length);
  const double *x = inputs.data();

  if (grind) {
    const char *which = argc >= 3 ? argv[2] : "new";
    double (*fn)(const double *, std::size_t, double) = sum_new;
    if (std::strcmp(which, "base") == 0) {
      fn = sum_base;
    } else if (std::strcmp(which, "old") == 0) {
      fn = sum_old;
    } else {
      which = "new";
    }
    const double sum = fn(x, n, k_length);
    std::printf("grind %s N %zu sum %.17g\n", which, n, sum);
    return 0;
  }

  std::uint64_t mismatch = 0;
  std::uint64_t ties = 0;
  const double half = 0.5 * k_length;
  for (std::size_t i = 0; i < n; ++i) {
    const double got = minimage::wrap_ortho_half_strict(x[i], k_length);
    const double ref_sum = sum_base(x + i, 1, k_length);
    if (bits_of(got) != bits_of(ref_sum)) {
      mismatch += 1;
    }
    const double mag = std::fabs(x[i]);
    if (mag == half || mag == 3.0 * half) {
      ties += 1;
    }
  }

  double base_s[24];
  double old_s[24];
  double new_s[24];
  int nb = 0;
  int no = 0;
  int nn = 0;
  for (int warm = 0; warm < 2; ++warm) {
    (void)time_call(sum_base, x, n, k_length);
    (void)time_call(sum_old, x, n, k_length);
    (void)time_call(sum_new, x, n, k_length);
    (void)time_call(sum_new, x, n, k_length);
    (void)time_call(sum_old, x, n, k_length);
    (void)time_call(sum_base, x, n, k_length);
  }
  for (int round = 0; round < 12; ++round) {
    base_s[nb++] = time_call(sum_base, x, n, k_length);
    old_s[no++] = time_call(sum_old, x, n, k_length);
    new_s[nn++] = time_call(sum_new, x, n, k_length);
    new_s[nn++] = time_call(sum_new, x, n, k_length);
    old_s[no++] = time_call(sum_old, x, n, k_length);
    base_s[nb++] = time_call(sum_base, x, n, k_length);
  }
  const double base_ns = median(base_s, nb);
  const double old_ns = median(old_s, no);
  const double new_ns = median(new_s, nn);
  std::printf("BASE %s\n", BASE_SHA);
  std::printf("CC %s\n", CC_ID);
  std::printf("FLAGS %s\n", BENCH_FLAGS);
  std::printf("N %zu\n", n);
  std::printf("rounds 12\n");
  std::printf("samples %d\n", nb);
  std::printf("ties %llu\n", static_cast<unsigned long long>(ties));
  std::printf("mismatch %llu\n", static_cast<unsigned long long>(mismatch));
  std::printf("base_ns %.3f\n", base_ns);
  std::printf("old_ns %.3f\n", old_ns);
  std::printf("new_ns %.3f\n", new_ns);
  std::printf("ratio %.3f\n", new_ns / old_ns);
  return mismatch == 0 ? 0 : 1;
}
