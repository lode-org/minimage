#include "minimage.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

std::uint64_t bits(double x) {
  std::uint64_t out = 0;
  std::memcpy(&out, &x, sizeof out);
  return out;
}

void same(double delta, double length, double expected) {
  const double got = minimage::wrap_ortho_half_strict(delta, length);
  if (bits(got) != bits(expected)) {
    std::cerr.precision(17);
    std::cerr << "FAILED: wrap(" << delta << ", " << length << ") = " << got
              << ", expected " << expected << "\n";
    ++failures;
  }
}

void in_range(double delta, double length) {
  const double got = minimage::wrap_ortho_half_strict(delta, length);
  if (!std::isfinite(got) || std::fabs(got) > 0.5 * length) {
    std::cerr.precision(17);
    std::cerr << "FAILED: wrap(" << delta << ", " << length << ") = " << got
              << " is outside the interval\n";
    ++failures;
  }
}

void nan(double delta, double length) {
  const double got = minimage::wrap_ortho_half_strict(delta, length);
  if (!std::isnan(got)) {
    std::cerr.precision(17);
    std::cerr << "FAILED: wrap(" << delta << ", " << length << ") = " << got
              << ", expected NaN\n";
    ++failures;
  }
}

} // namespace

int main() {
  constexpr double sub = std::numeric_limits<double>::denorm_min();
  constexpr double max = std::numeric_limits<double>::max();
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double qnan = std::numeric_limits<double>::quiet_NaN();

  static_assert(noexcept(minimage::wrap_ortho_half_strict(1.0, 1.0)),
                "wrap_ortho_half_strict is noexcept");

  // Exact binary ties keep the sign of delta.
  same(5.0, 10.0, 5.0);
  same(15.0, 10.0, 5.0);
  same(1.5, 3.0, 1.5);
  same(4.5, 3.0, 1.5);
  same(0.5, 1.0, 0.5);
  same(-5.0, 10.0, -5.0);
  same(-15.0, 10.0, -5.0);
  same(-1.5, 3.0, -1.5);
  same(-4.5, 3.0, -1.5);
  same(-0.5, 1.0, -0.5);

  same(4.0, 10.0, 4.0);
  same(6.0, 10.0, -4.0);
  same(-4.0, 10.0, -4.0);
  same(-6.0, 10.0, 4.0);

  same(0.0, 10.0, 0.0);
  same(-0.0, 10.0, -0.0);
  same(20.0, 10.0, 0.0);
  same(-20.0, 10.0, -0.0);

  // One ulp either side of the boundary.
  same(1.4999999999999998, 3.0, 1.4999999999999998);
  same(-1.4999999999999998, 3.0, -1.4999999999999998);
  same(0.49999999999999994, 1.0, 0.49999999999999994);
  same(-0.49999999999999994, 1.0, -0.49999999999999994);
  same(0.5000000000000001, 1.0, -0.4999999999999999);
  same(-0.5000000000000001, 1.0, 0.4999999999999999);

  // Decimal ties are not binary ties and stay inside the interval.
  same(0.45, 0.3, -0.14999999999999997);
  same(-0.75, 0.3, 0.14999999999999997);
  same(-63.45000000000001, 2.7, 1.3499999999999943);
  in_range(0.45, 0.3);
  in_range(-0.75, 0.3);
  in_range(-63.45000000000001, 2.7);

  // Large multiples reduce exactly.
  same(1e16, 3.0, 1.0);
  same(1e17, 0.3, -0.09925658458281139);
  same(1e308, 0.5, 0.0);
  same(-1e200, 1e-200, -4.18199169208316e-201);
  same(max, max, 0.0);
  same(1e308, 1.7e308, -6.999999999999999e307);
  same(-max, 1e308, 2.0230686513768431e307);
  in_range(1e16, 3.0);
  in_range(1e17, 0.3);
  in_range(1e308, 0.5);
  in_range(-1e200, 1e-200);
  in_range(max, 3.0);
  in_range(-max, 0.7);

  // Subnormal lengths.
  same(sub, 3.0 * sub, sub);
  same(2.0 * sub, 3.0 * sub, -sub);
  same(-2.0 * sub, 3.0 * sub, sub);
  same(1.0, 3.0 * sub, sub);
  in_range(1.0, 3.0 * sub);
  in_range(-1e300, 7.0 * sub);

  // Non-finite delta and invalid lengths give NaN.
  nan(qnan, 10.0);
  nan(inf, 10.0);
  nan(-inf, 10.0);
  nan(6.0, 0.0);
  nan(6.0, -0.0);
  nan(6.0, -10.0);
  nan(6.0, -sub);
  nan(6.0, qnan);
  nan(6.0, inf);
  nan(6.0, -inf);
  nan(qnan, -1.0);

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  return 0;
}
