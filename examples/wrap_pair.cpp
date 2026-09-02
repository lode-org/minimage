#include "minimage.hpp"

#include <cmath>
#include <exception>
#include <iostream>

int main() {
  try {
    const minimage::Cell ortho = minimage::Cell::ortho(10.0, 10.0, 10.0);
    const double d2 =
        ortho.dist2({0.2, 0.0, 0.0}, {9.4, 0.0, 0.0});
    if (std::abs(d2 - 0.64) > 1e-12) {
      std::cerr << "ortho dist2 " << d2 << "\n";
      return 1;
    }
    const minimage::Cell sheared = minimage::Cell::from_lammps_bounds(
        15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const double s2 = sheared.dist2({0.2, 0.1, 1.0}, {9.7, 0.1, 1.0});
    if (std::abs(s2 - 0.25) > 1e-12) {
      std::cerr << "sheared dist2 " << s2 << "\n";
      return 1;
    }
    const int pairs[] = {0, 1, 0, 0, 0, 1, 2, 3};
    const auto kept = minimage::reduce_pairs(pairs, 4);
    if (kept.size() != 2) {
      std::cerr << "reduce_pairs size " << kept.size() << "\n";
      return 1;
    }
    std::cout << "ortho " << d2 << "\n";
    std::cout << "sheared " << s2 << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
