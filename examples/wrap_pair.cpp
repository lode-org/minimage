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
    struct Collector {
      std::vector<mi_image_pair> rows;
      Collector() = default;
      Collector(const Collector &) = delete;
      Collector &operator=(const Collector &) = delete;
      void operator()(const mi_image_pair &pair) { rows.push_back(pair); }
    };
    const minimage::Cell narrow = minimage::Cell::ortho(4.0, 4.0, 4.0);
    const double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    Collector col;
    const std::size_t visited =
        minimage::visit_pairs_within_images(narrow, xyz, 2, 4.5, col);
    if (visited == 0 || col.rows.size() != visited) {
      std::cerr << "collector size " << col.rows.size() << " vs " << visited
                << "\n";
      return 1;
    }
    std::size_t lambda_seen = 0;
    auto count_lambda = [&lambda_seen,
                         local = std::size_t{0}](const mi_image_pair &) mutable {
      ++local;
      lambda_seen = local;
    };
    minimage::visit_pairs_within_images(narrow, xyz, 2, 4.5, count_lambda);
    minimage::visit_pairs_within_images(narrow, xyz, 2, 4.5, count_lambda);
    if (lambda_seen != 2 * visited) {
      std::cerr << "mutable lambda seen " << lambda_seen << " vs "
                << 2 * visited << "\n";
      return 1;
    }
    const auto all_pairs = minimage::pairs_within_images(narrow, xyz, 2, 4.5);
    if (all_pairs.size() != visited) {
      std::cerr << "pairs_within_images size " << all_pairs.size() << " vs "
                << visited << "\n";
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
