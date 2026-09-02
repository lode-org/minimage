#ifndef MINIMAGE_HPP
#define MINIMAGE_HPP

#if defined(__cplusplus) && __cplusplus < 201703L
#error "minimage.hpp requires C++17 or later"
#endif

extern "C" {
#include "minimage.h"
}

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minimage {

/// Failure from a cell or a distance call. `status()` is the C ABI
/// return (0 success, nonzero failure).
class Error : public std::runtime_error {
public:
  Error(int status, std::string message)
      : std::runtime_error(message.empty() ? "minimage: call failed"
                                           : std::move(message)),
        status_(status) {}

  explicit Error(std::string message) : Error(1, std::move(message)) {}

  [[nodiscard]] int status() const noexcept { return status_; }

private:
  int status_;
};

inline void check(int status, const char *what) {
  if (status != 0) {
    const char *msg = mi_last_error();
    throw Error(status, msg ? std::string(msg) : std::string(what));
  }
}

/// Periodic parallelepiped. Fields match `mi_cell` (lattice vectors
/// a, b, c and dump-cell origin).
struct Cell {
  std::array<double, 3> a{};
  std::array<double, 3> b{};
  std::array<double, 3> c{};
  std::array<double, 3> origin{};

  Cell() = default;

  explicit Cell(const mi_cell &raw) noexcept
      : a{raw.ax, raw.ay, raw.az}, b{raw.bx, raw.by, raw.bz},
        c{raw.cx, raw.cy, raw.cz}, origin{raw.ox, raw.oy, raw.oz} {}

  static Cell ortho(double lx, double ly, double lz) noexcept {
    return Cell(mi_cell_ortho(lx, ly, lz));
  }

  static Cell from_vectors(std::array<double, 3> a, std::array<double, 3> b,
                           std::array<double, 3> c,
                           std::array<double, 3> origin = {}) {
    mi_cell raw;
    check(mi_cell_from_vectors(a.data(), b.data(), c.data(), origin.data(),
                               &raw),
          "minimage: from_vectors failed");
    return Cell(raw);
  }

  static Cell from_lammps(double xlo, double xhi, double ylo, double yhi,
                          double zlo, double zhi, double xy, double xz,
                          double yz) {
    mi_cell raw;
    check(mi_cell_from_lammps(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz, &raw),
          "minimage: from_lammps failed");
    return Cell(raw);
  }

  static Cell from_lammps_bounds(double xspan, double yspan, double zspan,
                                 double xy, double xz, double yz, double xlo_b,
                                 double ylo_b, double zlo_b) {
    mi_cell raw;
    check(mi_cell_from_lammps_bounds(xspan, yspan, zspan, xy, xz, yz, xlo_b,
                                     ylo_b, zlo_b, &raw),
          "minimage: from_lammps_bounds failed");
    return Cell(raw);
  }

  static Cell from_ase(const double rows[9],
                       const double *origin = nullptr) {
    mi_cell raw;
    check(mi_cell_from_ase(rows, origin, &raw), "minimage: from_ase failed");
    return Cell(raw);
  }

  static Cell from_con(const double rows[9]) {
    mi_cell raw;
    check(mi_cell_from_con(rows, &raw), "minimage: from_con failed");
    return Cell(raw);
  }

  static Cell from_con_box(std::array<double, 3> boxl,
                           std::array<double, 3> angles_deg) {
    mi_cell raw;
    check(mi_cell_from_con_box(boxl.data(), angles_deg.data(), &raw),
          "minimage: from_con_box failed");
    return Cell(raw);
  }

  static Cell from_vesin(const double rows[9]) {
    mi_cell raw;
    check(mi_cell_from_vesin(rows, &raw), "minimage: from_vesin failed");
    return Cell(raw);
  }

  [[nodiscard]] mi_cell raw() const noexcept {
    return mi_cell{a[0],      a[1],      a[2],      b[0],      b[1],
                   b[2],      c[0],      c[1],      c[2],      origin[0],
                   origin[1], origin[2]};
  }

  [[nodiscard]] explicit operator mi_cell() const noexcept { return raw(); }

  [[nodiscard]] std::array<double, 3>
  displacement(std::array<double, 3> p, std::array<double, 3> q) const {
    const mi_cell box = raw();
    std::array<double, 3> dr{};
    check(mi_displacement(&box, p.data(), q.data(), dr.data()),
          "minimage: displacement failed");
    return dr;
  }

  [[nodiscard]] double dist2(std::array<double, 3> p,
                             std::array<double, 3> q) const {
    const mi_cell box = raw();
    double out = 0.0;
    check(mi_dist2(&box, p.data(), q.data(), &out), "minimage: dist2 failed");
    return out;
  }

  void dist2_many(std::array<double, 3> p, const double *qs, std::size_t n,
                  double *out) const {
    const mi_cell box = raw();
    check(mi_dist2_many(&box, p.data(), qs, n, out),
          "minimage: dist2_many failed");
  }
};

inline void dist2_ortho_diffs(const double *dx, const double *dy,
                              const double *dz, double bx, double by, double bz,
                              double *out, std::size_t n) {
  check(mi_dist2_ortho_diffs(dx, dy, dz, bx, by, bz, out, n),
        "minimage: dist2_ortho_diffs failed");
}

/// Drop self images and collapse duplicate `(i, j)` rows.
inline std::vector<std::array<int, 2>>
reduce_pairs(const int *pairs, std::size_t n) {
  if (n == 0) {
    return {};
  }
  std::vector<int> out(n * 2, 0);
  std::size_t kept = 0;
  check(mi_reduce_pairs(pairs, n, out.data(), &kept),
        "minimage: reduce_pairs failed");
  std::vector<std::array<int, 2>> rows(kept);
  for (std::size_t k = 0; k < kept; ++k) {
    rows[k] = {out[2 * k], out[2 * k + 1]};
  }
  return rows;
}

} // namespace minimage

#endif
