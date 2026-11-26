#ifndef MINIMAGE_HPP
#define MINIMAGE_HPP

#if defined(__cplusplus) && __cplusplus < 201703L
#error "minimage.hpp requires C++17 or later"
#endif

extern "C" {
#include "minimage.h"
}

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace minimage {

/// Failure from a cell or a distance call.
///
/// `status()` holds the nonzero C ABI return code, and `what()` holds the
/// thread-local `minimage_last_error()` string when available.
class Error : public std::runtime_error {
public:
  /// Construct an error with a C ABI status code and message.
  Error(int status, std::string message)
      : std::runtime_error(message.empty() ? "minimage: call failed"
                                           : std::move(message)),
        status_(status) {}

  /// Construct an error with default status code `1`.
  explicit Error(std::string message) : Error(1, std::move(message)) {}

  /// Nonzero C ABI return code from the failing `minimage_*` call.
  [[nodiscard]] int status() const noexcept { return status_; }

private:
  int status_;
};

/// Throw `minimage::Error` when a C ABI call returns a nonzero status.
///
/// Reads `minimage_last_error()` on the current thread and falls back to
/// `what` if the error slot is null.
inline void check(int status, const char *what) {
  if (status != 0) {
    const char *msg = minimage_last_error();
    throw Error(status, msg ? std::string(msg) : std::string(what));
  }
}

/// Periodic parallelepiped.
///
/// Fields match `mi_cell`: lattice vectors `a`, `b`, `c` (columns of
/// `H`) and dump-cell `origin`, all in consistent Cartesian length
/// units. Origin defaults to `{0.0, 0.0, 0.0}` on constructors that omit
/// it.
struct Cell {
  /// Lattice vector `a` (first column of `H`).
  std::array<double, 3> a{};
  /// Lattice vector `b` (second column of `H`).
  std::array<double, 3> b{};
  /// Lattice vector `c` (third column of `H`).
  std::array<double, 3> c{};
  /// Dump-cell origin.
  std::array<double, 3> origin{};

  /// Default-construct an uninitialized zero cell (not valid until populated).
  Cell() = default;

  /// Wrap an existing C `mi_cell` without validating it.
  explicit Cell(const mi_cell &raw) noexcept
      : a{raw.ax, raw.ay, raw.az}, b{raw.bx, raw.by, raw.bz},
        c{raw.cx, raw.cy, raw.cz}, origin{raw.ox, raw.oy, raw.oz} {}

  /// Orthorhombic box with side lengths `(lx, ly, lz)` and zero origin.
  ///
  /// Does not validate the lengths at construction (`noexcept`); an
  /// invalid box throws `minimage::Error` on the first distance or
  /// displacement call.
  static Cell ortho(double lx, double ly, double lz) noexcept {
    return Cell(mi_cell_ortho(lx, ly, lz));
  }

  /// Parallelepiped from lattice vectors `a`, `b`, `c` and optional `origin`
  /// (defaults to `{0.0, 0.0, 0.0}`).
  ///
  /// Vectors are columns of `H`. Throws `minimage::Error` if `H` is
  /// singular or non-finite.
  static Cell from_vectors(std::array<double, 3> a, std::array<double, 3> b,
                           std::array<double, 3> c,
                           std::array<double, 3> origin = {}) {
    mi_cell raw;
    check(minimage_cell_from_vectors(a.data(), b.data(), c.data(),
                                     origin.data(), &raw),
          "minimage: from_vectors failed");
    return Cell(raw);
  }

  /// LAMMPS restricted-triclinic box from dump-bound coordinates
  /// `xlo, xhi, ylo, yhi, zlo, zhi` and tilts `xy, xz, yz`.
  ///
  /// Expects tilt-padded dump bounds (`xlo_bound, xhi_bound, ...`).
  /// Throws `minimage::Error` if the recovered cell is singular or
  /// non-finite.
  static Cell from_lammps(double xlo, double xhi, double ylo, double yhi,
                          double zlo, double zhi, double xy, double xz,
                          double yz) {
    mi_cell raw;
    check(minimage_cell_from_lammps(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz,
                                    &raw),
          "minimage: from_lammps failed");
    return Cell(raw);
  }

  /// LAMMPS dump bound spans `(xspan, yspan, zspan)`, tilts `(xy, xz, yz)`,
  /// and bound origins `(xlo_b, ylo_b, zlo_b)`.
  ///
  /// Throws `minimage::Error` if the recovered cell is singular or
  /// non-finite.
  static Cell from_lammps_bounds(double xspan, double yspan, double zspan,
                                 double xy, double xz, double yz, double xlo_b,
                                 double ylo_b, double zlo_b) {
    mi_cell raw;
    check(minimage_cell_from_lammps_bounds(xspan, yspan, zspan, xy, xz, yz,
                                           xlo_b, ylo_b, zlo_b, &raw),
          "minimage: from_lammps_bounds failed");
    return Cell(raw);
  }

  /// Construct from an ASE-style row-major 3x3 array (`rows[0..9]`).
  ///
  /// `origin` defaults to `nullptr` (zero origin). Throws
  /// `minimage::Error` if `rows` is null or the cell is singular or
  /// non-finite.
  static Cell from_ase(const double rows[9],
                       const double *origin = nullptr) {
    mi_cell raw;
    check(minimage_cell_from_ase(rows, origin, &raw),
          "minimage: from_ase failed");
    return Cell(raw);
  }

  /// Construct from a CON row-major 3x3 lattice (`rows[0..9]`), zero origin.
  ///
  /// Throws `minimage::Error` if `rows` is null or the cell is singular or
  /// non-finite.
  static Cell from_con(const double rows[9]) {
    mi_cell raw;
    check(minimage_cell_from_con(rows, &raw), "minimage: from_con failed");
    return Cell(raw);
  }

  /// Construct from CON lengths `boxl` (length units) and crystallographic
  /// angles `angles_deg` `[alpha, beta, gamma]` (degrees), zero origin.
  ///
  /// Throws `minimage::Error` if lengths are non-positive or angles give a
  /// degenerate cell.
  static Cell from_con_box(std::array<double, 3> boxl,
                           std::array<double, 3> angles_deg) {
    mi_cell raw;
    check(minimage_cell_from_con_box(boxl.data(), angles_deg.data(), &raw),
          "minimage: from_con_box failed");
    return Cell(raw);
  }

  /// Construct from a vesin row-major 3x3 box (`rows[0..9]`), zero origin.
  ///
  /// Throws `minimage::Error` if `rows` is null or the cell is singular or
  /// non-finite.
  static Cell from_vesin(const double rows[9]) {
    mi_cell raw;
    check(minimage_cell_from_vesin(rows, &raw), "minimage: from_vesin failed");
    return Cell(raw);
  }

  /// Convert to the underlying C `mi_cell` struct.
  [[nodiscard]] mi_cell raw() const noexcept {
    return mi_cell{a[0],      a[1],      a[2],      b[0],      b[1],
                   b[2],      c[0],      c[1],      c[2],      origin[0],
                   origin[1], origin[2]};
  }

  /// Explicit conversion to `mi_cell`.
  [[nodiscard]] explicit operator mi_cell() const noexcept { return raw(); }

  /// Engine minimum-image displacement from `p` to `q`: `wrap(q - p)` in
  /// cell length units. Throws `minimage::Error` on an invalid cell.
  [[nodiscard]] std::array<double, 3>
  displacement(std::array<double, 3> p, std::array<double, 3> q) const {
    const mi_cell box = raw();
    std::array<double, 3> dr{};
    check(minimage_displacement(&box, p.data(), q.data(), dr.data()),
          "minimage: displacement failed");
    return dr;
  }

  /// Euclidean nearest-image displacement from `p` to `q` in cell length
  /// units (Smith test, else the Selling superbasis slicer; ties keep the
  /// engine vector). Throws `minimage::Error` on an invalid cell.
  [[nodiscard]] std::array<double, 3>
  displacement_euclidean(std::array<double, 3> p,
                         std::array<double, 3> q) const {
    const mi_cell box = raw();
    std::array<double, 3> dr{};
    check(minimage_displacement_euclidean(&box, p.data(), q.data(), dr.data()),
          "minimage: displacement_euclidean failed");
    return dr;
  }

  /// Squared engine minimum-image distance between `p` and `q` in squared
  /// cell length units. Throws `minimage::Error` on an invalid cell.
  [[nodiscard]] double dist2(std::array<double, 3> p,
                             std::array<double, 3> q) const {
    const mi_cell box = raw();
    double out = 0.0;
    check(minimage_dist2(&box, p.data(), q.data(), &out),
          "minimage: dist2 failed");
    return out;
  }

  /// Rapaport's bin pair: `|q + shift - p|^2` for `n` packed candidates in
  /// `qs` (`n * 3` doubles) into `out` (`n` doubles), in squared length
  /// units. Throws `minimage::Error` on null pointers when `n > 0`.
  void dist2_shifted_many(std::array<double, 3> p, const double *qs,
                          std::array<double, 3> shift, std::size_t n,
                          double *out) const {
    check(minimage_dist2_shifted_many(p.data(), qs, shift.data(), n, out),
          "minimage: dist2_shifted_many failed");
  }

  /// Squared engine MIC distances from `p` to `n` packed candidates in
  /// `qs` (`n * 3` doubles) into `out` (`n` doubles). Throws
  /// `minimage::Error` on invalid cell or null pointers when `n > 0`.
  void dist2_many(std::array<double, 3> p, const double *qs, std::size_t n,
                  double *out) const {
    const mi_cell box = raw();
    check(minimage_dist2_many(&box, p.data(), qs, n, out),
          "minimage: dist2_many failed");
  }

  /// Euclidean nearest-image squared distances from `p` to `n` packed
  /// candidates in `qs` (`n * 3` doubles) into `out` (`n` doubles), with
  /// one superbasis lookup per call. Throws `minimage::Error` on invalid
  /// cell or null pointers when `n > 0`.
  void dist2_euclidean_many(std::array<double, 3> p, const double *qs,
                            std::size_t n, double *out) const {
    const mi_cell box = raw();
    check(minimage_dist2_euclidean_many(&box, p.data(), qs, n, out),
          "minimage: dist2_euclidean_many failed");
  }

  /// Euclidean nearest-image squared distances for `n` packed pairs
  /// `(ps[k], qs[k])` (`n * 3` doubles each) into `out` (`n` doubles).
  /// Throws `minimage::Error` on invalid cell or null pointers when `n > 0`.
  void dist2_euclidean_pairs(const double *ps, const double *qs,
                             std::size_t n, double *out) const {
    const mi_cell box = raw();
    check(minimage_dist2_euclidean_pairs(&box, ps, qs, n, out),
          "minimage: dist2_euclidean_pairs failed");
  }

  /// Pairs that persist across frames: `images` (`n * 3` `int32_t`, zeros
  /// to start) holds each pair's lattice shift from the previous call and
  /// is updated in place; `out` receives `n` squared distances. Throws
  /// `minimage::Error` on invalid cell or null pointers when `n > 0`.
  void dist2_euclidean_pairs_warm(const double *ps, const double *qs,
                                  std::size_t n, std::int32_t *images,
                                  double *out) const {
    const mi_cell box = raw();
    check(minimage_dist2_euclidean_pairs_warm(&box, ps, qs, n, images, out),
          "minimage: dist2_euclidean_pairs_warm failed");
  }
};

namespace detail {
/// Fold `|rem| < length` into `[-length / 2, length / 2]`. Each branch
/// subtracts values within a factor of two of each other, so it is exact
/// by Sterbenz's lemma.
inline double fold_remainder(double rem, double length) noexcept {
  const double half = 0.5 * length;
  if (rem > half) {
    return rem - length;
  }
  if (rem < -half) {
    return rem + length;
  }
  return rem;
}

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
inline double wrap_ortho_half_strict_slow(double delta,
                                           double length) noexcept {
  if (!std::isfinite(delta) || !std::isfinite(length) || length <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double rem = std::fmod(delta, length);
  // 0.5 * length is inexact for a subnormal length with an odd last bit.
  constexpr double small_length = 0x1p-1000;
  constexpr double small_scale = 0x1p64;
  if (length < small_length) {
    return fold_remainder(rem * small_scale, length * small_scale) /
           small_scale;
  }
  return fold_remainder(rem, length);
}
} // namespace detail

/// Wrap one orthorhombic component into the closed interval
/// `[-length / 2, length / 2]`.
///
/// The result is the nearest periodic image `delta - n * length` for an
/// integer `n`, with no rounding error. A finite `delta` within two
/// lengths, and a length from `2^-1000` through half the largest finite
/// value, is reduced by exact additions and subtractions. Every other
/// input uses `std::fmod`, which is exact, then at most one exact fold.
/// A length below `2^-1000` is scaled by `2^64` before that fold so a
/// subnormal half stays exact. Values just inside `+-length / 2` are
/// kept, and multiples far larger than `length` still give a finite,
/// in-range result.
///
/// When the remainder equals `length / 2` exactly in binary, the result
/// keeps the sign of `delta` (`15.0` and `-15.0` with length `10.0` give
/// `5.0` and `-5.0`). A decimal tie such as `0.45` against `0.3` is not a
/// binary tie and returns its nearest image. The sign of a zero result
/// follows `delta`.
///
/// Returns NaN when `delta` is not finite, or `length` is not finite or not
/// strictly positive. Behaves identically to the Rust
/// `minimage::wrap_ortho_half_strict`.
[[nodiscard]] inline double wrap_ortho_half_strict(double delta,
                                                    double length) noexcept {
  // 2*length and length/2 are exact for every length in this window.
  constexpr double min_length = 0x1p-1000;
  constexpr double max_length = 0x1.fffffffffffffp1022;
  if (length >= min_length && length <= max_length) {
    const double span = length + length;
    if (std::fabs(delta) <= span) {
      const double down = static_cast<double>(delta > length) * length;
      const double up = static_cast<double>(delta < -length) * length;
      double wrapped = (delta - down) + up;
      const double half = 0.5 * length;
      const double fold_down = static_cast<double>(wrapped > half) * length;
      const double fold_up = static_cast<double>(wrapped < -half) * length;
      wrapped = (wrapped - fold_down) + fold_up;
      if (wrapped == 0.0) {
        wrapped = std::copysign(0.0, delta);
      }
      return wrapped;
    }
  }
  return detail::wrap_ortho_half_strict_slow(delta, length);
}

/// Orthorhombic wrap of `n` precomputed SoA differences `(dx, dy, dz)`
/// with box lengths `(bx, by, bz)` into `out` (`n` squared distances).
/// Throws `minimage::Error` if `n > 0` and any pointer is null or any box
/// length is not strictly positive.
inline void dist2_ortho_diffs(const double *dx, const double *dy,
                              const double *dz, double bx, double by, double bz,
                              double *out, std::size_t n) {
  check(minimage_dist2_ortho_diffs(dx, dy, dz, bx, by, bz, out, n),
        "minimage: dist2_ortho_diffs failed");
}

/// Drop self images (`i == j`) and collapse duplicate `(i, j)` rows from
/// `n` packed integer pairs (`2 * n` ints). Returns sorted unique pairs;
/// returns `{}` when `n == 0`. Throws `minimage::Error` if `n > 0` and
/// `pairs` is null.
inline std::vector<std::array<int, 2>>
reduce_pairs(const int *pairs, std::size_t n) {
  if (n == 0) {
    return {};
  }
  std::vector<int> out(n * 2, 0);
  std::size_t kept = 0;
  check(minimage_reduce_pairs(pairs, n, out.data(), &kept),
        "minimage: reduce_pairs failed");
  std::vector<std::array<int, 2>> rows(kept);
  for (std::size_t k = 0; k < kept; ++k) {
    rows[k] = {out[2 * k], out[2 * k + 1]};
  }
  return rows;
}

namespace detail {

/// Callable plus the first exception it threw, shared with the C trampoline.
template <typename F> struct VisitState {
  /// Caller's visitor.
  F *fn;
  /// First exception thrown by `fn`, rethrown after the C call returns.
  std::exception_ptr error;
};

} // namespace detail

/// Visit every periodic image of `n` points strictly inside `cutoff` in a
/// single pass without allocating a pair buffer.
///
/// Calls `visit(const mi_image_pair &)` for each emitted image and returns
/// the total number of emitted pair images. An exception thrown by `visit`
/// never crosses the C or Rust frames: the walk stops after that image and
/// the exception is rethrown from this function once the C call returns.
template <typename F>
inline std::size_t
visit_pairs_within_images(const Cell &cell, const double *xyz, std::size_t n,
                          double cutoff, F &&visit,
                          std::array<bool, 3> periodic = {true, true, true},
                          bool half = true) {
  if (n == 0 || cutoff == 0.0) {
    return 0;
  }
  using Fn = std::decay_t<F>;
  const mi_cell box = cell.raw();
  const int pbc[3] = {periodic[0] ? 1 : 0, periodic[1] ? 1 : 0,
                      periodic[2] ? 1 : 0};
  auto thunk = [](void *userdata, const mi_image_pair *pair) noexcept -> int {
    auto *state = static_cast<detail::VisitState<Fn> *>(userdata);
    try {
      (*state->fn)(*pair);
      return 0;
    } catch (...) {
      state->error = std::current_exception();
      return 1;
    }
  };
  Fn fn(std::forward<F>(visit));
  detail::VisitState<Fn> state{&fn, nullptr};
  std::size_t count = 0;
  const int status = minimage_visit_pairs_within_images(
      &box, pbc, xyz, n, cutoff, half ? 1 : 0, thunk, &state, &count);
  if (state.error) {
    std::rethrow_exception(state.error);
  }
  check(status, "minimage: visit_pairs_within_images failed");
  return count;
}

/// Enumerate every periodic image of `n` points strictly inside `cutoff`.
///
/// Each returned `mi_image_pair` holds `(i, j, shift, d, dist2)` with
/// `d = (xyz[i] - xyz[j]) - H shift` and `0 < dist2 < cutoff^2`.
/// When `half` is `true`, only `i <= j` are returned and self-images
/// (`i == j`) are restricted to the positive half-space.
inline std::vector<mi_image_pair>
pairs_within_images(const Cell &cell, const double *xyz, std::size_t n,
                    double cutoff,
                    std::array<bool, 3> periodic = {true, true, true},
                    bool half = true) {
  std::vector<mi_image_pair> out;
  visit_pairs_within_images(
      cell, xyz, n, cutoff,
      [&out](const mi_image_pair &pair) { out.push_back(pair); }, periodic,
      half);
  return out;
}

} // namespace minimage

#endif
