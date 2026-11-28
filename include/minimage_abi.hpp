#ifndef MINIMAGE_ABI_HPP
#define MINIMAGE_ABI_HPP

/// \file
/// C++ over the negotiated C ABI of `minimage.h`, for a host that loads
/// the library at run time.
///
/// Nothing here refers to a `minimage_*` symbol at link time: every call
/// goes through the `minimage_ops` table a `Library` holds. A host
/// compiles against the headers alone (the `minimage-headers` Meson
/// dependency), opens the shared library by path, and gets an exception
/// instead of a crash when the library is of another ABI or lacks an
/// operation.
///
/// \code
/// auto lib = minimage::abi::Library::open("libminimage.so");
/// auto box = lib.diagonal({10.0, 10.0, 10.0}, {true, true, false});
/// box.wrap_diffs(diffs.data(), n, diffs.data());  // in place
/// \endcode
///
/// A `Library` and the `Lattice` objects built from it share the loaded
/// library, which stays open until the last of them is gone. Both are
/// immutable after construction, so any number of threads may call the
/// const members of one object at once.

#if defined(__cplusplus) && __cplusplus < 201703L
#error "minimage_abi.hpp requires C++17 or later"
#endif

#include "minimage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace minimage {
namespace abi {

/// A nonzero status from the library, or a failed load.
///
/// `status()` is one of the `MINIMAGE_STATUS_*` codes: positive for a
/// failure the caller can recover from, negative for a misuse of the
/// interface or a fault inside the library.
class Error : public std::runtime_error {
public:
  /// An error with a status code and a message.
  Error(std::int32_t status, const std::string &message)
      : std::runtime_error(message), status_(status) {}

  /// The `MINIMAGE_STATUS_*` code.
  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

  /// True for a positive code, which the caller can recover from.
  [[nodiscard]] bool recoverable() const noexcept { return status_ > 0; }

private:
  std::int32_t status_;
};

/// The C ABI a header or a library was built for; see `minimage_abi`.
struct Stamp {
  /// `MINIMAGE_ABI_MAJOR`.
  std::uint32_t major;
  /// `MINIMAGE_ABI_MINOR`.
  std::uint32_t minor;
  /// `MINIMAGE_ABI_LAYOUT_REVISION`.
  std::uint32_t layout_revision;
};

/// The ABI of the header this translation unit was compiled against.
[[nodiscard]] constexpr Stamp header_stamp() noexcept {
  return {MINIMAGE_ABI_MAJOR, MINIMAGE_ABI_MINOR,
          MINIMAGE_ABI_LAYOUT_REVISION};
}

/// What a library can do; see `minimage_caps`. A member the library does
/// not report is false.
struct Capabilities {
  /// Lattice handles.
  bool lattice = false;
  /// A lattice may be open along any of its axes.
  bool open_axes = false;
  /// `Lattice::wrap_diffs`.
  bool wrap_diffs = false;
  /// `Lattice::wrap_pairs`.
  bool wrap_pairs = false;
  /// `Lattice::nearest_diffs`.
  bool nearest_diffs = false;
  /// `Lattice::nearest_pairs`.
  bool nearest_pairs = false;
  /// `Lattice::wrap_points`.
  bool wrap_points = false;
  /// The output of a batch call may be one of its inputs.
  bool in_place = false;
  /// The engine wrap of a sheared lattice with an open axis.
  bool wrap_open_sheared = false;
};

/// What a lattice stores; see `minimage_lattice_info`.
struct LatticeInfo {
  /// Every periodic vector lies along its own Cartesian axis.
  bool orthorhombic = false;
  /// Which of the axes a, b, c are periodic.
  std::array<bool, 3> periodic{};
  /// The stored lattice vectors a, b, c, row-major.
  std::array<double, 9> rows{};
  /// The origin of the cell.
  std::array<double, 3> origin{};
  /// The spacing of the lattice planes across each periodic vector.
  std::array<double, 3> widths{};
};

/// The three entry points a host resolves by name.
struct EntryPoints {
  /// `minimage_abi_stamp`.
  std::int32_t (*abi_stamp)(minimage_abi *) = nullptr;
  /// `minimage_abi_compatible`.
  std::int32_t (*abi_compatible)(const minimage_abi *) = nullptr;
  /// `minimage_ops_get`.
  std::int32_t (*ops_get)(minimage_ops *) = nullptr;
};

/// Packed rows of three doubles.
using Rows = std::vector<std::array<double, 3>>;

namespace detail {

static_assert(sizeof(std::array<double, 3>) == 3 * sizeof(double),
              "rows are packed doubles");

/// The negotiated table and, when this object opened it, the library.
struct State {
  minimage_ops ops{};
  Stamp stamp{};
  void *handle = nullptr;

  State() = default;
  State(const State &) = delete;
  State &operator=(const State &) = delete;
  ~State() {
#if !defined(_WIN32)
    if (handle != nullptr) {
      dlclose(handle);
    }
#endif
  }

  /// Throw `Error` for a nonzero status, with the library's message.
  void check(std::int32_t status, const char *what) const {
    if (status == MINIMAGE_STATUS_OK) {
      return;
    }
    const char *text = ops.last_error != nullptr ? ops.last_error() : nullptr;
    throw Error(status, std::string("minimage: ") + what + ": " +
                            (text != nullptr ? text : "call failed"));
  }
};

/// The operation behind a table member, or `Error` with
/// `MINIMAGE_STATUS_UNSUPPORTED` when the library does not have it.
template <typename Fn> Fn need(Fn slot, const char *name) {
  if (slot == nullptr) {
    throw Error(MINIMAGE_STATUS_UNSUPPORTED,
                std::string("minimage: this library has no ") + name);
  }
  return slot;
}

inline std::shared_ptr<const State> negotiate(const EntryPoints &entry,
                                              void *handle) {
  auto state = std::make_shared<State>();
  // The state owns the handle from here, so a throw below closes it.
  state->handle = handle;
  if (entry.abi_stamp == nullptr || entry.abi_compatible == nullptr ||
      entry.ops_get == nullptr) {
    throw Error(MINIMAGE_STATUS_UNSUPPORTED,
                "minimage: the library does not export the negotiated ABI");
  }
  minimage_abi library{};
  if (const std::int32_t status = entry.abi_stamp(&library); status != 0) {
    throw Error(status, "minimage: the library did not report its ABI");
  }
  state->stamp = {library.major, library.minor, library.layout_revision};
  const minimage_abi host{MINIMAGE_ABI_MAJOR, MINIMAGE_ABI_MINOR,
                          MINIMAGE_ABI_LAYOUT_REVISION};
  if (const std::int32_t status = entry.abi_compatible(&host); status != 0) {
    throw Error(status,
                "minimage: library ABI " + std::to_string(library.major) +
                    "." + std::to_string(library.minor) + " layout " +
                    std::to_string(library.layout_revision) +
                    " does not serve header ABI " +
                    std::to_string(host.major) + "." +
                    std::to_string(host.minor) + " layout " +
                    std::to_string(host.layout_revision));
  }
  state->ops.size = static_cast<std::uint32_t>(sizeof(minimage_ops));
  if (const std::int32_t status = entry.ops_get(&state->ops); status != 0) {
    throw Error(status, "minimage: the library did not fill its operations");
  }
  return state;
}

} // namespace detail

/// A lattice: three vectors, an origin, and which axes are periodic.
///
/// Move-only owner of a `minimage_lattice` handle. Each batch call takes
/// packed rows of three doubles; the pointer forms may write their
/// output over an input. A moved-from lattice holds no handle: `get()`
/// is null and every other member throws `Error` with
/// `MINIMAGE_STATUS_NULL_POINTER`.
class Lattice {
public:
  Lattice(const Lattice &) = delete;
  Lattice &operator=(const Lattice &) = delete;

  /// Take the handle of `other`, which is left empty.
  Lattice(Lattice &&other) noexcept
      : state_(std::move(other.state_)),
        raw_(std::exchange(other.raw_, nullptr)) {}

  /// Release this handle and take that of `other`.
  Lattice &operator=(Lattice &&other) noexcept {
    if (this != &other) {
      release();
      state_ = std::move(other.state_);
      raw_ = std::exchange(other.raw_, nullptr);
    }
    return *this;
  }

  ~Lattice() { release(); }

  /// The C handle, for a call through the table directly.
  [[nodiscard]] const minimage_lattice *get() const noexcept { return raw_; }

  /// What the lattice stores. An orthorhombic lattice stores positive
  /// edges, whatever sign it was built with.
  [[nodiscard]] LatticeInfo info() const {
    minimage_lattice_info raw{};
    raw.size = static_cast<std::uint32_t>(sizeof raw);
    const detail::State &state = live();
    state.check(detail::need(state.ops.lattice_get_info, "lattice_get_info")(
                    raw_, &raw),
                "lattice_get_info");
    LatticeInfo out;
    out.orthorhombic = raw.orthorhombic != 0;
    for (std::size_t k = 0; k < 3; ++k) {
      out.periodic[k] = raw.periodic[k] != 0;
      out.origin[k] = raw.origin[k];
      out.widths[k] = raw.widths[k];
    }
    for (std::size_t k = 0; k < 9; ++k) {
      out.rows[k] = raw.rows[k];
    }
    return out;
  }

  /// Engine wrap of `n` packed difference vectors; `out` may be `diffs`.
  /// Throws `Error` with `MINIMAGE_STATUS_UNSUPPORTED` for a sheared
  /// lattice with an open axis.
  void wrap_diffs(const double *diffs, std::size_t n, double *out) const {
    const detail::State &state = live();
    state.check(detail::need(state.ops.lattice_wrap_diffs, "lattice_wrap_diffs")(raw_, diffs, n, out),
                "lattice_wrap_diffs");
  }

  /// Engine wrap of `q[k] - p[k]`; `out` may be `p` or `q`.
  void wrap_pairs(const double *p, const double *q, std::size_t n,
                  double *out) const {
    const detail::State &state = live();
    state.check(detail::need(state.ops.lattice_wrap_pairs, "lattice_wrap_pairs")(raw_, p, q, n, out),
                "lattice_wrap_pairs");
  }

  /// Nearest image of `n` packed difference vectors, for every lattice;
  /// `out` may be `diffs`.
  void nearest_diffs(const double *diffs, std::size_t n, double *out) const {
    const detail::State &state = live();
    state.check(detail::need(state.ops.lattice_nearest_diffs, "lattice_nearest_diffs")(raw_, diffs, n, out),
                "lattice_nearest_diffs");
  }

  /// Nearest image of `q[k] - p[k]`; `out` may be `p` or `q`.
  void nearest_pairs(const double *p, const double *q, std::size_t n,
                     double *out) const {
    const detail::State &state = live();
    state.check(detail::need(state.ops.lattice_nearest_pairs, "lattice_nearest_pairs")(raw_, p, q, n, out),
                "lattice_nearest_pairs");
  }

  /// Wrap `n` packed points into the cell; `out` may be `points`.
  void wrap_points(const double *points, std::size_t n, double *out) const {
    const detail::State &state = live();
    state.check(detail::need(state.ops.lattice_wrap_points, "lattice_wrap_points")(raw_, points, n, out),
                "lattice_wrap_points");
  }

  /// `wrap_diffs` into a new vector.
  [[nodiscard]] Rows wrap_diffs(const Rows &diffs) const {
    Rows out(diffs.size());
    wrap_diffs(flat(diffs), diffs.size(), flat(out));
    return out;
  }

  /// `wrap_pairs` into a new vector. `p` and `q` have one length.
  [[nodiscard]] Rows wrap_pairs(const Rows &p, const Rows &q) const {
    same_length(p, q);
    Rows out(p.size());
    wrap_pairs(flat(p), flat(q), p.size(), flat(out));
    return out;
  }

  /// `nearest_diffs` into a new vector.
  [[nodiscard]] Rows nearest_diffs(const Rows &diffs) const {
    Rows out(diffs.size());
    nearest_diffs(flat(diffs), diffs.size(), flat(out));
    return out;
  }

  /// `nearest_pairs` into a new vector. `p` and `q` have one length.
  [[nodiscard]] Rows nearest_pairs(const Rows &p, const Rows &q) const {
    same_length(p, q);
    Rows out(p.size());
    nearest_pairs(flat(p), flat(q), p.size(), flat(out));
    return out;
  }

  /// `wrap_points` into a new vector.
  [[nodiscard]] Rows wrap_points(const Rows &points) const {
    Rows out(points.size());
    wrap_points(flat(points), points.size(), flat(out));
    return out;
  }

private:
  friend class Library;

  Lattice(std::shared_ptr<const detail::State> state, minimage_lattice *raw)
      : state_(std::move(state)), raw_(raw) {}

  static const double *flat(const Rows &rows) {
    return rows.empty() ? nullptr : rows.front().data();
  }

  static double *flat(Rows &rows) {
    return rows.empty() ? nullptr : rows.front().data();
  }

  /// The shared state of a lattice that holds a handle. A moved-from
  /// lattice holds none, and every call on it throws `Error` with
  /// `MINIMAGE_STATUS_NULL_POINTER` instead of passing a null handle on.
  const detail::State &live() const {
    if (raw_ == nullptr || state_ == nullptr) {
      throw Error(MINIMAGE_STATUS_NULL_POINTER,
                  "minimage: this lattice was moved from");
    }
    return *state_;
  }

  static void same_length(const Rows &p, const Rows &q) {
    if (p.size() != q.size()) {
      throw Error(MINIMAGE_STATUS_BAD_COUNT,
                  "minimage: p and q differ in length");
    }
  }

  void release() noexcept {
    if (raw_ != nullptr && state_ != nullptr &&
        state_->ops.lattice_free != nullptr) {
      state_->ops.lattice_free(raw_);
    }
    raw_ = nullptr;
  }

  std::shared_ptr<const detail::State> state_;
  minimage_lattice *raw_ = nullptr;
};

/// A minimage library whose ABI was checked against this header.
class Library {
public:
  /// Negotiate with a library through its three entry points, however
  /// the host resolved them. Throws `Error` with
  /// `MINIMAGE_STATUS_INCOMPATIBLE` when the library does not serve this
  /// header. The host keeps the library loaded while this object and the
  /// lattices built from it live.
  explicit Library(const EntryPoints &entry)
      : state_(detail::negotiate(entry, nullptr)) {}

#if !defined(_WIN32)
  /// Open the shared library at `path` with `dlopen` and negotiate. The
  /// library is closed when this object and every lattice built from it
  /// are gone. Throws `Error` when the file cannot be opened, is not a
  /// minimage with the negotiated ABI, or does not serve this header.
  [[nodiscard]] static Library open(const std::string &path) {
    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char *why = dlerror();
      throw Error(MINIMAGE_STATUS_UNSUPPORTED,
                  "minimage: cannot open " + path + ": " +
                      (why != nullptr ? why : "dlopen failed"));
    }
    EntryPoints entry;
    resolve(handle, "minimage_abi_stamp", entry.abi_stamp);
    resolve(handle, "minimage_abi_compatible", entry.abi_compatible);
    resolve(handle, "minimage_ops_get", entry.ops_get);
    return Library(detail::negotiate(entry, handle));
  }
#endif

  /// The ABI the library was built for.
  [[nodiscard]] Stamp stamp() const noexcept { return state_->stamp; }

  /// The version string of the library.
  [[nodiscard]] std::string version() const {
    const char *text = detail::need(state_->ops.version, "version")();
    return text != nullptr ? std::string(text) : std::string();
  }

  /// What the library can do.
  [[nodiscard]] Capabilities capabilities() const {
    minimage_caps raw{};
    raw.size = static_cast<std::uint32_t>(sizeof raw);
    state_->check(
        detail::need(state_->ops.capabilities, "capabilities")(&raw),
        "capabilities");
    Capabilities out;
    out.lattice = raw.lattice != 0;
    out.open_axes = raw.open_axes != 0;
    out.wrap_diffs = raw.wrap_diffs != 0;
    out.wrap_pairs = raw.wrap_pairs != 0;
    out.nearest_diffs = raw.nearest_diffs != 0;
    out.nearest_pairs = raw.nearest_pairs != 0;
    out.wrap_points = raw.wrap_points != 0;
    out.in_place = raw.in_place != 0;
    out.wrap_open_sheared = raw.wrap_open_sheared != 0;
    return out;
  }

  /// A lattice from three edge lengths along x, y, z. A negative edge is
  /// stored as its magnitude. Throws `Error` with
  /// `MINIMAGE_STATUS_BAD_CELL` for a zero or non-finite periodic edge.
  [[nodiscard]] Lattice
  diagonal(const std::array<double, 3> &edges,
           const std::array<bool, 3> &periodic = {true, true, true},
           const std::array<double, 3> &origin = {0.0, 0.0, 0.0}) const {
    const std::array<std::uint32_t, 3> mask = flags(periodic);
    minimage_lattice *raw = nullptr;
    state_->check(
        detail::need(state_->ops.lattice_diagonal, "lattice_diagonal")(
            edges.data(), mask.data(), origin.data(), &raw),
        "lattice_diagonal");
    return Lattice(state_, raw);
  }

  /// A lattice from the vectors a, b, c, row-major. Throws `Error` with
  /// `MINIMAGE_STATUS_BAD_CELL` when the periodic vectors are not a
  /// lattice.
  [[nodiscard]] Lattice
  general(const std::array<double, 9> &rows,
          const std::array<bool, 3> &periodic = {true, true, true},
          const std::array<double, 3> &origin = {0.0, 0.0, 0.0}) const {
    const std::array<std::uint32_t, 3> mask = flags(periodic);
    minimage_lattice *raw = nullptr;
    state_->check(
        detail::need(state_->ops.lattice_general, "lattice_general")(
            rows.data(), mask.data(), origin.data(), &raw),
        "lattice_general");
    return Lattice(state_, raw);
  }

  /// The negotiated table: a null member is an operation the library
  /// does not have.
  [[nodiscard]] const minimage_ops &ops() const noexcept {
    return state_->ops;
  }

private:
  explicit Library(std::shared_ptr<const detail::State> state)
      : state_(std::move(state)) {}

  static std::array<std::uint32_t, 3>
  flags(const std::array<bool, 3> &periodic) {
    return {periodic[0] ? 1u : 0u, periodic[1] ? 1u : 0u,
            periodic[2] ? 1u : 0u};
  }

#if !defined(_WIN32)
  /// A function pointer through an object pointer, as `dlsym` returns
  /// it. A missing symbol leaves `slot` null.
  template <typename Fn>
  static void resolve(void *handle, const char *name, Fn &slot) {
    void *symbol = dlsym(handle, name);
    static_assert(sizeof(Fn) == sizeof(void *),
                  "function and object pointers have one size");
    slot = reinterpret_cast<Fn>(symbol);
  }
#endif

  std::shared_ptr<const detail::State> state_;
};

} // namespace abi
} // namespace minimage

#endif // MINIMAGE_ABI_HPP
