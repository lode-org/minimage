// A C++ host that compiles against the minimage headers alone and takes
// the library at run time through minimage_abi.hpp: no minimage symbol is
// linked. The path of the shared library is argv[1], or MINIMAGE_LIBRARY.
// Results are compared with the entry points a linked host calls
// directly, resolved here by name.
// Exit 0 on success; a failure prints its line and exits 1.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "minimage_abi.hpp"

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,  \
                   #cond);                                                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

using minimage::abi::Rows;

template <typename Fn> Fn load(void *lib, const char *name) {
  void *found = dlsym(lib, name);
  if (found == nullptr) {
    std::fprintf(stderr, "missing symbol %s\n", name);
    std::exit(1);
  }
  return reinterpret_cast<Fn>(found);
}

std::uint64_t rng_state = UINT64_C(0xd1b54a32d192ed03);

double next(double scale) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (static_cast<double>(rng_state >> 11) / 9007199254740992.0 * 2.0 -
          1.0) *
         scale;
}

Rows points(std::size_t n, double scale) {
  Rows out(n);
  for (auto &row : out) {
    row = {next(scale), next(scale), next(scale)};
  }
  return out;
}

bool close_to(double a, double b) {
  const double scale = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
  return std::fabs(a - b) <= 1e-9 * (1.0 + scale);
}

template <typename Call> std::int32_t status_of(Call &&call) {
  try {
    call();
  } catch (const minimage::abi::Error &err) {
    CHECK(std::string(err.what()).rfind("minimage: ", 0) == 0);
    CHECK(err.recoverable() == (err.status() > 0));
    return err.status();
  }
  return MINIMAGE_STATUS_OK;
}

} // namespace

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : std::getenv("MINIMAGE_LIBRARY");
  if (path == nullptr) {
    std::fprintf(stderr, "usage: %s <path to libminimage>\n", argv[0]);
    return 2;
  }

  // A file that is not there, and a library without the negotiation.
  CHECK(status_of([] {
          (void)minimage::abi::Library::open("/nonexistent/libminimage.so");
        }) == MINIMAGE_STATUS_UNSUPPORTED);
  CHECK(status_of([] {
          (void)minimage::abi::Library(minimage::abi::EntryPoints{});
        }) == MINIMAGE_STATUS_UNSUPPORTED);

  const minimage::abi::Library lib = minimage::abi::Library::open(path);
  const minimage::abi::Stamp header = minimage::abi::header_stamp();
  const minimage::abi::Stamp stamp = lib.stamp();
  std::printf("header ABI %u.%u layout %u, library ABI %u.%u layout %u, "
              "version %s\n",
              static_cast<unsigned>(header.major),
              static_cast<unsigned>(header.minor),
              static_cast<unsigned>(header.layout_revision),
              static_cast<unsigned>(stamp.major),
              static_cast<unsigned>(stamp.minor),
              static_cast<unsigned>(stamp.layout_revision),
              lib.version().c_str());
  CHECK(stamp.major == header.major);
  CHECK(stamp.layout_revision == header.layout_revision);
  CHECK(!lib.version().empty());

  const minimage::abi::Capabilities caps = lib.capabilities();
  CHECK(caps.lattice && caps.open_axes && caps.in_place);
  CHECK(caps.wrap_diffs && caps.wrap_pairs && caps.wrap_points);
  CHECK(caps.nearest_diffs && caps.nearest_pairs);
  CHECK(!caps.wrap_open_sheared);

  // The entry points a linked host calls directly. The library is the
  // one `lib` holds open, so this second handle is the same image.
  void *raw = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  CHECK(raw != nullptr);
  const auto cell_from_vectors =
      load<int (*)(const double *, const double *, const double *,
                   const double *, mi_cell *)>(raw,
                                               "minimage_cell_from_vectors");
  const auto wrap_many =
      load<int (*)(const mi_cell *, const double *, std::size_t, double *)>(
          raw, "minimage_wrap_many");
  const auto displacement_euclidean =
      load<int (*)(const mi_cell *, const double *, const double *,
                   double *)>(raw, "minimage_displacement_euclidean");

  const Rows ps = points(500, 30.0);
  const Rows qs = points(500, 30.0);
  Rows diffs(ps.size());
  for (std::size_t k = 0; k < ps.size(); ++k) {
    for (std::size_t c = 0; c < 3; ++c) {
      diffs[k][c] = qs[k][c] - ps[k][c];
    }
  }

  const std::array<std::array<double, 9>, 4> cells = {{
      {15, 0, 0, 0, 15, 0, 0, 0, 15},
      {-15, 0, 0, 0, -15, 0, 0, 0, 15},
      {10, 0, 0, 3, 9, 0, 1, 2, 8},
      {-10, 0, 0, 3, -9, 0, 1, 2, -8},
  }};
  const std::array<double, 3> origin = {0.3, -1.1, 2.0};
  for (const auto &rows : cells) {
    const minimage::abi::Lattice lattice =
        lib.general(rows, {true, true, true}, origin);
    const minimage::abi::LatticeInfo info = lattice.info();
    mi_cell cell;
    CHECK(cell_from_vectors(rows.data(), rows.data() + 3, rows.data() + 6,
                            origin.data(), &cell) == 0);
    const std::array<double, 9> reported = {cell.ax, cell.ay, cell.az,
                                            cell.bx, cell.by, cell.bz,
                                            cell.cx, cell.cy, cell.cz};
    CHECK(info.rows == reported);
    CHECK(info.origin == origin);
    if (info.orthorhombic) {
      CHECK(info.rows[0] > 0.0 && info.rows[4] > 0.0 && info.rows[8] > 0.0);
    }

    Rows want(diffs.size());
    CHECK(wrap_many(&cell, diffs.front().data(), diffs.size(),
                    want.front().data()) == 0);
    CHECK(lattice.wrap_diffs(diffs) == want);
    const Rows pairs = lattice.wrap_pairs(ps, qs);
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      for (std::size_t c = 0; c < 3; ++c) {
        CHECK(close_to(pairs[k][c], want[k][c]));
      }
    }
    // In place, through the pointer form.
    Rows same = diffs;
    lattice.wrap_diffs(same.front().data(), same.size(), same.front().data());
    for (std::size_t k = 0; k < same.size(); ++k) {
      for (std::size_t c = 0; c < 3; ++c) {
        CHECK(close_to(same[k][c], want[k][c]));
      }
    }

    for (std::size_t k = 0; k < ps.size(); ++k) {
      CHECK(displacement_euclidean(&cell, ps[k].data(), qs[k].data(),
                                   want[k].data()) == 0);
    }
    CHECK(lattice.nearest_pairs(ps, qs) == want);
    const Rows nearest = lattice.nearest_diffs(diffs);
    for (std::size_t k = 0; k < nearest.size(); ++k) {
      for (std::size_t c = 0; c < 3; ++c) {
        CHECK(close_to(nearest[k][c], want[k][c]));
      }
    }

    // A wrapped point stays where it is when wrapped again.
    const Rows folded = lattice.wrap_points(ps);
    CHECK(lattice.wrap_points(folded) == folded);
    if (info.orthorhombic) {
      for (const auto &row : folded) {
        for (std::size_t c = 0; c < 3; ++c) {
          const double x = row[c] - origin[c];
          CHECK(x >= 0.0 && x < info.rows[4 * c]);
        }
      }
    }
  }

  // A slab keeps its open axis; the two negative edges are one lattice
  // with the positive ones.
  const minimage::abi::Lattice slab =
      lib.diagonal({-15.0, -15.0, 15.0}, {true, true, false});
  const minimage::abi::Lattice positive =
      lib.diagonal({15.0, 15.0, 15.0}, {true, true, false});
  CHECK(slab.info().rows == positive.info().rows);
  const Rows wrapped = slab.wrap_diffs(diffs);
  CHECK(wrapped == positive.wrap_diffs(diffs));
  for (std::size_t k = 0; k < wrapped.size(); ++k) {
    CHECK(wrapped[k][2] == diffs[k][2]);
    CHECK(wrapped[k][0] >= -7.5 && wrapped[k][0] < 7.5);
    CHECK(wrapped[k][1] >= -7.5 && wrapped[k][1] < 7.5);
  }

  // What the library cannot do is a status, never another result.
  const minimage::abi::Lattice sheared_slab =
      lib.general({10, 0, 0, 3, 9, 0, 1, 2, 8}, {true, true, false});
  CHECK(status_of([&] { (void)sheared_slab.wrap_diffs(diffs); }) ==
        MINIMAGE_STATUS_UNSUPPORTED);
  CHECK(sheared_slab.nearest_diffs(diffs).size() == diffs.size());
  CHECK(status_of([&] { (void)lib.diagonal({10.0, 0.0, 10.0}); }) ==
        MINIMAGE_STATUS_BAD_CELL);
  CHECK(status_of([&] {
          (void)slab.wrap_pairs(ps, Rows(ps.size() + 1));
        }) == MINIMAGE_STATUS_BAD_COUNT);
  // An empty batch reads no buffer.
  CHECK(slab.wrap_points(Rows{}).empty());

  // A lattice outlives the Library object it came from, and moves.
  minimage::abi::Lattice kept = [&] {
    const minimage::abi::Library second = minimage::abi::Library::open(path);
    return second.diagonal({10.0, 10.0, 10.0});
  }();
  minimage::abi::Lattice moved = std::move(kept);
  CHECK(kept.get() == nullptr && moved.get() != nullptr);
  // The moved-from object holds no handle, and says so with a status.
  CHECK(status_of([&] { (void)kept.info(); }) == MINIMAGE_STATUS_NULL_POINTER);
  CHECK(status_of([&] { (void)kept.wrap_points(Rows{{1.0, 2.0, 3.0}}); }) ==
        MINIMAGE_STATUS_NULL_POINTER);
  const Rows one = moved.wrap_points(Rows{{12.5, -0.5, 3.0}});
  CHECK(one.front()[0] == 2.5 && one.front()[1] == 9.5 && one.front()[2] == 3.0);

  dlclose(raw);
  std::printf("abi host (C++): ok\n");
  return 0;
}
