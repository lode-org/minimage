#include "minimage.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct VisitorFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

int failures = 0;

void expect(bool ok, const char *what) {
  if (!ok) {
    std::cerr << "FAILED: " << what << "\n";
    ++failures;
  }
}

} // namespace

int main() {
  const minimage::Cell cell = minimage::Cell::ortho(4.0, 4.0, 4.0);
  const double xyz[] = {0.4, 0.5, 0.6, 1.1, 0.2, 3.3};
  const std::size_t n = 2;
  const double cutoff = 4.5;

  const auto all = minimage::pairs_within_images(cell, xyz, n, cutoff);
  expect(!all.empty(), "cutoff yields image pairs");

  std::size_t seen = 0;
  const std::size_t total = minimage::visit_pairs_within_images(
      cell, xyz, n, cutoff, [&seen](const mi_image_pair &) { ++seen; });
  expect(total == all.size(), "visitor count matches collector");
  expect(seen == all.size(), "visitor sees every pair");

  std::size_t calls = 0;
  bool caught = false;
  try {
    minimage::visit_pairs_within_images(
        cell, xyz, n, cutoff, [&calls](const mi_image_pair &) {
          if (++calls == 3) {
            throw VisitorFailure("stop here");
          }
        });
  } catch (const VisitorFailure &e) {
    caught = std::string(e.what()) == "stop here";
  }
  expect(caught, "visitor exception reaches the caller with its type");
  expect(calls == 3, "walk stops at the throwing image");

  int thrown = 0;
  try {
    minimage::visit_pairs_within_images(
        cell, xyz, n, cutoff, [](const mi_image_pair &) { throw 42; });
  } catch (int value) {
    thrown = value;
  }
  expect(thrown == 42, "non-standard exception objects are rethrown");

  bool library_error = false;
  try {
    minimage::visit_pairs_within_images(
        minimage::Cell::ortho(4.0, 4.0, 4.0), xyz, n, -1.0,
        [](const mi_image_pair &) {});
  } catch (const minimage::Error &) {
    library_error = true;
  }
  expect(library_error, "library failures still throw minimage::Error");

  const auto again = minimage::pairs_within_images(cell, xyz, n, cutoff);
  expect(again.size() == all.size(), "library is usable after a throw");

  if (failures != 0) {
    return 1;
  }
  std::cout << "visitor pairs " << all.size() << "\n";
  return 0;
}
