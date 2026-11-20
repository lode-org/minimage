#include "minimage.hpp"

#include <cassert>

int main() {
  using minimage::wrap_ortho_half_strict;

  assert(wrap_ortho_half_strict(5.0, 10.0) == -5.0);
  assert(wrap_ortho_half_strict(15.0, 10.0) == -5.0);
  assert(wrap_ortho_half_strict(-5.0, 10.0) == -5.0);
  assert(wrap_ortho_half_strict(-15.0, 10.0) == -5.0);

  assert(wrap_ortho_half_strict(4.0, 10.0) == 4.0);
  assert(wrap_ortho_half_strict(6.0, 10.0) == -4.0);
  assert(wrap_ortho_half_strict(-4.0, 10.0) == -4.0);
  assert(wrap_ortho_half_strict(-6.0, 10.0) == 4.0);
}
