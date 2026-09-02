#include "minimage.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_dist2(const char *label, const mi_cell *box, const double *p,
                       const double *q, double expect) {
  double got = -1.0;
  if (mi_dist2(box, p, q, &got) != 0) {
    fprintf(stderr, "%s: %s\n", label, mi_last_error());
    return 1;
  }
  if (fabs(got - expect) > 1e-12) {
    fprintf(stderr, "%s: dist2 %g != %g\n", label, got, expect);
    return 1;
  }
  printf("%s %g\n", label, got);
  return 0;
}

int main(void) {
  const double left[3] = {0.2, 0.0, 0.0};
  const double right[3] = {9.4, 0.0, 0.0};
  const mi_cell ortho = mi_cell_ortho(10.0, 10.0, 10.0);
  mi_cell sheared;
  if (mi_cell_from_lammps_bounds(15.0, 8.660254037844386, 10.0, 5.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, &sheared) != 0) {
    fprintf(stderr, "%s\n", mi_last_error());
    return 1;
  }
  const double p[3] = {0.2, 0.1, 1.0};
  const double q[3] = {9.7, 0.1, 1.0};
  if (check_dist2("ortho", &ortho, left, right, 0.64) != 0 ||
      check_dist2("sheared", &sheared, p, q, 0.25) != 0) {
    return 1;
  }

  const int pairs[] = {0, 1, 0, 0, 0, 1, 2, 3};
  int out[8];
  size_t kept = 0;
  if (mi_reduce_pairs(pairs, 4, out, &kept) != 0 || kept != 2) {
    fprintf(stderr, "reduce_pairs failed\n");
    return 1;
  }
  return 0;
}
