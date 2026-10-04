/* Same C driver against this tree and against published 0.1.2.
 * Prints `name ns_per_pair=<f64>` for the seams-shaped calls.
 */
#include "minimage.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { N = 4096 };

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void report(const char *name, double (*body)(void)) {
  double start = now_sec();
  unsigned warm = 0;
  while ((now_sec() - start) < 0.03 && warm < 2000u) {
    body();
    warm++;
  }
  double elapsed = now_sec() - start;
  if (elapsed < 1e-6) {
    elapsed = 1e-6;
  }
  unsigned repeats = (unsigned)((0.10 / elapsed) * (double)warm);
  if (repeats < 1u) {
    repeats = 1u;
  }
  if (repeats > 4000u) {
    repeats = 4000u;
  }
  start = now_sec();
  for (unsigned i = 0; i < repeats; i++) {
    body();
  }
  double ns = (now_sec() - start) * 1e9;
  printf("%s ns_per_pair=%.6f\n", name, ns / ((double)repeats * (double)N));
}

static double *ps;
static double *qs;
static double *out;
static mi_cell box;

static double run_dist2(void) {
  double acc = 0.0;
  for (int i = 0; i < N; i++) {
    double got = 0.0;
    if (mi_dist2(&box, ps + (size_t)i * 3, qs + (size_t)i * 3, &got) != 0) {
      fprintf(stderr, "mi_dist2 failed\n");
      exit(1);
    }
    acc += got;
  }
  return acc;
}

static double run_many(void) {
  if (mi_dist2_many(&box, ps, qs, N, out) != 0) {
    fprintf(stderr, "mi_dist2_many failed\n");
    exit(1);
  }
  return out[0];
}

static double run_pairs(void) {
  if (mi_dist2_pairs(&box, ps, qs, N, out) != 0) {
    fprintf(stderr, "mi_dist2_pairs failed\n");
    exit(1);
  }
  return out[0];
}

static double lcg(unsigned long long *s, double span) {
  *s = *s * 6364136223846793005ull + 1ull;
  double u = (double)(*s >> 33) / (double)(1ull << 31);
  return u * span;
}

int main(void) {
  box = mi_cell_ortho(10.0, 11.0, 12.0);
  const double left[3] = {0.2, 0.0, 0.0};
  const double right[3] = {9.4, 0.0, 0.0};
  double got = 0.0;
  if (mi_dist2(&box, left, right, &got) != 0 || fabs(got - 0.64) > 1e-9) {
    fprintf(stderr, "in-box mi_dist2 %g wanted 0.64\n", got);
    return 1;
  }
  ps = calloc((size_t)N * 3, sizeof(double));
  qs = calloc((size_t)N * 3, sizeof(double));
  out = calloc((size_t)N, sizeof(double));
  if (ps == NULL || qs == NULL || out == NULL) {
    return 1;
  }
  unsigned long long state = 0x123456789abull;
  const double lens[3] = {10.0, 11.0, 12.0};
  for (int i = 0; i < N; i++) {
    for (int a = 0; a < 3; a++) {
      ps[(size_t)i * 3 + (size_t)a] = lcg(&state, lens[a]);
      qs[(size_t)i * 3 + (size_t)a] = lcg(&state, lens[a]);
    }
  }
  if (mi_dist2_many(&box, ps, qs, N, out) != 0) {
    fprintf(stderr, "mi_dist2_many failed\n");
    return 1;
  }
  double one = 0.0;
  if (mi_dist2(&box, ps, qs, &one) != 0 || fabs(one - out[0]) > 1e-8) {
    fprintf(stderr, "mi_dist2_many disagrees with mi_dist2\n");
    return 1;
  }
  report("c_dist2", run_dist2);
  report("c_dist2_many", run_many);
  report("c_dist2_pairs", run_pairs);
  free(ps);
  free(qs);
  free(out);
  return 0;
}
