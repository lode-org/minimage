#include "minimage.h"

#include <stdio.h>

struct stop_state {
  size_t calls;
  size_t limit;
};

static int count_and_stop(void *userdata, const mi_image_pair *pair) {
  struct stop_state *state = (struct stop_state *)userdata;
  (void)pair;
  state->calls += 1;
  return state->calls >= state->limit ? 1 : 0;
}

int main(void) {
  const mi_cell cell = mi_cell_ortho(4.0, 4.0, 4.0);
  const double xyz[6] = {0.4, 0.5, 0.6, 1.1, 0.2, 3.3};
  const double cutoff = 4.5;
  size_t total = 0;
  if (minimage_visit_pairs_within_images(&cell, NULL, xyz, 2, cutoff, 1, NULL,
                                         NULL, &total) != 0) {
    fprintf(stderr, "count: %s\n", minimage_last_error());
    return 1;
  }
  if (total < 4) {
    fprintf(stderr, "expected several pairs, got %zu\n", total);
    return 1;
  }

  struct stop_state state = {0, 3};
  size_t stopped = 0;
  if (minimage_visit_pairs_within_images(&cell, NULL, xyz, 2, cutoff, 1,
                                         count_and_stop, &state,
                                         &stopped) != 0) {
    fprintf(stderr, "stop: %s\n", minimage_last_error());
    return 1;
  }
  if (state.calls != 3 || stopped != 3) {
    fprintf(stderr, "stop after 3: calls %zu count %zu\n", state.calls,
            stopped);
    return 1;
  }

  printf("visitor pairs %zu stopped at %zu\n", total, stopped);
  return 0;
}
