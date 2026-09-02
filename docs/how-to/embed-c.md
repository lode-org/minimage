# Embed from C

```c
#include "minimage.h"

mi_cell box = mi_cell_ortho(10.0, 10.0, 10.0);
const double p[3] = {0.2, 0.0, 0.0};
const double q[3] = {9.4, 0.0, 0.0};
double d2 = 0.0;
if (mi_dist2(&box, p, q, &d2) != 0) {
    fprintf(stderr, "%s\n", mi_last_error());
}
```

Link `libminimage` and add `include/` to the include path. Meson
exposes `minimage_dep`. CMake exposes `minimage::minimage`.
