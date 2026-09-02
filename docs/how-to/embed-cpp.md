# Embed from C++

```cpp
#include "minimage.hpp"

const minimage::Cell box = minimage::Cell::ortho(10.0, 10.0, 10.0);
const double d2 = box.dist2({0.2, 0.0, 0.0}, {9.4, 0.0, 0.0});
```

The header is a RAII wrap of `minimage.h`. Failures throw
`minimage::Error`.
