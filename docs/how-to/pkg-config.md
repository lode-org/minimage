# pkg-config

```
pkg-config --cflags --libs minimage
```

Meson and CMake both write `minimage.pc`. `Libs` includes `-L${libdir}`
so a wrap consumer can find `-lminimage`.
