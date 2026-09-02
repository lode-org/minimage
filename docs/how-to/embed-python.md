# Embed from Python

```python
import minimage

cell = minimage.Cell.ortho(10.0, 10.0, 10.0)
print(cell.dist2([0.2, 0.0, 0.0], [9.4, 0.0, 0.0]))
```

The module is a thin PyO3 wrap of the Rust `Cell`.
