`PeriodicSpace` and the `Metric` trait add independent optional periods to
N-dimensional feature vectors. Rust, C, and Python APIs wrap points and
differences, compute scalar and contiguous row-major batch distances,
and provide per-axis and point-to-AABB lower bounds for spatial-tree
pruning. The zero-sized `Euclidean` implementation uses the same trait.
