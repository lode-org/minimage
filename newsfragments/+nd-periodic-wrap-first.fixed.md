`PeriodicSpace` and the N-dimensional C API now reduce each coordinate
into the periodic cell before subtracting in `displacement`, `dist2`,
`distance`, `dist2_many`, `dist2_pairs`, `axis_min_dist2`, and
`aabb_min_dist2`. Extreme finite coordinates such as `+-f64::MAX` on
periodic axes now produce finite distances and displacements instead of
overflowing the raw difference to `inf` or `NaN`.
