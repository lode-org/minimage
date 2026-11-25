`PeriodicSpace` and the N-dimensional C API now reduce coordinates into
the periodic cell before subtracting in `displacement`, `dist2`,
`distance`, `dist2_many`, `dist2_pairs`, `axis_min_dist2`, and
`aabb_min_dist2` whenever the raw difference is a period or more apart.
Extreme finite coordinates such as `+-f64::MAX` on periodic axes now
produce finite distances and displacements instead of overflowing the raw
difference to `inf` or `NaN`. Raw differences shorter than the period are
kept, and coordinates more than `2^26` periods from the origin take the
exact remainder, so nearby points at huge image counts keep their
separation and interval bound instead of collapsing to zero.
