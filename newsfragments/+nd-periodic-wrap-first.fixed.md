`PeriodicSpace` displacements are now the exact minimum image of `q - p`,
rounded once. The same exact images feed `dist2`, `distance`, `dist2_many`,
`dist2_pairs`, `axis_min_dist2`, `aabb_min_dist2`, and the N-dimensional C
API's `minimage_nd_dist2`, `minimage_nd_distance`, `minimage_nd_dist2_many`,
`minimage_nd_axis_min_dist2`, and `minimage_nd_aabb_min_dist2`. Extreme
finite coordinates such as `+-f64::MAX` give finite results instead of
`inf` or `NaN`. Large coordinates with small periods no longer lose their
image: with period `1e-6`, `-10 -> 0` gives `4.525188817411374e-16`, not
zero. Interval and box bounds never exceed the correctly rounded exact
minimum, nor `dist2` to any point in the box.
