`PeriodicSpace` displacements are now the exact minimum image of `q - p`,
rounded once. The same exact images feed `dist2`, `distance`, `dist2_many`,
`dist2_pairs`, `axis_min_dist2`, `aabb_min_dist2`, and the N-dimensional C
API's `minimage_nd_dist2`, `minimage_nd_distance`, `minimage_nd_dist2_many`,
`minimage_nd_axis_min_dist2`, and `minimage_nd_aabb_min_dist2`. On a
periodic axis, finite coordinates up to `+-f64::MAX` now give a
displacement component in `[-period / 2, period / 2]` instead of `inf` or
`NaN`, so squared distances stay finite when the periods are moderate.
Open axes still subtract directly and can overflow, and a squared
component above `f64::MAX` still rounds to `inf`. Large coordinates with
small periods no longer lose their image: with period `1e-6`, `-10 -> 0`
gives `4.525188817411374e-16`, not zero. Interval and box bounds never
exceed the correctly rounded exact minimum, nor `dist2` to any point in
the box.
