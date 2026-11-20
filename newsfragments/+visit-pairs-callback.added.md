The C ABI adds `minimage_visit_pairs_within_images` and the
`mi_image_pair_visitor` callback type, and the C++ header adds
`minimage::visit_pairs_within_images`. They walk every periodic image within
a cutoff in one pass without allocating a pair buffer. A nonzero callback
return stops the C walk early, and a C++ visitor that throws stops the walk
and has its exception rethrown to the caller once the C call has returned.
`minimage::pairs_within_images` now collects through the same visitor.
