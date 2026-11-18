Documentation and CI now track every default `minimage_*` export. The
reference table lists `minimage_wrap_many`, `minimage_filter_shifted`, and
`minimage_ring_centroid`, and a test compares that table to the built cdylib.
`cbindgen.toml` lists the full export surface. CI runs `abi` tests under
`capi-compat-names`.