`Error::Empty` now reads `empty input` instead of `no pairs`. It covers
an empty or misaligned ring-polymer input and a zero-dimension
`PeriodicSpace`, and the C API reports the same text through
`minimage_last_error`.
