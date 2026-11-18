The default C ABI now exports `minimage_*` function symbols, using the full
crate name to avoid collisions with mimalloc's `mi_*` symbols. The
non-default `capi-compat-names` feature provides deprecated thin aliases for
the old names during migration.
