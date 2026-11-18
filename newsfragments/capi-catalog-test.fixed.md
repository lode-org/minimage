A test now parses the C API catalog in `docs/orgmode/reference.org` and
compares it with the `minimage_*` symbols exported by the built cdylib.
A symbol that is documented but not exported, or exported but not
documented, fails `cargo test`.
