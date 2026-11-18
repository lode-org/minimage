//! The reference table in docs/orgmode/reference.org lists every `minimage_*`
//! symbol the default C ABI exports. This test compares that set to `nm` on
//! the built cdylib so either side cannot drift.

#![cfg(feature = "capi")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Keep sorted; must match the `minimage_*` rows in reference.org.
const DOCUMENTED: &[&str] = &[
    "minimage_cell_from_ase",
    "minimage_cell_from_con",
    "minimage_cell_from_con_box",
    "minimage_cell_from_lammps",
    "minimage_cell_from_lammps_bounds",
    "minimage_cell_from_vectors",
    "minimage_cell_from_vesin",
    "minimage_displacement",
    "minimage_displacement_euclidean",
    "minimage_displacement_euclidean_pbc",
    "minimage_dist2",
    "minimage_dist2_euclidean_many",
    "minimage_dist2_euclidean_pairs",
    "minimage_dist2_euclidean_pairs_warm",
    "minimage_dist2_euclidean_pbc",
    "minimage_dist2_fixed",
    "minimage_dist2_fixed32",
    "minimage_dist2_many",
    "minimage_dist2_many_fixed",
    "minimage_dist2_many_fixed32",
    "minimage_dist2_ortho_diffs",
    "minimage_dist2_pairs",
    "minimage_dist2_pairs_fixed",
    "minimage_dist2_pairs_fixed32",
    "minimage_dist2_shifted_many",
    "minimage_filter_shifted",
    "minimage_fixed32_many",
    "minimage_fixed_many",
    "minimage_last_error",
    "minimage_pbc_cell_from_ase",
    "minimage_reduce_pairs",
    "minimage_ring_centroid",
    "minimage_version",
    "minimage_wrap_many",
];

fn cdylib_path() -> PathBuf {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .to_string_lossy()
            .into_owned()
    });
    let base = Path::new(&target_dir).join(profile);
    if cfg!(target_os = "macos") {
        base.join("libminimage.dylib")
    } else if cfg!(windows) {
        base.join("minimage.dll")
    } else {
        base.join("libminimage.so")
    }
}

fn nm_exports(library: &Path) -> BTreeSet<String> {
    let output = Command::new("nm")
        .args([
            "-g",
            "--defined-only",
            library.to_str().expect("utf-8 path"),
        ])
        .output()
        .expect("run nm");
    assert!(
        output.status.success(),
        "nm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .filter_map(|line| line.split_whitespace().last())
        .filter(|sym| sym.starts_with("minimage_"))
        .map(|sym| sym.to_string())
        .collect()
}

#[test]
fn reference_table_matches_cdylib_exports() {
    let library = cdylib_path();
    assert!(
        library.exists(),
        "missing {:?}; run `cargo build --lib` first",
        library
    );
    let exported = nm_exports(&library);
    let documented: BTreeSet<String> = DOCUMENTED.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        exported, documented,
        "reference.org and the cdylib disagree on exported symbols"
    );
}
