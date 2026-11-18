//! The C ABI catalog in `docs/orgmode/reference.org` is the single source of
//! truth for the exported `minimage_*` symbols. This test parses the table
//! under the `C ABI` heading and compares it with the symbols `nm` reports
//! for a freshly built cdylib. A symbol that is documented but not exported,
//! or exported but not documented, fails the test.

#![cfg(all(feature = "capi", unix))]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREFIX: &str = "minimage_";

/// The Org verbatim spans (`=name=`) in one table cell.
fn code_spans(cell: &str) -> impl Iterator<Item = &str> {
    cell.split('=').skip(1).step_by(2)
}

/// Function names in the first column of the table under the `C ABI` heading.
/// Type rows such as `mi_cell` do not carry the `minimage_` prefix and are
/// skipped, as are the prose paragraphs and later language sections.
fn documented_symbols(org: &str) -> Vec<String> {
    let mut in_section = false;
    let mut symbols = Vec::new();
    for line in org.lines() {
        if line.starts_with("* ") {
            in_section = line.starts_with("* C ABI");
            continue;
        }
        if !in_section || !line.starts_with('|') {
            continue;
        }
        let first_cell = line.trim_start_matches('|').split('|').next().unwrap_or("");
        symbols.extend(
            code_spans(first_cell)
                .filter(|span| span.starts_with(PREFIX))
                .map(str::to_string),
        );
    }
    symbols
}

fn build_cdylib(manifest_dir: &Path) -> PathBuf {
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("capi-exports");
    let features = if cfg!(feature = "capi-compat-names") {
        "capi,capi-compat-names"
    } else {
        "capi"
    };
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(manifest_dir)
        .args(["build", "--lib", "--locked", "--no-default-features"])
        .args(["--features", features])
        .arg("--target-dir")
        .arg(&target_dir)
        .status()
        .expect("run cargo build --lib");
    assert!(status.success(), "cargo build --lib failed");
    let name = if cfg!(target_os = "macos") {
        "libminimage.dylib"
    } else {
        "libminimage.so"
    };
    let library = target_dir.join("debug").join(name);
    assert!(library.exists(), "missing cdylib at {library:?}");
    library
}

fn exported_symbols(library: &Path) -> BTreeSet<String> {
    let output = Command::new("nm")
        .args(["-g", "--defined-only"])
        .arg(library)
        .output()
        .expect("run nm");
    assert!(
        output.status.success(),
        "nm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(|sym| {
            if cfg!(target_os = "macos") {
                sym.strip_prefix('_').unwrap_or(sym)
            } else {
                sym
            }
        })
        .filter(|sym| sym.starts_with(PREFIX))
        .map(str::to_string)
        .collect()
}

#[test]
fn reference_catalog_matches_cdylib_exports() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let reference = manifest_dir.join("docs/orgmode/reference.org");
    let org = std::fs::read_to_string(&reference).expect("read docs/orgmode/reference.org");

    let listed = documented_symbols(&org);
    let documented: BTreeSet<String> = listed.iter().cloned().collect();
    assert!(
        !documented.is_empty(),
        "no `{PREFIX}*` rows found under the C ABI heading of {reference:?}"
    );
    assert_eq!(
        listed.len(),
        documented.len(),
        "reference.org documents a symbol more than once"
    );

    let exported = exported_symbols(&build_cdylib(manifest_dir));
    assert!(
        !exported.is_empty(),
        "cdylib exports no `{PREFIX}*` symbols"
    );

    let undocumented: Vec<_> = exported.difference(&documented).collect();
    let missing: Vec<_> = documented.difference(&exported).collect();
    assert!(
        undocumented.is_empty() && missing.is_empty(),
        "reference.org and the cdylib disagree\n  exported but not documented: {undocumented:?}\n  documented but not exported: {missing:?}"
    );
}

#[test]
fn catalog_parser_reads_only_c_abi_function_rows() {
    let org = "\
* Intro
| =minimage_ignored= | before the section |
* C ABI (=minimage_*=)
Prose mentions =minimage_prose= outside a table.
| Symbol | Role |
|--------+------|
| =mi_cell= | A type, not a function |
| =minimage_a= | Mentions =minimage_in_role= in the role column |
| =minimage_b=, =minimage_c= | Two symbols in one row |
* C++
| =minimage_cpp= | after the section |
";
    assert_eq!(
        documented_symbols(org),
        ["minimage_a", "minimage_b", "minimage_c"]
    );
}
