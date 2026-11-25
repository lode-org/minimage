//! The C ABI catalog in `docs/orgmode/reference.org` is the single source of
//! truth for the exported `minimage_*` symbols. This test parses the table
//! under the `C ABI` heading and compares it with the symbols `nm` reports
//! for a freshly built cdylib. A symbol that is documented but not exported,
//! or exported but not documented, fails the test. The `[export] include`
//! allowlist in `cbindgen.toml` and the declarations in the shipped
//! `include/minimage.h` are held to the same export set.

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

/// `minimage_*` entries of the `include` array under `[export]`.
fn allowlisted_symbols(toml: &str) -> Vec<String> {
    let mut in_export = false;
    let mut in_include = false;
    let mut symbols = Vec::new();
    for line in toml.lines() {
        let line = line.trim();
        if line.starts_with('[') && !in_include {
            in_export = line == "[export]";
            continue;
        }
        if in_export && line.starts_with("include") && line.contains('[') {
            in_include = true;
        }
        if !in_include {
            continue;
        }
        symbols.extend(
            line.split('"')
                .skip(1)
                .step_by(2)
                .filter(|name| name.starts_with(PREFIX))
                .map(str::to_string),
        );
        if line.contains(']') {
            in_include = false;
        }
    }
    symbols
}

/// Function names declared at the start of a header line. Doc comments,
/// continuation lines, and preprocessor lines are skipped.
fn declared_symbols(header: &str) -> Vec<String> {
    header
        .lines()
        .filter(|line| !line.starts_with([' ', '\t', '*', '/', '#']))
        .filter_map(|line| {
            let start = line.find(PREFIX)?;
            let rest = &line[start..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            rest[end..]
                .starts_with('(')
                .then(|| rest[..end].to_string())
        })
        .collect()
}

fn assert_same(what: &str, listed: &[String], exported: &BTreeSet<String>) {
    let set: BTreeSet<String> = listed.iter().cloned().collect();
    assert_eq!(
        listed.len(),
        set.len(),
        "{what} names a symbol more than once"
    );
    let extra: Vec<_> = set.difference(exported).collect();
    let missing: Vec<_> = exported.difference(&set).collect();
    assert!(
        extra.is_empty() && missing.is_empty(),
        "{what} and the cdylib disagree\n  exported but not listed: {missing:?}\n  listed but not exported: {extra:?}"
    );
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
fn cbindgen_allowlist_and_header_match_cdylib_exports() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let toml =
        std::fs::read_to_string(manifest_dir.join("cbindgen.toml")).expect("read cbindgen.toml");
    let header = std::fs::read_to_string(manifest_dir.join("include/minimage.h"))
        .expect("read include/minimage.h");

    let allowlisted = allowlisted_symbols(&toml);
    assert!(
        !allowlisted.is_empty(),
        "no `{PREFIX}*` entries in the [export] include list of cbindgen.toml"
    );
    let declared = declared_symbols(&header);
    assert!(
        !declared.is_empty(),
        "no `{PREFIX}*` declarations in include/minimage.h"
    );

    let exported = exported_symbols(&build_cdylib(manifest_dir));
    assert_same("cbindgen.toml [export] include", &allowlisted, &exported);
    assert_same("include/minimage.h", &declared, &exported);
}

#[test]
fn allowlist_parser_reads_only_export_include() {
    let toml = r#"
[fn]
include = ["minimage_fn_section"]

[export]
include = [
    "mi_cell",
    "minimage_a",
    "minimage_b", "minimage_c",
]
exclude = ["minimage_excluded"]

[export.rename]
"#;
    assert_eq!(
        allowlisted_symbols(toml),
        ["minimage_a", "minimage_b", "minimage_c"]
    );
}

#[test]
fn header_parser_reads_only_declarations() {
    let header = "\
/**
 * See `minimage_in_doc()` for details.
 */
int minimage_a(const double *p,
               double *out);
const char *minimage_b(void);
#define minimage_macro(x) (x)
    minimage_indented(void);
typedef struct mi_cell mi_cell;
";
    assert_eq!(declared_symbols(header), ["minimage_a", "minimage_b"]);
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
