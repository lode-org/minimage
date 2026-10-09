//! AVX-512 intrinsics are stable from Rust 1.89. Newer compilers build
//! the AVX-512 kernels, picked at run time; older ones keep AVX2.

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let minor = env::var_os("RUSTC")
        .and_then(|rustc| Command::new(rustc).arg("--version").output().ok())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|v| v.split('.').nth(1).and_then(|m| m.parse::<u32>().ok()))
        .unwrap_or(0);
    if minor >= 80 {
        println!("cargo:rustc-check-cfg=cfg(minimage_avx512)");
    }
    let x86_64 = env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "x86_64");
    if minor >= 89 && x86_64 {
        println!("cargo:rustc-cfg=minimage_avx512");
    }
}
