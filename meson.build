project(
  'minimage',
  ['c', 'cpp'],
  # Keep in lockstep with Cargo.toml [package].version (source of truth).
  version: '0.1.0',
  meson_version: '>=1.2.0',
  license: 'MIT',
  default_options: [
    'buildtype=debugoptimized',
    'warning_level=2',
    'c_std=c11',
    'cpp_std=c++17',
  ],
)

cc = meson.get_compiler('c')
cargo_prog = find_program('cargo', required: true)
py = find_program('python3', 'python')
_install = not meson.is_subproject()
_msproot = meson.project_source_root()
_mbproot = meson.project_build_root()

if host_machine.system() == 'windows'
  cargo_shared = 'minimage.dll'
  cargo_static = 'minimage.lib'
else
  _so = host_machine.system() == 'darwin' ? '.dylib' : '.so'
  cargo_shared = 'libminimage' + _so
  cargo_static = 'libminimage.a'
endif

_profile = get_option('buildtype') == 'debug' ? 'debug' : 'release'

minimage_cargo = custom_target(
  'minimage-cargo',
  input: files(
    'Cargo.toml',
    'src/lib.rs',
    'src/cell.rs',
    'src/error.rs',
    'src/batch.rs',
    'src/pairs.rs',
    'src/capi.rs',
  ),
  output: [cargo_shared, cargo_static],
  command: [
    py, '-c',
    'import shutil, subprocess, sys; from pathlib import Path; ' +
    'cargo, src, target, profile, shared, archive, out_s, out_a = sys.argv[1:9]; ' +
    'cmd = [cargo, "rustc", "--lib", "--manifest-path", str(Path(src) / "Cargo.toml"), ' +
    '"--target-dir", target, "--features", "capi"]; ' +
    'Path(src, "Cargo.lock").is_file() and cmd.append("--locked"); ' +
    'profile == "release" and cmd.append("--release"); ' +
    'rustc = ["--crate-type=cdylib", "--crate-type=staticlib"]; ' +
    'sys.platform.startswith("linux") and rustc.append("-Clink-arg=-Wl,-soname," + shared); ' +
    'sys.platform == "darwin" and rustc.append("-Clink-arg=-Wl,-install_name," + chr(64) + "rpath/" + shared); ' +
    'cmd.extend(["--"] + rustc); subprocess.check_call(cmd); ' +
    'built = Path(target) / profile; ' +
    'shutil.copy2(built / shared, out_s); shutil.copy2(built / archive, out_a)',
    cargo_prog.full_path(),
    _msproot,
    _mbproot / 'cargo-target',
    _profile,
    cargo_shared,
    cargo_static,
    '@OUTPUT0@',
    '@OUTPUT1@',
  ],
  build_by_default: true,
  install: _install,
  install_dir: [get_option('libdir'), get_option('libdir')],
)

rust_sys = [dependency('threads')]
if host_machine.system() == 'linux'
  rust_sys += [
    cc.find_library('dl', required: true),
    cc.find_library('m', required: true),
  ]
elif host_machine.system() == 'darwin'
  rust_sys += [cc.find_library('m', required: true)]
elif host_machine.system() == 'windows'
  rust_sys += [
    cc.find_library('ws2_32', required: true),
    cc.find_library('userenv', required: true),
    cc.find_library('bcrypt', required: true),
    cc.find_library('ntdll', required: true),
  ]
endif

minimage_built_h = custom_target(
  'minimage-built-h',
  input: minimage_cargo,
  output: 'minimage_built.h',
  command: [
    py,
    '-c',
    'from pathlib import Path; Path("@OUTPUT@").write_text("#pragma once\\n")',
  ],
  build_by_default: true,
)

_use_static = meson.is_subproject() or get_option('default_library') == 'static'
_link_idx = _use_static ? 1 : 0

minimage_dep = declare_dependency(
  link_args: [minimage_cargo[_link_idx].full_path()],
  sources: minimage_built_h,
  include_directories: include_directories('include'),
  dependencies: rust_sys,
  version: meson.project_version(),
)
meson.override_dependency('minimage', minimage_dep)

if _install
  install_headers('include/minimage.h', 'include/minimage.hpp')
  pkg = import('pkgconfig')
  pkg.generate(
    name: 'minimage',
    filebase: 'minimage',
    description: 'Minimum-image convention for periodic parallelepiped cells',
    version: meson.project_version(),
    url: 'https://github.com/lode-org/minimage',
    libraries: ['-L${libdir}', '-lminimage'],
    libraries_private: rust_sys,
    subdirs: '.',
  )
endif

if get_option('with_tests')
  test(
    'cargo-tests',
    cargo_prog,
    args: ['test', '--manifest-path', _msproot / 'Cargo.toml'],
    timeout: 300,
  )
endif

if get_option('with_examples')
  subdir('examples')
endif
