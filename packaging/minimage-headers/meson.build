# The minimage headers as a Meson dependency, with no Rust toolchain.
#
# A host that loads libminimage at run time needs the declarations and
# nothing to link. packaging/wrapdb/minimage-headers.wrap fetches the
# minimage sources and lays this file over their meson.build
# (patch_directory), so configuring the subproject never looks for cargo
# or rustc. The version comes from Cargo.toml through Python, which runs
# Meson anyway.
project(
  'minimage-headers',
  'c',
  version: run_command('scripts/crate_version.py', check: true).stdout().strip(),
  license: 'MIT',
  meson_version: '>=1.2.0',
)

minimage_headers_dep = declare_dependency(
  include_directories: include_directories('include'),
  version: meson.project_version(),
)
meson.override_dependency('minimage-headers', minimage_headers_dep)

if not meson.is_subproject()
  install_headers('include/minimage.h', 'include/minimage_abi.hpp')
endif
