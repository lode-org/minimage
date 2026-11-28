# A host with a C and a C++ compiler and no Rust toolchain. It takes the
# minimage declarations from the headers-only subproject and loads the
# library at run time. scripts/check-headers-only.sh assembles the
# subproject the way packaging/wrapdb/minimage-headers.wrap does and
# configures this with no cargo or rustc on PATH.
project(
  'minimage-headers-consumer',
  ['c', 'cpp'],
  default_options: ['c_std=c11', 'cpp_std=c++17', 'warning_level=2'],
)

minimage_headers = dependency('minimage-headers')
cc = meson.get_compiler('c')
dl = cc.find_library('dl', required: false)
executable(
  'abi_host_c',
  'abi_host.c',
  c_args: ['-D_DEFAULT_SOURCE'],
  dependencies: [minimage_headers, dl, cc.find_library('m', required: false)],
)
executable(
  'abi_host_cpp',
  'abi_host.cpp',
  dependencies: [minimage_headers, dl],
)
