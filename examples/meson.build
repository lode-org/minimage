wrap_pair_c = executable(
  'wrap_pair',
  'wrap_pair.c',
  dependencies: minimage_dep,
  build_rpath: meson.project_build_root(),
)

wrap_pair_cpp = executable(
  'wrap_pair_cpp',
  'wrap_pair.cpp',
  dependencies: minimage_dep,
  build_rpath: meson.project_build_root(),
)

_test_env = environment()
if host_machine.system() == 'linux'
  _test_env.prepend('LD_LIBRARY_PATH', meson.project_build_root())
elif host_machine.system() == 'darwin'
  _test_env.prepend('DYLD_LIBRARY_PATH', meson.project_build_root())
endif

test('wrap-pair-c', wrap_pair_c, env: _test_env)
test('wrap-pair-cpp', wrap_pair_cpp, env: _test_env)
