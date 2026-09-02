cmake_minimum_required(VERSION 3.22)

# Version source of truth: Cargo.toml [package].version.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" _minimage_cargo_toml)
set(MINIMAGE_VERSION "")
foreach(_line ${_minimage_cargo_toml})
  if(_line MATCHES "^version = \"([0-9]+\\.[0-9]+\\.[0-9]+.*)\"")
    set(MINIMAGE_VERSION "${CMAKE_MATCH_1}")
    break()
  endif()
endforeach()
if(MINIMAGE_VERSION STREQUAL "")
  message(FATAL_ERROR "could not parse version from ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml")
endif()
unset(_minimage_cargo_toml)
unset(_line)

string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3"
  MINIMAGE_CMAKE_VERSION "${MINIMAGE_VERSION}")
project(minimage VERSION ${MINIMAGE_CMAKE_VERSION} LANGUAGES C CXX)
set(PROJECT_VERSION ${MINIMAGE_VERSION})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

option(MINIMAGE_BUILD_EXAMPLES "Build C and C++ example programs" ON)
option(MINIMAGE_INSTALL "Install headers, libraries, pkg-config, and CMake package" ON)

if(WIN32)
  set(MINIMAGE_SHARED_NAME minimage.dll)
  set(MINIMAGE_IMPLIB_NAME minimage.dll.lib)
  set(MINIMAGE_STATIC_NAME minimage.lib)
else()
  set(MINIMAGE_SHARED_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}minimage${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(MINIMAGE_STATIC_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}minimage${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

set(MINIMAGE_CARGO_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
string(TOLOWER "${CMAKE_BUILD_TYPE}" _minimage_build_type)
if(_minimage_build_type STREQUAL "debug")
  set(MINIMAGE_CARGO_PROFILE debug)
  set(MINIMAGE_CARGO_ARGS)
else()
  set(MINIMAGE_CARGO_PROFILE release)
  set(MINIMAGE_CARGO_ARGS --release)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock")
  list(APPEND MINIMAGE_CARGO_ARGS --locked)
endif()
set(MINIMAGE_CARGO_OUT "${MINIMAGE_CARGO_DIR}/${MINIMAGE_CARGO_PROFILE}")
set(MINIMAGE_SHARED "${CMAKE_CURRENT_BINARY_DIR}/${MINIMAGE_SHARED_NAME}")
set(MINIMAGE_STATIC "${CMAKE_CURRENT_BINARY_DIR}/${MINIMAGE_STATIC_NAME}")
if(WIN32)
  set(MINIMAGE_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/${MINIMAGE_IMPLIB_NAME}")
endif()

file(GLOB_RECURSE MINIMAGE_RUST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs")
list(APPEND MINIMAGE_RUST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock")
  list(APPEND MINIMAGE_RUST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock")
endif()

set(MINIMAGE_RUSTC_LINK)
if(APPLE)
  set(MINIMAGE_RUSTC_LINK "-Clink-arg=-Wl,-install_name,@rpath/${MINIMAGE_SHARED_NAME}")
elseif(UNIX)
  set(MINIMAGE_RUSTC_LINK "-Clink-arg=-Wl,-soname,${MINIMAGE_SHARED_NAME}")
endif()

set(MINIMAGE_CARGO_OUTPUTS "${MINIMAGE_SHARED}" "${MINIMAGE_STATIC}")
set(MINIMAGE_COPY_IMPLIB)
if(WIN32)
  list(APPEND MINIMAGE_CARGO_OUTPUTS "${MINIMAGE_IMPLIB}")
  set(MINIMAGE_COPY_IMPLIB
    COMMAND "${CMAKE_COMMAND}" -E copy
            "${MINIMAGE_CARGO_OUT}/${MINIMAGE_IMPLIB_NAME}"
            "${MINIMAGE_IMPLIB}")
endif()

add_custom_command(
  OUTPUT ${MINIMAGE_CARGO_OUTPUTS}
  COMMAND "${CARGO_EXECUTABLE}" rustc --lib ${MINIMAGE_CARGO_ARGS}
          --features capi
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
          --target-dir "${MINIMAGE_CARGO_DIR}"
          -- --crate-type=cdylib --crate-type=staticlib ${MINIMAGE_RUSTC_LINK}
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${MINIMAGE_CARGO_OUT}/${MINIMAGE_SHARED_NAME}"
          "${MINIMAGE_SHARED}"
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${MINIMAGE_CARGO_OUT}/${MINIMAGE_STATIC_NAME}"
          "${MINIMAGE_STATIC}"
  ${MINIMAGE_COPY_IMPLIB}
  DEPENDS ${MINIMAGE_RUST_SOURCES}
  COMMENT "Building libminimage via cargo rustc (cdylib + staticlib)"
  VERBATIM
)
add_custom_target(minimage_cargo ALL DEPENDS ${MINIMAGE_CARGO_OUTPUTS})

add_library(minimage SHARED IMPORTED GLOBAL)
add_library(minimage::minimage ALIAS minimage)
add_dependencies(minimage minimage_cargo)
set_target_properties(minimage PROPERTIES
  IMPORTED_LOCATION "${MINIMAGE_SHARED}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if(WIN32)
  set_target_properties(minimage PROPERTIES IMPORTED_IMPLIB "${MINIMAGE_IMPLIB}")
endif()

set(_minimage_sys Threads::Threads)
if(UNIX AND NOT APPLE)
  list(APPEND _minimage_sys ${CMAKE_DL_LIBS} m)
elseif(APPLE)
  list(APPEND _minimage_sys m)
elseif(WIN32)
  list(APPEND _minimage_sys ws2_32 userenv bcrypt ntdll)
endif()

add_library(minimage_static STATIC IMPORTED GLOBAL)
add_library(minimage::static ALIAS minimage_static)
add_dependencies(minimage_static minimage_cargo)
set_target_properties(minimage_static PROPERTIES
  IMPORTED_LOCATION "${MINIMAGE_STATIC}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
  INTERFACE_LINK_LIBRARIES "${_minimage_sys}"
)

if(MINIMAGE_BUILD_EXAMPLES)
  add_executable(wrap_pair examples/wrap_pair.c)
  target_link_libraries(wrap_pair PRIVATE minimage::minimage)
  set_target_properties(wrap_pair PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
  add_executable(wrap_pair_cpp examples/wrap_pair.cpp)
  target_link_libraries(wrap_pair_cpp PRIVATE minimage::minimage)
  set_target_properties(wrap_pair_cpp PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
  enable_testing()
  add_test(NAME wrap-pair-c COMMAND wrap_pair)
  add_test(NAME wrap-pair-cpp COMMAND wrap_pair_cpp)
  if(UNIX AND NOT APPLE)
    set_tests_properties(wrap-pair-c wrap-pair-cpp PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}")
  elseif(APPLE)
    set_tests_properties(wrap-pair-c wrap-pair-cpp PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endif()

if(MINIMAGE_INSTALL)
  if(UNIX AND NOT APPLE)
    set(MINIMAGE_PKG_LIBS_PRIVATE "-ldl -lpthread -lm")
  elseif(APPLE)
    set(MINIMAGE_PKG_LIBS_PRIVATE "-lpthread -lm")
  else()
    set(MINIMAGE_PKG_LIBS_PRIVATE "")
  endif()

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/minimage.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/minimage.pc"
    @ONLY
  )
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/minimageConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/minimageConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/minimage"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/minimageConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  if(WIN32)
    install(FILES "${MINIMAGE_SHARED}" DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(FILES "${MINIMAGE_IMPLIB}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  else()
    install(FILES "${MINIMAGE_SHARED}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(FILES "${MINIMAGE_STATIC}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/minimage.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/minimage.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/minimage.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/minimageConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/minimageConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/minimage"
  )
endif()
