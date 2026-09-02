cmake_minimum_required(VERSION 3.22)
project(minimage-consumer LANGUAGES CXX)

find_package(minimage 0.1 REQUIRED CONFIG)
if(NOT TARGET minimage::minimage)
  message(FATAL_ERROR "find_package(minimage) did not define minimage::minimage")
endif()

add_executable(find_mi "${MINIMAGE_EXAMPLE}")
target_link_libraries(find_mi PRIVATE minimage::minimage)
