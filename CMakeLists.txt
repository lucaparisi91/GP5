
set(DIMENSIONS 3)
project(gp)

cmake_minimum_required(VERSION 3.20)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Compilation mode")
set_property(CACHE CMAKE_BUILD_TYPE  PROPERTY STRINGS Release Debug)
mark_as_advanced( CMAKE_INSTALL_PREFIX)

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELEASE "")


include(init.cmake)



add_subdirectory(pybind11)

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)


add_subdirectory(src)
add_subdirectory(tst)
add_subdirectory(petsc)
