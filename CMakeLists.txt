
set(DIMENSIONS 3)
project(gp)


mark_as_advanced(
  HDF5_C_LIBRARY_z HDF5_C_LIBRARY_m HDF5_C_LIBRARY_hdf5 HDF5_C_LIBRARY_dl CMAKE_INSTALL_PREFIX  )

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." )
# Set the possible values of build type for cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")


# optimization level
if ( ${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_CXX_COMPILE_FLAGS_ADDITIONAL " -g -pg -O0 -Wfatal-errors")
  set(CMAKE_CXX_LINK_FLAGS_ADDITIONAL " -g -pg ")
elseif (${CMAKE_BUILD_TYPE} MATCHES Release) 
  set(CMAKE_CXX_COMPILE_FLAGS_ADDITIONAL " -DNDEBUG -Wfatal-errors -O3 ")
  set(CMAKE_CXX_LINK_FLAGS_ADDITIONAL " ")
else()
  message(FATAL_ERROR "Unrecognized build type: " ${CMAKE_BUILD_TYPE}  )
endif()


SET(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_CXX_LINK_FLAGS_ADDITIONAL}" )



cmake_minimum_required(VERSION 3.20)

include(init.cmake)


add_subdirectory(src)
add_subdirectory(tst)






