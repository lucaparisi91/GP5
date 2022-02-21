
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


mark_as_advanced(
  HDF5_C_LIBRARY_z HDF5_C_LIBRARY_m HDF5_C_LIBRARY_hdf5 HDF5_C_LIBRARY_dl CMAKE_INSTALL_PREFIX  )

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." )
# Set the possible values of build type for cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")



############## GTEST  

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



function (set_library_path lib VAR )
  string(REPLACE ":" ";"  LOOKUP_PATHS "$ENV{LD_LIBRARY_PATH}" )
  message("${lib} ${LIB_FILE}" )

  find_library( LIB_FILE NAMES ${lib} REQUIRED TRUE PATHS ${LOOKUP_PATHS} )
  cmake_path(GET LIB_FILE PARENT_PATH LIB_PATH )
  set(${VAR} ${LIB_PATH} PARENT_SCOPE)
endfunction()


  



# optimization level
if ( ${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_CXX_COMPILE_FLAGS_ADDITIONAL " -g -pg -Wfatal-errors")
  set(CMAKE_CXX_LINK_FLAGS_ADDITIONAL " -g -pg ")
elseif (${CMAKE_BUILD_TYPE} MATCHES Release) 
  set(CMAKE_CXX_COMPILE_FLAGS_ADDITIONAL " -DNDEBUG -Wfatal-errors -O3 ")
  set(CMAKE_CXX_LINK_FLAGS_ADDITIONAL " ")
else()
  message(FATAL_ERROR "Unrecognized build type: " ${CMAKE_BUILD_TYPE}  )
endif()


SET(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_CXX_LINK_FLAGS_ADDITIONAL}" )

function(configure target)

  target_link_libraries(${target} PRIVATE p3dfft.3 )
  target_link_libraries(${target} PRIVATE stdc++fs)
  target_link_libraries(${target} PRIVATE hdf5)
  target_link_libraries(${target} PRIVATE fftw3)
  target_link_libraries(${target} PRIVATE fftw3f)
  target_link_libraries(${target} PRIVATE yaml-cpp)
  
  set( TARGET_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS_ADDITIONAL} -DHAVE_CONFIG_H   ")


  #message( " LIB: $ENV{LD_LIBRARY_PATH}" )

  set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
  
  string(REPLACE ":" " -L"  LIB_FLAGS "$ENV{LIBPATH}" )

  #set_target_properties(${target} PROPERTIES LINK_FLAGS "-L $ENV{LD_LIBRARY_PATH}" )
  target_link_libraries(${target} PRIVATE yaml-cpp)

  set_target_properties(${target} PROPERTIES LINK_FLAGS "-L ${LIB_FLAGS}" )
  #if(OpenMP_CXX_FOUND)
  #    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
  #endif()

endfunction()
