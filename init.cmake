
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

function(configure target)

  target_link_libraries(${target} PRIVATE p3dfft.3 )
  target_link_libraries(${target} PRIVATE stdc++fs)
  target_link_libraries(${target} PRIVATE hdf5)
  target_link_libraries(${target} PRIVATE fftw3)
  target_link_libraries(${target} PRIVATE fftw3f)
  target_link_libraries(${target} PRIVATE yaml-cpp)
  
  set( TARGET_COMPILE_FLAGS " -DHAVE_CONFIG_H   ")

  set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
  
  string(REPLACE ":" " -L"  LIB_FLAGS "$ENV{LIBPATH}" )
  if ( NOT (LIB_FLAGS STREQUAL "") )
    set_target_properties(${target} PROPERTIES LINK_FLAGS "-L ${LIB_FLAGS}" )
  endif()

  #set_target_properties(${target} PROPERTIES LINK_FLAGS "-L $ENV{LD_LIBRARY_PATH}" )
  target_link_libraries(${target} PRIVATE yaml-cpp)

  set_target_properties(${target} PROPERTIES LINK_FLAGS "-L ${LIB_FLAGS}" )
  #if(OpenMP_CXX_FOUND)
  #    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
  #endif()

endfunction()

add_compile_options(
  -Wfatal-errors
       $<$<CONFIG:RELEASE>:-O3>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-g>
)

add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
        $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>
)