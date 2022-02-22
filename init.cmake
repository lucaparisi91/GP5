



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
