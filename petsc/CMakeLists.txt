
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(../init.cmake)

add_executable( gs ex1.cpp )

#target_link_libraries(gpPetsc PUBLIC gp_lib)
target_link_libraries(gpPetsc PUBLIC petsc)
configure(gpPetsc)