
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(../init.cmake)

add_executable( gs ex1.cpp model.cpp )

add_executable( eigenGP eigenGP.cpp model.cpp)


#target_link_libraries(gpPetsc PUBLIC gp_lib)
target_link_libraries(gs PUBLIC petsc)
target_link_libraries(eigenGP PUBLIC petsc slepc)


configure(eigenGP)
configure(gs)