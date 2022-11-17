
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(../init.cmake)
add_library(gp_lib geometry.cpp io.cpp tools.cpp fourierTransform.cpp operators.cpp functional.cpp stepper.cpp externalPotential.cpp timers.cpp constraint.cpp cluster.cpp  )
configure(gp_lib)

add_executable(gp main.cpp)
target_link_libraries(gp PUBLIC gp_lib)
configure(gp)



pybind11_add_module( gpCpp gpCpp.cpp pyTools.cpp )
target_link_libraries(gpCpp PUBLIC gp_lib)
configure(gpCpp)




