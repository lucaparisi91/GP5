
set(DIMENSIONS 3)

project(gp)
cmake_minimum_required(VERSION 3.20)
include(init.cmake)

add_executable(gp main.cpp geometry.cpp)
configure(gp)





