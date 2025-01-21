cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(LIVOX_MID360)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (LIVOX_MID360 LIVOX_MID360.cpp)
target_link_libraries (LIVOX_MID360 ${PCL_LIBRARIES})
