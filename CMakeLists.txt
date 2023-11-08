cmake_minimum_required(VERSION 3.26)
project(parallel_quicksort)

set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)

add_executable(parallel_quicksort
        hyperquicksort.cpp
)

target_link_libraries(parallel_quicksort PRIVATE MPI::MPI_CXX)
