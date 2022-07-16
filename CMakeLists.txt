cmake_minimum_required(VERSION 3.0.0)
project(WroclawZero VERSION 0.1.0)

# include(CTest)
# enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

# include(CPack)
 
set(CMAKE_CXX_FLAGS "-Ofast -std=c++17 -march=native -mbmi -mbmi2 -mavx2")
# -fsanitize=undefined -fsanitize=address -g -pg 

add_subdirectory(wroclaw_zero)
add_subdirectory(tests)
add_subdirectory(bin)


# CMake defines several variables for often-used paths
# CMAKE_SOURCE_DIR
# Contains the full path to the top level of the source tree, i.e. the location of the
# top-level CMakeLists.txt
# CMAKE_CURRENT_SOURCE_DIR
# Contains the full path the the source directory that is currently being processed
# by CMake. Differs from CMAKE_SOURCE_DIR in directories added through
# add_subdirectory.
# CMAKE_BINARY_DIR
# Contains the full path to the top level of the build tree, i.e. the build directory in
# which cmake is invoked.
# CMAKE_CURRENT_BINARY_DIR
# Contains the full path the binary directory that is currently being processed.
# Each directory added through add_subdirectory will create a corresponding
# binary directory in the build tree.