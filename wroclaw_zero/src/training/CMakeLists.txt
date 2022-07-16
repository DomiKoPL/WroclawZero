add_library(
    training 
    INTERFACE
)

target_include_directories(
    training 
    INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

find_package(yaml-cpp REQUIRED)
target_link_libraries(training INTERFACE mcts yaml-cpp)