add_library(
    mcts 
    INTERFACE
)

target_include_directories(
    mcts 
    INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(mcts INTERFACE games model training)