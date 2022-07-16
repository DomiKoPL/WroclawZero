add_library(
    games 
    INTERFACE
)

target_include_directories(
    games 
    INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)