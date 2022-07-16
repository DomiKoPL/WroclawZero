add_library(
    model 
    INTERFACE
)

target_include_directories(
    model 
    INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(model INTERFACE nn_avx_fast games)