add_library(
    nn_avx_fast
    src/nn_avx_fast/activation_layers.cpp
)

target_include_directories(
    nn_avx_fast 
    INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)