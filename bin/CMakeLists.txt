add_executable(main main.cpp)
target_link_libraries(main PRIVATE training)

add_executable(find_best_cpuct find_best_cpuct.cpp)
target_link_libraries(find_best_cpuct PRIVATE training)

add_executable(play_against_compressed play_against_compressed.cpp)
target_link_libraries(play_against_compressed PRIVATE training)

add_executable(compress_weights compress_weights.cpp)

add_custom_command(
    TARGET main POST_BUILD
    COMMAND cp --verbose -r ${CMAKE_SOURCE_DIR}/wroclaw_zero/src/python_training ${CMAKE_CURRENT_BINARY_DIR}
)