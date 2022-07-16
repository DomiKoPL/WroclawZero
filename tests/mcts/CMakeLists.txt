add_executable(selfplay_test selfplay_test.cpp)
target_link_libraries(selfplay_test PRIVATE mcts games model)

add_executable(selfplay_connect4_test selfplay_connect4_test.cpp)
target_link_libraries(selfplay_connect4_test PRIVATE mcts games model)

add_executable(pit_play_test pit_play_test.cpp)
target_link_libraries(pit_play_test PRIVATE mcts games model)

add_executable(dataset_test dataset_test.cpp)
target_link_libraries(dataset_test PRIVATE mcts games model)