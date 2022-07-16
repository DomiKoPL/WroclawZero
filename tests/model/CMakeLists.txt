add_executable(model_test model_test.cpp)
target_link_libraries(model_test PRIVATE model)

add_executable(oware_test oware_test.cpp)
target_link_libraries(oware_test PRIVATE model mcts games)

add_executable(connect4_test connect4_test.cpp)
target_link_libraries(connect4_test PRIVATE model mcts games)

add_executable(tictactoe_test tictactoe_test.cpp)
target_link_libraries(tictactoe_test PRIVATE model mcts games)