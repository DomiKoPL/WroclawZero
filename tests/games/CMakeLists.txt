add_executable(tictactoe tictactoe_test.cpp)
target_link_libraries(tictactoe PRIVATE games model)

add_executable(oware oware_test.cpp)
target_link_libraries(oware PRIVATE games model)

add_executable(connect4 connect4_test.cpp)
target_link_libraries(connect4 PRIVATE games model)