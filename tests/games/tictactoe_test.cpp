#include <games/tictactoe.hpp>
#include <model/model.hpp>

int main() {
    TicTacToeGame<Tensor> game;

    srand(time(0));

    while (not game.is_terminal()) {
        game.calc_legal_moves();
        auto move = game.legal_moves[rand() % game.legal_moves_cnt];
        game.make_move(move);
        std::cerr << game << "\n";
        std::cerr << "---\n";
    }

    std::cerr << game.get_game_result() << "\n";
}