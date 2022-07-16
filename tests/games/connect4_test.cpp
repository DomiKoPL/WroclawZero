#include <games/connect4.hpp>
#include <model/model.hpp>

int main() {
    Connect4Game<Tensor> game;

    // std::cerr << game.get_input_for_network() << "\n";

    // game.make_move(1);
    // game.make_move(2);
    // game.make_move(5);
    // game.make_move(2);
    // game.make_move(5);
    // game.make_move(1);
    // game.make_move(2);
    // game.make_move(1);
    // game.make_move(2);
    // game.make_move(1);

    // game.make_move(8);
    // game.make_move(1);
    // game.make_move(6);
    // game.make_move(6);
    // game.make_move(6);
    // game.make_move(5);
    // game.make_move(5);
    // game.make_move(5);
    // game.make_move(5);

    // game.make_move(2);
    // game.make_move(7);

    game.read();

    game.debug();

    auto debug_mask = [](uint64_t mask) {
        constexpr int WIDTH = 9;
        constexpr int HEIGHT = 7;

        for (int y = HEIGHT - 1; y >= 0; y--) {
            for (int x = 0; x < WIDTH; x++) {
                const uint64_t m = 1ULL << (x * HEIGHT + y);

                if (mask & m) {
                    std::cerr << "X";
                } else {
                    std::cerr << ".";
                }
            }

            std::cerr << '\n';
        }

        std::cerr << '\n';
    };

    debug_mask(game.possible());
    debug_mask(game.compute_my_winning());
    debug_mask(game.compute_opponent_winning());

    game.calc_legal_moves();

    std::cerr << "legal\n";
    for (int i = 0; i < game.legal_moves_cnt; i++) {
        const int move = game.legal_moves[i];
        auto temp = game;
        temp.make_move(move);
        std::cerr << move << ", ";
    }

    std::cerr << "\n";

    exit(0);

    // game.read();

    while (not game.is_terminal()) {
        std::cerr << "GAME STATE\n";
        std::cerr << game << "\n";

        std::cerr << "JAZDA\n";
        game.calc_legal_moves();
        std::vector<float> val;

        std::cerr << "legal\n";
        for (int i = 0; i < game.legal_moves_cnt; i++) {
            const int move = game.legal_moves[i];
            auto temp = game;
            temp.make_move(move);
            if (temp.is_terminal()) {
                val.push_back(1);
            } else {
                val.push_back(0);
            }
            std::cerr << move << ", ";
        }
        std::cerr << "\n";

        int best = std::max_element(val.begin(), val.end()) - val.begin();
        int move = game.legal_moves[best];
        game.make_move(move);

        // std::cerr << game << "\n";
        std::cerr << "CHOSEN MOVE:";
        std::cout << move << "\n";

        static int cnt = 0;
        cnt++;
        // if (cnt == 20) break;
    }

    std::cerr << game << "\n";
    Tensor tensor(game.get_input_shape());
    game.get_input_for_network(tensor);

    std::cerr << tensor << "\n";
}