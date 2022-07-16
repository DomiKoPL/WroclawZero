#include <string>
#include <sstream>
#include <immintrin.h>
#include <model/model.hpp>
#include <mcts/MCTS.hpp>
#include <games/connect4.hpp>
#include <chrono>
#include <cstdint>

int main() {
    Stopwatch watch(100);

std::wstring w = L"No model for you :D";
float mult, norm_min;
std::string decoded = "";
for (auto c : w) {
    decoded += (char)(c >> 8);
    decoded += (char)(c & 255);
}

std::stringstream ss2(decoded), ss;
for (int i = 0; i < decoded.size(); i += 2) {
    char s1, s2;
    ss2.get(s1);
    ss2.get(s2);
    uint32_t s = (uint8_t)s2 + (((uint8_t)s1) << 8);
    float val = (s / mult) + norm_min;
    ss.write((char*)&val, 4);
}




    auto model = Model(
        std::make_shared<InputLayer>(std::vector<size_t>{1, 9, 7}),
        std::make_shared<Conv2DLayer>(16, 4, 4, 1, 1, 0, 0, activationRELU),
        std::make_shared<FlattenLayer>(),
        std::make_shared<LinearLayer>(64, activationRELU), 
        std::make_shared<LinearLayer>(64, activationRELU), 
        std::make_shared<LinearLayer>(11)
    );

    model.load(ss);

    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    MCTSConfig config;
    config.cpuct_init = 4;
    config.temperature_turns = 0;
    config.temperature_max = 2;
    config.temperature_min = 0.5;
    config.number_of_iterations_per_turn = 10;
    config.dirichlet_noise_alpha = 1.11;
    config.dirichlet_noise_epsilon = 0.05;
    config.init_reserved_nodes = 10'000'000;

    int my_id; // 0 or 1 (Player 0 plays first)
    int opp_id; // if your index is 0, this will be 1, and vice versa
    std::cin >> my_id >> opp_id; std::cin.ignore();

    auto game = std::make_shared<Connect4Game<Tensor>>();
    // game->read();

    Connect4Game<Tensor> temp;
    MCTS mcts(game, config);
    mcts.search(model);

    // mcts.debug_stats();

    int timeout = 900;
    while (true) {
        temp.read();

        int num_valid_actions; // number of unfilled columns in the board
        std::cin >> num_valid_actions; std::cin.ignore();
        for (int i = 0; i < num_valid_actions; i++) {
            int action; // a valid column index into which a chip can be dropped
            std::cin >> action; std::cin.ignore();
        }

        int opp_previous_action; // opponent's previous chosen column index (will be -1 for first player in the first turn)
        std::cin >> opp_previous_action; std::cin.ignore();

        if (opp_previous_action != -1) {
            if (opp_previous_action == -2) {
                mcts.restore_root(Connect4Game<Tensor>::WIDTH, model);
                game->make_move(Connect4Game<Tensor>::WIDTH);
            } else {
                mcts.restore_root(opp_previous_action, model);
                game->make_move(opp_previous_action);
            }
        }

        watch.start(timeout);

        std::cerr << *game << "\n";

        int iters = 0;
        while (not watch.timeout()) {
            mcts.search(model);
            iters += config.number_of_iterations_per_turn;
        }

        // mcts.debug_stats();
        
        auto [move, msg] = mcts.get_cg_best();
        // auto move = mcts.get_best();
        

        std::cerr << "ITERS: " << iters << "\n";
        // std::cerr << "HIT:" << model.cache_hit << " MISS:" << model.cache_miss << "\n";
        std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";
        // std::cerr << "SELECTED MOVE: " << move << " " << msg << "\n";
        
        
        game->make_move(move);
        std::cerr << "GAME\n";
        std::cerr << *game << "\n";
        mcts.restore_root(move, model);

        if (move == Connect4Game<Tensor>::WIDTH) {
            std::cout << "-2 " << msg << "\n";
        } else {
            std::cout << move << " " << msg << "\n";
        }

        std::cout.flush();

        timeout = 90;
    }
}