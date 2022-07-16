#include <immintrin.h>

#include <chrono>
#include <cstdint>
#include <games/oware.hpp>
#include <mcts/MCTS.hpp>
#include <model/model.hpp>
#include <sstream>
#include <string>

int main() {
    Stopwatch watch(100);
    std::wstring w = L"No model for you";
    float mult = 0, norm_min = 0;

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
        int32_t s = (uint8_t)s2 + (((uint8_t)s1) << 8) - 255;
        float val = (s / mult) + norm_min;
        ss.write((char*)&val, 4);
    }
    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    Model model(std::make_shared<InputLayer>(std::vector<size_t>{342}),
                std::make_shared<LinearLayer>(..., ...),
                std::make_shared<LinearLayer>(7));

    model.load(ss);

    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    MCTSConfig config;
    config.cpuct_init = 4;
    config.number_of_iterations_per_turn = 10;
    config.dirichlet_noise_alpha = 1.5;
    config.dirichlet_noise_epsilon = 0.02;
    config.temperature_turns = 0;
    config.temperature_max = 1;
    config.temperature_min = 1;
    config.init_reserved_nodes = 10'000'000;

    auto game = std::make_shared<OwareGame<Tensor>>();

    // FIRST TURN
    game->read();
    watch.start(900);
    MCTS mcts(game, config);
    std::cerr << *game << "\n";

    while (not watch.timeout()) {
        mcts.search(model);
    }

    int move = mcts.get_best();
    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    mcts.debug_stats();
    game->make_move(move);
    mcts.restore_root(move, model);
    std::cout << move << "\n";

    while (true) {
        game->read();
        watch.start(45);
        mcts.restore_root(game, model);

        std::cerr << *game << "\n";

        int iters = 0;
        mcts.debug_stats();
        while (not watch.timeout()) {
            mcts.search(model);
            iters += config.number_of_iterations_per_turn;
        }

        mcts.debug_stats();
        mcts.debug_select();

        const int move = mcts.get_best();

        std::cerr << "ITERS: " << iters << "\n";
        // std::cerr << "HIT:" << model.cache_hit << " MISS:" <<
        // model.cache_miss << "\n";
        std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

        game->make_move(move);
        mcts.restore_root(move, model);
        std::cout << move << "\n";
    }
}