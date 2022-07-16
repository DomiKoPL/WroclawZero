#include <chrono>
#include <games/oware.hpp>
#include <mcts/self_play_worker.hpp>

int main() {

    MCTSConfig config;
    config.cpuct_init = 3;
    config.temperature_turns = 20;
    config.temperature_max = 0.75;
    config.temperature_min = 0.75;
    config.number_of_iterations_per_turn = 1600;
    config.dirichlet_noise_epsilon = 0.15;
    config.dirichlet_noise_alpha = 1.5;
    config.init_reserved_nodes = 1'000'000;

    auto game = std::make_shared<OwareGame<Tensor>>();

    ModelFactory factory = [=]() {
        Model model(std::make_shared<InputLayer>(std::vector<size_t>{342}),
                    std::make_shared<LinearLayer>(7));

        std::ifstream file(model_path, std::ios::binary);
        assert(file.is_open());
        model.load(file);
        file.close();
        return model;
    };

    auto start = std::chrono::high_resolution_clock::now();
    SelfPlayWorker worker(game, factory, config, factory, config, 1000, 8,
                          true);
    auto end = std::chrono::high_resolution_clock::now();

    std::cerr << "Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms\n";
}