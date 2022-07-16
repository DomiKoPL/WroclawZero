#include <games/connect4.hpp>
#include <mcts/self_play_worker.hpp>

int main() {
    auto game = std::make_shared<Connect4Game<Tensor>>();

    MCTSConfig config;
    config.cpuct_init = 4;
    config.temperature_turns = 12;
    config.temperature_max = 2;
    config.temperature_min = 0.5;
    config.number_of_iterations_per_turn = 800;
    config.dirichlet_noise_epsilon = 0.25;
    config.dirichlet_noise_alpha = 1.66;
    config.init_reserved_nodes = 10'000'000;

    ModelFactory factory = [=]() {
        Model model(std::make_shared<InputLayer>(std::vector<size_t>{2, 9, 7}),
                    std::make_shared<Conv2DLayer>(16, 4, 4, 1, 1, 2, 2, activationRELU),
                    std::make_shared<Conv2DLayer>(16, 2, 2, 1, 1, 1, 1, activationRELU),
                    std::make_shared<Conv2DLayer>(16, 2, 2, 1, 1, 1, 1, activationRELU),
                    std::make_shared<FlattenLayer>(),
                    std::make_shared<LinearLayer>(1920, activationRELU),
                    std::make_shared<LinearLayer>(11));

        std::ifstream is("", std::ios::binary);
        model.load(is);
        is.close();

        return model;
    };

    SelfPlayWorker worker(game, factory, config, factory, config, 1, 1, false);
}