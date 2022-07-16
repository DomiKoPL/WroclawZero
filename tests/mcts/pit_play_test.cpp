#include <mcts/pit_play_worker.hpp>
#include <games/tictactoe.hpp>

int main() {
    auto game = std::make_shared<TicTacToeGame<Tensor>>();

    MCTSConfig config;
    config.temperature_turns = 0;
    config.number_of_iterations_per_turn = 1;

    ModelFactory factory = []() {
        Model model(std::make_shared<InputLayer>(std::vector<size_t>{18}),
                    std::make_shared<LinearLayer>(64, activationTANH),
                    std::make_shared<LinearLayer>(64, activationTANH),
                    std::make_shared<LinearLayer>(10));

        std::fstream file("models/model.weights", std::fstream::in);
        model.load(file);
        return model;
    };

    MCTSConfig config2;
    config2.temperature_turns = 0;
    config2.number_of_iterations_per_turn = 50;

    PitPlayWorker worker(game, factory, config, factory, config2, 1'000'000, 8, true);
}