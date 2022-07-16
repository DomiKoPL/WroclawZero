#include <games/oware.hpp>
#include <mcts/dataset_validator.hpp>
#include <model/model.hpp>

int main() {
   

    std::string ww{ss.str()};

    auto factory = [=]() {
        Model model(std::make_shared<InputLayer>(std::vector<size_t>{342}),

        );
        std::stringstream ss{ww};
        model.load(ss);
        return model;
    };

    auto game = std::make_shared<OwareGame<Tensor>>();

    DatasetValidator validator(game, factory,
                               "oware-endgame/samples", true);
}