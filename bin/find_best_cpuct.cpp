#include <games/connect4.hpp>
#include <games/oware.hpp>
#include <games/tictactoe.hpp>
#include <mcts/pit_play_worker.hpp>
#include <training/utils.hpp>

YAML::Node config;
Game game;
std::string data_path;
int threads;
int pit_play_games;
MCTSConfig pit_play_config;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Invalid number of arguments!\n";
        std::cerr << "Usage ./main config_file\n";
        return 1;
    }

    config = YAML::LoadFile(argv[1]);

    std::cerr << config << "\n";

    if (config["game"]) {
        std::string game_name = config["game"].as<std::string>();
        std::cerr << "Loading game: " << game_name << "\n";

        if (game_name == "oware") {
            game = std::make_shared<OwareGame<Tensor>>();
        } else if (game_name == "tictactoe") {
            game = std::make_shared<TicTacToeGame<Tensor>>();
        } else if (game_name == "connect4") {
            game = std::make_shared<Connect4Game<Tensor>>();
        } else {
            std::cerr << "Unkown game: " << game_name << "\n";
            return 1;
        }
    } else {
        std::cerr << "Config require: game\n";
        return 1;
    }

    if (config["data_path"]) {
        data_path = config["data_path"].as<std::string>();
        std::cerr << "Loading data_path: " << data_path << "\n";
    } else {
        std::cerr << "Config require: data_path\n";
        return 1;
    }

    if (config["threads"]) {
        threads = config["threads"].as<int>();
        std::cerr << "Loading threads: " << threads << "\n";
    } else {
        std::cerr << "Config require: threads\n";
        return 1;
    }

    if (config["pit_play_games"]) {
        pit_play_games = config["pit_play_games"].as<int>();
        std::cerr << "Loading pit_play_games: " << pit_play_games << "\n";
    } else {
        std::cerr << "Config require: pit_play_games\n";
        return 1;
    }

    // if (config["pit_play_config"]) {
    //     std::cerr << "Loading pit_play_config\n";
    //     pit_play_config = parse_mcts_config(config["pit_play_config"]);
    // } else {
    //     std::cerr << "Config require: pit_play_config\n";
    //     return 1;
    // }

    if (config["model"]) {
        std::cerr << "Loading model\n";
        parse_model(config["model"]);
    } else {
        std::cerr << "Config require: model\n";
        return 1;
    }

    auto model_factory = [=]() {
        Model model = *parse_model(config["model"]);

        std::ifstream file(data_path + "/model_best", std::ios::binary);
        model.load(file);
        file.close();

        return model;
    };

    Model model = model_factory();

    game->calc_legal_moves();
    game->get_input_for_network(model.input_layer->get_output());
    model.forward(game);
    std::cerr << model.get_output() << "\n";
    std::cerr << "value:" << model.get_value() << "\n";
    for (int i = 0; i < game->get_maximum_number_of_moves(); i++) {
        std::cerr << i << ":" << model.get_policy(i) << "\n";
    }

    pit_play_config.number_of_iterations_per_turn = 1;
    MCTS mcts(game, pit_play_config);
    mcts.search(model);
    std::cerr << mcts.get_best() << "\n";

    // std::vector<float> cpucts{1, 2, 3, 4, 5, 6};

    // for (auto cpuctA : cpucts) {
    //     std::cerr << "CPUCTA: " << cpuctA << "\n";

    //     for (auto cpuctB : cpucts) {
    //         auto pit_play_configA = pit_play_config;
    //         pit_play_configA.cpuct_init = cpuctA;

    //         auto pit_play_configB = pit_play_config;
    //         pit_play_configB.cpuct_init = cpuctB;

    //         auto pit_play_worker = PitPlayWorker(
    //             game, model_factory, pit_play_configA, model_factory,
    //             pit_play_configB, 100, threads, false, 30);

    //         auto wr = pit_play_worker.get_first_player_winrate();

    //         std::cerr << "\tCPUTCTB:" << cpuctB << " wr: " << wr << "\n";
    //     }
    // }
}