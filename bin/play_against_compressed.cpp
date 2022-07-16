#include <games/connect4.hpp>
#include <games/oware.hpp>
#include <games/tictactoe.hpp>
#include <mcts/pit_play_worker.hpp>
#include <training/utils.hpp>

YAML::Node config;
Game game;
std::string data_path;
int threads;
MCTSConfig self_play_config;

std::string read_file(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

std::vector<float> to_float(std::string s) {
    int size = s.size() / sizeof(float);
    std::vector<float> res(size, 0);
    std::stringstream ss{s};

    ss.read(reinterpret_cast<char*>(res.data()), s.size());

    return res;
}

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

    if (config["self_play_config"]) {
        std::cerr << "Loading self_play_config\n";
        self_play_config = parse_mcts_config(config["self_play_config"]);
    } else {
        std::cerr << "Config require: self_play_config\n";
        return 1;
    }

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

    auto w32 = read_file(data_path + "/model_best");
    auto a = to_float(w32);

    float max = *std::max_element(a.begin(), a.end());
    float min = *std::min_element(a.begin(), a.end());

    std::cerr << "MAX:" << max << "\n";
    std::cerr << "MIN:" << min << "\n";

    float norm_min = std::floor(min * 1000) / 1000.f;
    float norm_max = std::ceil(max * 1000) / 1000.f;
    std::cerr << "NORM: " << norm_min << " " << norm_max << "\n";

    float mult = 55'000 / (norm_max - norm_min);
    std::cerr << "MULT: " << mult << "\n";

    std::stringstream ss;

    float max_diff = 0;
    for (auto f : a) {
        int32_t s = (int32_t)std::round(mult * (f - norm_min));
        float val = (s / mult) + norm_min;
        ss.write((char*)&val, 4);
    }

    auto w = ss.str();

    auto compressed_model_factory = [=]() {
        Model model = *parse_model(config["model"]);

        std::stringstream ss{w};
        model.load(ss);

        return model;
    };

    auto temp = self_play_config;

    auto pit_play_worker =
        PitPlayWorker(game, compressed_model_factory, temp,
                      model_factory, self_play_config, 500, threads, true, 30);

    auto wr = pit_play_worker.get_first_player_winrate();

    std::cerr << "WR: " << wr << "\n";
}