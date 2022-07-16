#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <games/connect4.hpp>
#include <games/oware.hpp>
#include <games/tictactoe.hpp>
#include <iostream>
#include <istream>
#include <iterator>
#include <mcts/dataset_validator.hpp>
#include <mcts/pit_play_worker.hpp>
#include <mcts/self_play_worker.hpp>
#include <mcts/validator_worker.hpp>
#include <sstream>
#include <streambuf>
#include <string>
#include <training/utils.hpp>
#include <unordered_map>
#include <vector>

constexpr char BEGIN_RED[] = "\033[31m";
constexpr char BEGIN_YELLOW[] = "\033[33m";
constexpr char BEGIN_BLUE[] = "\033[34m";
constexpr char COLOR_END[] = "\033[0m";

struct membuf : std::streambuf {
    membuf(char const* base, size_t size) {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }
};

struct imemstream : virtual membuf, std::istream {
    imemstream(char const* base, size_t size)
        : membuf(base, size),
          std::istream(static_cast<std::streambuf*>(this)) {}
};

Game game;
std::string data_path;
int threads;
int self_play_games;
int pit_play_games;
float win_rate_accepted;
MCTSConfig self_play_config;
MCTSConfig validation_config;
MCTSConfig pit_play_config;
std::string models_stats_path;
YAML::Node config;

void send_scalar(std::string tag, float val, int step) {
    int type = 2;
    std::cout.write(reinterpret_cast<char*>(&type), sizeof(type));
    int len = tag.size();
    std::cout.write(reinterpret_cast<char*>(&len), sizeof(len));
    std::cout.write(reinterpret_cast<char*>(tag.data()), len);
    std::cout.write(reinterpret_cast<char*>(&val), sizeof(val));
    std::cout.write(reinterpret_cast<char*>(&step), sizeof(step));
    std::cout.flush();
}

void send_scalars(std::string main_tag, std::vector<std::string> tags,
                  std::vector<float> vals, int step) {
    int type = 3;
    std::cout.write(reinterpret_cast<char*>(&type), sizeof(type));
    int len = main_tag.size();
    std::cout.write(reinterpret_cast<char*>(&len), sizeof(len));
    std::cout.write(reinterpret_cast<char*>(main_tag.data()), len);

    assert(tags.size() == vals.size());
    len = tags.size();
    std::cout.write(reinterpret_cast<char*>(&len), sizeof(len));

    for (int i = 0; i < tags.size(); i++) {
        len = tags[i].size();
        std::cout.write(reinterpret_cast<char*>(&len), sizeof(len));
        std::cout.write(reinterpret_cast<char*>(tags[i].data()), len);
        std::cout.write(reinterpret_cast<char*>(&vals[i]), sizeof(vals[i]));
    }

    std::cout.write(reinterpret_cast<char*>(&step), sizeof(step));
    std::cout.flush();
}

// https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
size_t hash_vector(std::vector<T>& in) {
    size_t size = in.size();
    size_t seed = size;
    for (size_t i = 0; i < size; i++) {
        hash_combine(seed, in[i]);
    }
    return seed;
}

size_t hash_tensor(const Tensor& tensor) {
    size_t size = tensor.size;
    size_t seed = size;
    for (size_t i = 0; i < size; i++) {
        hash_combine(seed, tensor.xmm[0].f[i]);
    }
    return seed;
}

void average_samples_scores(
    std::vector<std::unique_ptr<SelfPlayWorker>>& self_play_workers,
    int generation) {
    std::unordered_map<size_t, int> sample_cnt;
    std::unordered_map<size_t, float> sample_score;
    std::unordered_map<size_t, std::vector<float>> sample_policy;

    for (auto& self_play_worker : self_play_workers) {
        const auto& samples = self_play_worker->game_samples;

        for (int i = 0; i < self_play_worker->game_samples_count; i++) {
            const auto& sample = samples[i];
            auto hash = hash_tensor(sample.input);

            if (sample_cnt.count(hash)) {
                sample_cnt[hash]++;
                sample_score[hash] += sample.score;
                for (int i = 0; i < sample.policy.size(); i++) {
                    sample_policy[hash][i] += sample.policy[i];
                }
            } else {
                sample_cnt[hash] = 1;
                sample_score[hash] = sample.score;
                sample_policy[hash] = sample.policy;
            }
        }
    }
    float mse_loss_value = 0;
    float mse_loss_policy = 0;
    int samples_cnt = 0;

    for (auto& self_play_worker : self_play_workers) {
        auto& samples = self_play_worker->game_samples;
        samples_cnt += self_play_worker->game_samples_count;

        for (int i = 0; i < self_play_worker->game_samples_count; i++) {
            auto& sample = samples[i];
            auto hash = hash_tensor(sample.input);
            assert(sample_cnt.count(hash));
            assert(sample_cnt[hash] > 0);
            float avg = sample_score[hash] / sample_cnt[hash];
            mse_loss_value += (avg - sample.score) * (avg - sample.score);
            sample.score = avg;

            for (int j = 0; j < sample.policy.size(); j++) {
                float new_val = sample_policy[hash][j] / sample_cnt[hash];
                mse_loss_policy +=
                    (new_val - sample.policy[j]) * (new_val - sample.policy[j]);
                // sample.policy[j] = new_val;
            }
        }
    }

    mse_loss_value /= samples_cnt;
    mse_loss_policy /= samples_cnt;
    
    std::cerr << "[CPP] different samples: " << sample_cnt.size() << "\n";
    std::cerr << "[CPP] mse loss value in samples: " << mse_loss_value << "\n";
    std::cerr << "[CPP] mse loss policy in samples: " << mse_loss_policy
              << "\n";

    send_scalar("Self Play/MSE loss in samples", mse_loss_value, generation);
    send_scalars("Self Play", {"samples", "unique samples"},
                 {(float)samples_cnt, (float)sample_cnt.size()}, generation);
}

void self_play(int generation) {
    std::cerr << BEGIN_RED << "[CPP] Doing self play\n" << COLOR_END;

    YAML::Node models_stats = YAML::LoadFile(models_stats_path);

    bool new_model = (generation == models_stats["best1"].as<int>());

    std::vector<std::string> models_paths;
    for (auto model : models_stats) {
        auto model_generation = model.second.as<std::string>();
        models_paths.push_back(config["data_path"].as<std::string>() +
                               "/models/model_" + model_generation);
    }

    if (generation == 0) {
        models_paths.push_back(config["data_path"].as<std::string>() +
                               "/models/model_0");
    } else {
        models_paths.push_back(
            config["data_path"].as<std::string>() + "/models/model_" +
            std::to_string(Random::instance().next_int(
                std::max(0, generation - 5), generation - 1)));
    }

    std::cerr << "[CPP] Loaded " << models_paths.size()
              << " models for self play\n";

    for (auto model : models_paths) {
        std::cerr << "[CPP] " + model << " generation\n";
    }

    int all_games = self_play_games;

    // We don't need many games in first generations
    // This will speed up training
    if (generation <= 10) {
        all_games = self_play_games * (0.25 + (generation / 10.) * 0.75);
    }

    std::cerr << "[CPP] All_games = " << all_games << "\n";

    send_scalar("Self Play/Games", all_games, generation);
    // Divide games for each model
    // Best vs Best
    // Best vs BestX X = [2...]
    // Best vs Random from last 5 generations

    std::vector<float> games_percent(models_paths.size());
    // 10% games for Best vs Random
    games_percent.back() = 0.1;
    // x, 2x, 4x, ... for rest
    float x = 0.9f / (float)((1 << (games_percent.size() - 1)) - 1);
    // std::cerr << "X: " << x << "\n";

    for (int i = (int)games_percent.size() - 2; i >= 0; i--) {
        games_percent[i] = x;
        x *= 2;
    }

    std::vector<int> games(models_paths.size());
    for (int i = 0; i < games.size(); i++) {
        games[i] = games_percent[i] * all_games;
    }

    std::cerr << "[CPP] games=";
    for (auto g : games) std::cerr << g << ", ";
    std::cerr << "\n";

    for (int i = (int)models_paths.size() - 1; i >= 0; i--) {
        for (int j = 0; j < i; j++) {
            if (models_paths[i] == models_paths[j]) {
                games[j] += games[i];
                models_paths.erase(models_paths.begin() + i);
                games.erase(games.begin() + i);
                break;
            }
        }
    }

    std::cerr << "[CPP] games=";
    for (auto g : games) std::cerr << g << ", ";
    std::cerr << "\n";

    std::vector<std::unique_ptr<SelfPlayWorker>> self_play_workers;

    auto best_model_factory = [=]() {
        Model model = *parse_model(config["model"]);

        std::ifstream file(models_paths[0], std::ios::binary);
        model.load(file);
        file.close();

        return model;
    };

    for (int i = 0; i < models_paths.size(); i++) {
        auto model_factory = [=]() {
            Model model = *parse_model(config["model"]);

            std::ifstream file(models_paths[i], std::ios::binary);
            model.load(file);
            file.close();

            return model;
        };

        std::cerr << "[CPP] self play " << games[i] << " games against "
                  << models_paths[i] << "\n";

        self_play_workers.emplace_back(new SelfPlayWorker(
            game, model_factory, self_play_config, best_model_factory,
            self_play_config, games[i], threads, true));
    }

    int games_length = 0, games_played = 0, samples_cnt = 0;
    auto first_moves_vis =
        std::vector<float>(self_play_workers[0]->first_moves_vis.size(), 0);

    for (auto& self_play_worker : self_play_workers) {
        games_length += self_play_worker->games_length;
        games_played += self_play_worker->games_to_play;
        samples_cnt += self_play_worker->game_samples_count;

        for (int i = 0; i < first_moves_vis.size(); i++) {
            first_moves_vis[i] += self_play_worker->first_moves_vis[i];
        }
    }

    std::vector<std::string> tags(first_moves_vis.size());
    for (int i = 0; i < tags.size(); i++) {
        tags[i] = std::to_string(i);
    }

    //* normalize first moves stats
    for (auto& i : first_moves_vis) {
        i /= games_played;
    }

    send_scalars("First move counts", tags, first_moves_vis, generation);

    float avg_game_length = (float)games_length / (float)games_played;
    std::cerr << "[CPP] average game length: " << avg_game_length << "\n";
    send_scalar("Self Play/Average game length", avg_game_length, generation);

    std::cerr << "[CPP] Samples count: " << samples_cnt << "\n";

    average_samples_scores(self_play_workers, generation);

    int type = 4;
    std::cout.write(reinterpret_cast<char*>(&type), sizeof(type));
    int state_len = self_play_workers[0]->game_samples[0].input.size;
    int policy_len = self_play_workers[0]->game_samples[0].policy.size();

    std::cerr << "[CPP] state_len=" << state_len << "\n";
    std::cerr << "[CPP] policy_len=" << policy_len << "\n";

    std::cout.write(reinterpret_cast<char*>(&samples_cnt), sizeof(samples_cnt));
    std::cout.write(reinterpret_cast<char*>(&state_len), sizeof(state_len));
    std::cout.write(reinterpret_cast<char*>(&policy_len), sizeof(policy_len));

    for (auto& self_play_worker : self_play_workers) {
        const auto& samples = self_play_worker->game_samples;

        for (int i = 0; i < self_play_worker->game_samples_count; i++) {
            samples[i].input.save(std::cout);
        }
    }

    for (auto& self_play_worker : self_play_workers) {
        const auto& samples = self_play_worker->game_samples;

        for (int i = 0; i < self_play_worker->game_samples_count; i++) {
            std::cout.write(
                reinterpret_cast<const char*>(samples[i].legal_moves.data()),
                policy_len * sizeof(int));
        }
    }

    for (auto& self_play_worker : self_play_workers) {
        const auto& samples = self_play_worker->game_samples;

        for (int i = 0; i < self_play_worker->game_samples_count; i++) {
            std::cout.write(
                reinterpret_cast<const char*>(samples[i].policy.data()),
                policy_len * sizeof(float));
        }
    }

    for (auto& self_play_worker : self_play_workers) {
        const auto& samples = self_play_worker->game_samples;

        for (int i = 0; i < self_play_worker->game_samples_count; i++) {
            std::cout.write(reinterpret_cast<const char*>(&samples[i].score),
                            sizeof(float));
        }
    }

    std::cout.flush();

    std::cerr << BEGIN_BLUE << "[CPP] self play END\n" << COLOR_END;

    if (new_model and config["validators"]) {
        for (auto validator : config["validators"]) {
            if (not validator["type"]) {
                std::cerr << "Validator require: type.\n";
                exit(1);
            }

            auto type = validator["type"].as<std::string>();

            if (type == "dataset") {
                if (not validator["data_path"]) {
                    std::cerr << "Validator dataset require: data_path!\n";
                    exit(1);
                }

                auto path = validator["data_path"].as<std::string>();
                auto valid = DatasetValidator(game, best_model_factory, path);

                auto main_tag = "Validation/dataset " + path;
                send_scalars(
                    main_tag,
                    {"value_mse", "policy_mse", "good_sign", "optimal move"},
                    {valid.value_mse, valid.policy_mse, valid.good_sign,
                     valid.optimal_move},
                    generation);
                continue;
            }

            int games;
            if (validator["games"]) {
                games = validator["games"].as<int>();
                std::cerr << "Loading games: " << games << "\n";
            } else {
                std::cerr << "validator require: games\n";
                exit(1);
            }

            if (type == "RandomAgent") {
                auto valid_worker = ValidatorWorker<RandomAgent>(
                    game, best_model_factory, validation_config, games, threads,
                    true);

                send_scalar("Validation/RandomAgent WR",
                            valid_worker.get_win_rate(), generation);

                send_scalar("Validation/RandomAgent game length",
                            valid_worker.get_average_game_length(), generation);
            } else if (type == "Depth1Agent") {
                auto valid_worker = ValidatorWorker<Depth1Agent>(
                    game, best_model_factory, validation_config, games, threads,
                    true);

                send_scalar("Validation/Depth1Agent WR",
                            valid_worker.get_win_rate(), generation);

                send_scalar("Validation/Depth1Agent game length",
                            valid_worker.get_average_game_length(), generation);
            } else if (type == "model") {
                if (not validator["data_path"]) {
                    std::cerr << "Validator model require: data_path.\n";
                    exit(1);
                }

                auto val_data_path = validator["data_path"].as<std::string>();

                auto val_model_factory = [=]() {
                    Model model = *parse_model(validator["model"]);

                    std::ifstream file(
                        validator["data_path"].as<std::string>() +
                            "/model_best",
                        std::ios::binary);
                    model.load(file);
                    file.close();

                    return model;
                };

                auto val_config = parse_mcts_config(validator["config"]);

                auto pit_play_worker = PitPlayWorker(
                    game, best_model_factory, pit_play_config,
                    val_model_factory, val_config, games, threads, true);

                std::clog << "Win rate against validator: "
                          << pit_play_worker.get_first_player_winrate() << "\n";

                send_scalar("Validation/" + val_data_path + " WR",
                            pit_play_worker.get_first_player_winrate(),
                            generation);

                send_scalar("Validation/" + val_data_path + " game length",
                            pit_play_worker.get_average_game_length(),
                            generation);
            } else {
                std::cerr << "Invalid validator type: " << type << "\n";
                exit(1);
            }
        }
    }
}

void comparision(int generation) {
    std::cerr << BEGIN_YELLOW << "[CPP] Doing comparison play\n" << COLOR_END;

    int model_bytes;
    std::cin.read(reinterpret_cast<char*>(&model_bytes), sizeof(model_bytes));
    std::vector<char> candidate_model_bytes(model_bytes);
    std::cin.read(reinterpret_cast<char*>(candidate_model_bytes.data()),
                  model_bytes);

    auto candidate_model_factory = [=]() {
        std::stringstream ss;

        imemstream stream(
            reinterpret_cast<const char*>(candidate_model_bytes.data()),
            candidate_model_bytes.size());

        Model model = *parse_model(config["model"]);

        model.load(stream);

        return model;
    };

    YAML::Node models_stats = YAML::LoadFile(models_stats_path);

    float candidate_win_ratio = 100;

    for (auto best_model : {"best1", "best2"}) {
        if (not models_stats[best_model]) {
            continue;
        }

        auto best_model_generation = models_stats[best_model].as<std::string>();

        std::cerr << "[CPP] " << best_model
                  << " generation:" << best_model_generation << "\n";

        std::string best_model_path = config["data_path"].as<std::string>() +
                                      "/models/model_" + best_model_generation;

        auto best_model_factory = [=]() {
            Model model = *parse_model(config["model"]);

            std::ifstream file(best_model_path, std::ios::binary);
            model.load(file);
            file.close();

            return model;
        };

        auto pit_play_worker = PitPlayWorker(
            game, candidate_model_factory, pit_play_config, best_model_factory,
            pit_play_config, pit_play_games, threads, true);

        std::clog << "[CPP] Candidate model winratio: "
                  << pit_play_worker.get_first_player_winrate() << "\n";

        auto wr = pit_play_worker.get_first_player_winrate();

        candidate_win_ratio = std::min(candidate_win_ratio, wr);
    }

    send_scalar("Comparision/win rate", candidate_win_ratio, generation);

    if (candidate_win_ratio >= win_rate_accepted) {
        models_stats["best2"] = models_stats["best1"].as<std::string>();
        models_stats["best1"] = generation;

        std::ofstream file(models_stats_path);
        file << models_stats;
        file.close();

        std::ofstream model_file(config["data_path"].as<std::string>() +
                                 "/model_best");
        model_file.write(
            reinterpret_cast<const char*>(candidate_model_bytes.data()),
            model_bytes);
        model_file.close();
    }

    std::clog << BEGIN_YELLOW << "Comparison end\n" << COLOR_END;

    int type = 1;
    std::cout.write(reinterpret_cast<char*>(&type), sizeof(type));
    std::cout.flush();
}

void train_loop() {
    while (true) {
        int type;
        std::cin.read(reinterpret_cast<char*>(&type), sizeof(type));
        std::cerr << "[CPP] Type:" << type << "\n";

        if (type == 0) {
            int generation;
            std::cin.read(reinterpret_cast<char*>(&generation),
                          sizeof(generation));
            std::cerr << "[CPP] Requested self play. Generation: " << generation
                      << "\n";

            self_play(generation);
        } else if (type == 1) {
            int generation;
            std::cin.read(reinterpret_cast<char*>(&generation),
                          sizeof(generation));
            std::cerr << "[CPP] Requested model comparison. Generation: "
                      << generation << "\n";

            std::this_thread::sleep_for(std::chrono::seconds(2));

            comparision(generation);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
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

    if (config["self_play_games"]) {
        self_play_games = config["self_play_games"].as<int>();
        std::cerr << "Loading self_play_games: " << self_play_games << "\n";
    } else {
        std::cerr << "Config require: self_play_games\n";
        return 1;
    }

    if (config["pit_play_games"]) {
        pit_play_games = config["pit_play_games"].as<int>();
        std::cerr << "Loading pit_play_games: " << pit_play_games << "\n";
    } else {
        std::cerr << "Config require: pit_play_games\n";
        return 1;
    }

    if (config["win_rate_accepted"]) {
        win_rate_accepted = config["win_rate_accepted"].as<float>();
        std::cerr << "Loading win_rate_accepted: " << win_rate_accepted << "\n";
    } else {
        std::cerr << "Config require: win_rate_accepted\n";
        return 1;
    }

    if (config["self_play_config"]) {
        std::cerr << "Loading self_play_config\n";
        self_play_config = parse_mcts_config(config["self_play_config"]);
    } else {
        std::cerr << "Config require: self_play_config\n";
        return 1;
    }

    if (config["validation_config"]) {
        std::cerr << "Loading validation_config\n";
        validation_config = parse_mcts_config(config["validation_config"]);
    } else {
        std::cerr << "Config require: validation_config\n";
        return 1;
    }

    if (config["pit_play_config"]) {
        std::cerr << "Loading pit_play_config\n";
        pit_play_config = parse_mcts_config(config["pit_play_config"]);
    } else {
        std::cerr << "Config require: pit_play_config\n";
        return 1;
    }

    if (config["model"]) {
        std::cerr << "Loading model\n";
        parse_model(config["model"]);
    } else {
        std::cerr << "Config require: model\n";
        return 1;
    }

    models_stats_path = std::string(data_path + "/models_stats.yaml");
    train_loop();
}