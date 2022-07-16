#pragma once

#include <yaml-cpp/yaml.h>

#include <mcts/MCTS_config.hpp>
#include <model/model.hpp>

std::shared_ptr<Model> parse_model(const YAML::Node& config) {
    std::shared_ptr<Model> model;

    for (auto layer : config) {
        if (not layer["type"]) {
            std::cerr << "Layer require type.\n";
            exit(1);
        }

        auto type = layer["type"].as<std::string>();
        std::string activation_str =
            (layer["activation"] ? layer["activation"].as<std::string>()
                                 : "NONE");

        Activation activation = activationNONE;
        if (activation_str == "ReLU")
            activation = activationRELU;
        else if (activation_str == "Sigmoid")
            activation = activationSIGMOID;
        else if (activation_str == "Softmax")
            activation = activationSOFTMAX;
        else if (activation_str == "Tanh")
            activation = activationTANH;

        if (type == "Linear") {
            const size_t inputs = layer["input"].as<size_t>();
            const size_t outputs = layer["output"].as<size_t>();

            if (model == nullptr) {
                model = std::make_shared<Model>(
                    std::make_shared<InputLayer>(std::vector<size_t>{inputs}),
                    std::make_shared<LinearLayer>(outputs, activation));
            } else {
                model->append(
                    std::make_shared<LinearLayer>(outputs, activation));
            }
        } else if (type == "Input") {
            assert(model == nullptr);
            auto shape = layer["shape"].as<std::vector<size_t>>();

            model =
                std::make_shared<Model>(std::make_shared<InputLayer>(shape));
        } else if (type == "Conv2d") {
            auto get = [&](std::string name) -> std::pair<int, int> {
                if (layer[name].IsSequence()) {
                    auto vals = layer[name].as<std::vector<int>>();
                    if (vals.size() == 1) {
                        return {vals[0], vals[0]};
                    } else if (vals.size() == 2) {
                        return {vals[0], vals[1]};
                    }

                    std::cerr << "Invalid numer or parameters in " << name
                              << "\n";
                    exit(1);
                }

                int val = layer[name].as<int>();
                return {val, val};
            };

            int out_channels = layer["out_channels"].as<int>();
            auto kernel = get("kernel");
            auto stride = get("stride");
            auto padding = get("padding");

            model->append(std::make_shared<Conv2DLayer>(
                out_channels, kernel.first, kernel.second, stride.first,
                stride.second, padding.first, padding.second, activation));
        } else {
            std::cerr << "TYPE:" << type << "\n";
            assert(0 and "Invalid layer type");
        }
    }

    // std::cerr << "Model params: " << model->count_params() << "\n";

    return model;
}

MCTSConfig parse_mcts_config(YAML::Node config) {
    MCTSConfig mcts_config{};

    if (config["cpuct_init"]) {
        mcts_config.cpuct_init = config["cpuct_init"].as<float>();
    }

    if (config["dirichlet_noise_epsilon"]) {
        mcts_config.dirichlet_noise_epsilon =
            config["dirichlet_noise_epsilon"].as<float>();
    }

    if (config["dirichlet_noise_alpha"]) {
        mcts_config.dirichlet_noise_alpha =
            config["dirichlet_noise_alpha"].as<float>();
    }

    if (config["number_of_iterations_per_turn"]) {
        mcts_config.number_of_iterations_per_turn =
            config["number_of_iterations_per_turn"].as<int>();
    }

    if (config["temperature_turns"]) {
        mcts_config.temperature_turns = config["temperature_turns"].as<int>();
    }

    if (config["temperature_max"]) {
        mcts_config.temperature_max = config["temperature_max"].as<float>();
    }

    if (config["temperature_min"]) {
        mcts_config.temperature_min = config["temperature_min"].as<float>();
    }

    if (config["init_reserved_nodes"]) {
        mcts_config.init_reserved_nodes =
            config["init_reserved_nodes"].as<int>();
    }

    return mcts_config;
}