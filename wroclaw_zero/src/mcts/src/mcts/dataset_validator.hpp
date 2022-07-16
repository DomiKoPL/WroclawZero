#pragma once

#include <algorithm>
#include <fstream>
#include <model/model.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "MCTS.hpp"
#include "random.hpp"

class DatasetValidator {
   public:
    DatasetValidator(const Game& game, const ModelFactory& factory,
                     std::string data_path, bool verbose = false) {
        Model model = factory();

        std::ifstream data(data_path, std::ios::binary);
        assert(data.is_open());

        data.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
        std::cerr << "CNT: " << cnt << "\n";
        int not_draws_cnt = 0;

        for (int i = 0; i < cnt; i++) {
            // std::cerr << "input size: " << model.input_layer->output->size
            //           << "\n";

            model.input_layer->get_output().load(data);
            // std::cerr << model.input_layer->get_output() << "\n";
            model.forward();

            // std::cerr << model.get_output() << "\n";
            std::vector<float> policy(model.get_output().size - 1);
            data.read(reinterpret_cast<char*>(policy.data()),
                      sizeof(float) * policy.size());
            float value;
            data.read(reinterpret_cast<char*>(&value), sizeof(value));

            std::vector<float> model_policy(model.get_output().size - 1);
            for (int i = 0; i < model_policy.size(); i++) {
                model_policy[i] = model.get_policy(i);
            }

            float model_value = model.get_value();

            if (value != 0) {
                not_draws_cnt++;
            }

            if (value > 0 and model_value > 0) {
                good_sign++;
            } else if (value < 0 and model_value < 0) {
                good_sign++;
            }

            // for (auto i : policy) std::cerr << i << " "; std::cerr << "\n";
            // for (auto i : model_policy) std::cerr << i << " "; std::cerr <<
            // "\n";

            for (int i = 0; i < policy.size(); i++) {
                float a = policy[i], b = model_policy[i];
                policy_mse += (a - b) * (a - b);
            }

            auto best_move =
                std::max_element(model_policy.begin(), model_policy.end()) -
                model_policy.begin();

            if (policy[best_move] > 0) {
                optimal_move++;
            }

            // std::cerr << value << "\n";
            // std::cerr << model_value << "\n";
            value_mse += (value - model_value) * (value - model_value);
        }

        value_mse /= cnt;
        policy_mse /= cnt;
        optimal_move /= cnt;

        if (not_draws_cnt > 0) {
            good_sign /= not_draws_cnt;
        }

        if (verbose) {
            std::cerr << "value_mse=" << value_mse << "\n";
            std::cerr << "policy_mse=" << policy_mse << "\n";
            std::cerr << "good sign cnt=" << good_sign << "\n";
        }
    }

    float value_mse = 0;
    float policy_mse = 0;
    float good_sign = 0;
    float optimal_move = 0;
    int cnt;
};