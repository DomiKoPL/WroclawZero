#pragma once

#include <nn_avx_fast/common.hpp>
using namespace nn_avx_fast;

#include <functional>
#include <games/abstract_game.hpp>
#include <unordered_map>

class Model : public Sequential {
   public:
    template <class... Layers>
    Model(std::string name, std::shared_ptr<InputLayer> input_layer,
          Layers... layers)
        : Sequential(name, input_layer, layers...),
          cache_hit(0),
          cache_miss(0) {}

    template <class... Layers>
    Model(std::shared_ptr<InputLayer> input_layer, Layers... layers)
        : Model("Model", input_layer, layers...) {
        // cache_.reserve(1'000'000);
    }

    virtual void forward() override {
        Sequential::forward();

        //* copy and clear gamestate value from output
        auto& output = Sequential::get_output();
        const float value = std::tanh(output.get_element(0));
        const float inf = 999999999.99f;
        output.set_element(0, -inf);

        activationSOFTMAX(output);

        //* copy gamestate value back to output
        output.set_element(0, value);
    }

    virtual void forward(const std::shared_ptr<AbstractGame<Tensor>>& game) {
        // uint64_t hash = game->calc_hash();

        // auto it = cache_.find(hash);

        // if (it != cache_.end()) {
        //     // TODO: we can copy only legal moves
        //     output->xmm = it->second;
        //     cache_hit++;
        //     return;
        // }

        Sequential::forward();

        // std::cerr << "AFTER FORWARD\n";
        // std::cerr << *output << "\n";

        auto& output = Sequential::get_output();
        //* copy and clear gamestate value from output
        const float value = std::tanh(output.get_element(0));
        const float inf = 999999999.99f;
        output.set_element(0, -inf);

        if (game->legal_moves_cnt > 0u) {
            //* Clear invalid moves
            static std::vector<bool> legal;

            legal.assign(output.size, false);

            for (int i = 0; i < game->legal_moves_cnt; i++) {
                const int move = game->legal_moves[i];
                legal[move] = true;
            }

            for (int i = 1; i < output.size; i++) {
                if (not legal[i - 1]) {
                    output.set_element(i, -inf);
                }
            }

            // std::cerr << "BEFORE SOFTMAX\n";
            // std::cerr << *output << "\n";
            activationSOFTMAX(output);
            // std::cerr << "AFTER SOFTMAX\n";
            // std::cerr << *output << "\n";
        } else {
            output.fill(0);
        }

        //* copy gamestate value back to output
        output.set_element(0, value);
        // std::cerr << *output << "\n";

        // cache_[hash] = output->xmm;
        // cache_miss++;
    }

    void set_input(size_t idx, float val) {
        input_layer->get_output().set_element(idx, val);
    }

    void set_input(Tensor input) { input_layer->get_output() = input; }

    float get_value() { return Sequential::get_output().get_element(0); }

    float get_policy(int move) {
        return Sequential::get_output().get_element(move + 1);
    }

    int cache_hit, cache_miss;

   private:
    // std::unordered_map<uint64_t, aligned_vector> cache_;
};

using ModelFactory = std::function<Model(void)>;