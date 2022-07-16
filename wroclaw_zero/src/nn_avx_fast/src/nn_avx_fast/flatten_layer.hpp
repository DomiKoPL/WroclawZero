#pragma once

#include "layer.hpp"

namespace nn_avx_fast {
class FlattenLayer : public Layer {
   public:
    FlattenLayer(std::string name) : Layer(name) {}

    FlattenLayer() : FlattenLayer("flatten") {}

    void init() override {}

    Tensor &get_output() override { return input_layer->get_output(); }

    void forward() override {
        assert(input_layer != nullptr and
               "Layer must be linked to input layer.");

        auto &input = input_layer->get_output();
        input.shape = std::vector<size_t>{input.size};
    }

    virtual void save(std::ostream &os) override {}

    virtual void load(std::istream &is) override {}

    virtual size_t count_params() const override { return 0; }

    virtual void fill(const float value) override {}

    virtual void fill_random(const float min_value, const float max_value) {}
};

}  // namespace nn_avx_fast
