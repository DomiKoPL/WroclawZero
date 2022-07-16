#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "layer.hpp"

namespace nn_avx_fast {

class InputLayer : public Layer {
   public:
    InputLayer(std::string name, std::vector<size_t> input_shape)
        : Layer(name) {
        output = Tensor(input_shape);
    }

    InputLayer(std::vector<size_t> input_shape)
        : InputLayer("input", input_shape) {}

    virtual void init() override {}
    virtual void forward() override {}

    virtual void save(std::ostream &os) override {}
    virtual void load(std::istream &is) override {}

    virtual size_t count_params() const override { return 0; }

    void set(size_t idx, float val) { output.set_element(idx, val); }

    virtual void fill(const float value) override { output.fill(value); }

    virtual void fill_random(const float min_value,
                             const float max_value) override {
        output.fill_random(min_value, max_value);
    }
};

}  // namespace nn_avx_fast

#endif