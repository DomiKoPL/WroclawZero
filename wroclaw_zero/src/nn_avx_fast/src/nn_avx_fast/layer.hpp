#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <memory>
#include <string>
#include <utility>  // move

#include "activation_layers.hpp"
#include "tensor.hpp"

namespace nn_avx_fast {

class Layer {
   public:
    explicit Layer(std::string name, Activation activation = activationNONE)
        : name(std::move(name)), activation(activation), input_layer(nullptr) {}

    explicit Layer(std::string name, std::shared_ptr<Layer> input_layer,
                   Activation activation = activationNONE)
        : name(std::move(name)),
          activation(activation),
          input_layer(std::move(input_layer)) {}

    const std::string &get_name() const { return name; }

    void link(std::shared_ptr<Layer> _input_layer) {
        assert(input_layer == nullptr and "Cannot connect to multiple inputs.");
        input_layer = std::move(_input_layer);
        init();
    }

    virtual Tensor &get_output() { return output; }
    virtual void init() = 0;
    virtual void forward() = 0;
    virtual void precompute() {}

    virtual void save(std::ostream &os) = 0;
    virtual void load(std::istream &is) = 0;

    virtual size_t count_params() const = 0;

    virtual void fill(const float value) = 0;
    virtual void fill_random(const float min_value, const float max_value) = 0;

    std::string name;
    Activation activation;
    std::shared_ptr<Layer> input_layer;

   protected:
    Tensor output;
};

}  // namespace nn_avx_fast

#endif