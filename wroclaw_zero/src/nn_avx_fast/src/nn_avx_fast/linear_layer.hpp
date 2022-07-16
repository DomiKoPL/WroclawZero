#pragma once
#include "layer.hpp"

namespace nn_avx_fast {

class LinearLayer : public Layer {
   public:
    LinearLayer(std::string name, size_t out_features,
                std::shared_ptr<Layer> input_layer,
                Activation activation = activationNONE)
        : Layer(name, input_layer, activation), out_features(out_features) {
        init();
    }

    LinearLayer(size_t out_features, std::shared_ptr<Layer> input_layer,
                Activation activation = activationNONE)
        : LinearLayer("linear", out_features, input_layer, activation) {}

    LinearLayer(std::string name, size_t out_features,
                Activation activation = activationNONE)
        : Layer(name, activation), out_features(out_features) {}

    LinearLayer(size_t out_features, Activation activation = activationNONE)
        : LinearLayer("linear", out_features, activation) {}

    void init() override {
        assert(input_layer->get_output().shape.size() == 1 and
               "Input must be a vector");

        auto in_features = input_layer->get_output().size;
        weights.resize(in_features, Tensor({out_features}));
        bias = Tensor({out_features});
        output = Tensor(std::vector<size_t>{out_features});
    }

    void forward() override {
        assert(input_layer != nullptr and
               "Layer must be linked to input layer.");

        const auto &input = input_layer->get_output();

        assert(input.shape.size() == 1 and "Input must be a vector");

        for (size_t i = 0; i < output.xmm_size; i++) {
            output.xmm[i].v = bias.xmm[i].v;
        }

        assert(input.shape.size() == 1 and "Input must be a vector");
            
        for (int j = 0; j < input.size; j++) {
            const float val = input.xmm[0].f[j];

            if (val == 0.0f) {
                continue;
            } else if (val == 1.0f) {
                for (int i = 0; i < output.xmm_size; i++) {
                    output.xmm[i].v =
                        _mm256_add_ps(weights[j].xmm[i].v, output.xmm[i].v);
                }
            } else {
                const __m256 fm = _mm256_set1_ps(val);

                for (int i = 0; i < output.xmm_size; i++) {
                    // a * b + c
                    output.xmm[i].v = _mm256_fmadd_ps(fm, weights[j].xmm[i].v,
                                                      output.xmm[i].v);
                }
            }
        }

        activation(output);
    }

    virtual void save(std::ostream &os) override {
        for (auto &weight : weights) {
            weight.save(os);
        }

        bias.save(os);
    }

    virtual void load(std::istream &is) override {
        for (auto &weight : weights) {
            weight.load(is);
        }

        bias.load(is);
    }

    virtual size_t count_params() const override {
        return bias.size + weights.size() * weights[0].size;
    }

    virtual void fill(const float value) override {
        for (auto &weight : weights) {
            weight.fill(value);
        }

        bias.fill(value);
    }

    virtual void fill_random(const float min_value, const float max_value) {
        for (auto &weight : weights) {
            weight.fill_random(min_value, max_value);
        }

        bias.fill_random(min_value, max_value);
    }

    std::vector<Tensor> weights;
    Tensor bias;
    size_t out_features;
};

}  // namespace nn_avx_fast