#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <memory>
#include <vector>

#include "layer.hpp"

namespace nn_avx_fast {

class Sequential : public Layer {
   public:
    template <class... Layers>
    Sequential(std::string name, std::shared_ptr<Layer> input_layer,
               Layers... layers)
        : Layer(name, input_layer) {
        m_layers.push_back(std::move(input_layer));

        if constexpr (sizeof...(layers) > 0) {
            link_layers(layers...);
        }
    }

    template <class... Layers>
    Sequential(std::shared_ptr<Layer> input_layer, Layers... layers)
        : Sequential("Sequential", input_layer, layers...) {}

    virtual void init() override {}

    virtual Tensor &get_output() override {
        return m_layers.back()->get_output();
    }

    virtual void forward() override {
        for (auto &layer : m_layers) {
            layer->forward();
        }
    }

    virtual void save(std::ostream &os) override {
        for (auto &layer : m_layers) {
            layer->save(os);
        }
    }
    virtual void load(std::istream &is) override {
        for (auto &layer : m_layers) {
            layer->load(is);
        }
    }

    virtual size_t count_params() const override {
        size_t params = 0;
        for (auto &layer : m_layers) {
            params += layer->count_params();
        }
        return params;
    }

    virtual void fill(const float value) override {
        for (auto &layer : m_layers) {
            layer->fill(value);
        }
    }

    virtual void fill_random(const float min_value,
                             const float max_value) override {
        for (auto &layer : m_layers) {
            layer->fill_random(min_value, max_value);
        }
    }

    void append(std::shared_ptr<Layer> layer) {
        m_layers.push_back(std::move(layer));
        m_layers.back()->link(m_layers[(int)m_layers.size() - 2]);
    }

   private:
    template <class... Layers>
    void link_layers(std::shared_ptr<Layer> layer, Layers... layers) {
        m_layers.push_back(std::move(layer));
        m_layers.back()->link(m_layers[(int)m_layers.size() - 2]);

        if constexpr (sizeof...(layers) > 0) {
            link_layers(layers...);
        }
    }

    std::vector<std::shared_ptr<Layer>> m_layers;
};

}  // namespace nn_avx_fast

#endif