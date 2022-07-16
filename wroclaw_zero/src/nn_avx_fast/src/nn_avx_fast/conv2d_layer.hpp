#pragma once

#include "layer.hpp"

namespace nn_avx_fast {

class Conv2DLayer : public Layer {
   public:
    Conv2DLayer(std::string name, size_t out_channels, size_t kernel_x,
                size_t kernel_y, size_t stride_x, size_t stride_y,
                size_t padding_x, size_t padding_y,
                Activation activation = activationNONE)
        : Layer(name, activation),
          out_channels(out_channels),
          kernel_x(kernel_x),
          kernel_y(kernel_y),
          stride_x(stride_x),
          stride_y(stride_y),
          padding_x(padding_x),
          padding_y(padding_y) {
        kernel = Tensor(std::vector<size_t>{out_channels, kernel_x, kernel_y});
        kernel.fill(1);
    }

    Conv2DLayer(size_t out_channels, size_t kernel_x, size_t kernel_y,
                size_t stride_x, size_t stride_y, size_t padding_x,
                size_t padding_y, Activation activation = activationNONE)
        : Conv2DLayer("Conv2D", out_channels, kernel_x, kernel_y, stride_x,
                      stride_y, padding_x, padding_y, activation) {}

    void init() override {
        const auto input_shape = input_layer->get_output().shape;
        assert(input_shape.size() == 3 and "Input must be a 3D tensor");

        kernel_weights = Tensor(std::vector<size_t>{
            out_channels, input_shape[0], kernel.shape[1], kernel.shape[2]});

        kernel_bias = Tensor(std::vector<size_t>{out_channels});

        output = Tensor(std::vector<size_t>{
            out_channels,
            (size_t)std::floor(
                (double)(input_shape[1] + (padding_x * 2) - kernel.shape[1]) /
                    stride_x +
                1),
            (size_t)std::floor(
                (double)(input_shape[2] + (padding_y * 2) - kernel.shape[2]) /
                    stride_y +
                1)});

        bias = Tensor(std::vector<size_t>{output.size});
    }

    void precompute() override {
        auto& input = input_layer->get_output();
        const auto input_shape = input.shape;

        std::vector<Tensor> weights(input.size,
                                    Tensor(std::vector<size_t>{output.size}));
        weight_idx.resize(input.size);
        weight_val.resize(input.size);

        const int temp = output.shape[1] * output.shape[2];

        for (int i = 0; i < bias.size; i++) {
            bias.set_element(i, kernel_bias.get_element(i / temp));
        }

        // std::cerr << "BIAS:" << bias << "\n";

        const int in_sx = input_shape[1];
        const int in_sy = input_shape[2];

        for (int d = 0; d < output.shape[0]; d++) {
            int y = -(int)padding_y;

            for (int ay = 0; ay < output.shape[2]; ay++) {
                int x = -(int)padding_x;

                for (int ax = 0; ax < output.shape[1]; ax++) {
                    for (int fy = 0; fy < kernel.shape[2]; fy++) {
                        int oy = y + fy;

                        for (int fx = 0; fx < kernel.shape[1]; fx++) {
                            int ox = x + fx;

                            if (oy < 0 or oy >= in_sy or ox < 0 or
                                ox >= in_sx) {
                                continue;
                            }

                            // std::cerr << "OX OY:" << ox << " " << oy << "\n";
                            // std::cerr << "AX AY:" << ax << " " << ay << "\n";

                            for (int ind = 0; ind < input_shape[0]; ind++) {
                                int indw = Tensor::get_index_3D(ind, ox, oy,
                                                                input_shape);

                                // std::cerr << "INDW: " << indw << "\n";

                                float kw = kernel.get_element3D(d, fx, fy);

                                // std::cerr << "KW:" << kw << "\n";
                                if (kw != 0.f) {
                                    // std::cerr << "W idx: " <<
                                    // Tensor::get_index_3D(
                                    //     d, ax, ay, output.shape) << "\n";

                                    // std::cerr << "KWeight: " <<
                                    // kernel_weights.get_element4D(d, ind, fx,
                                    // fy) << "\n";

                                    weights[indw].xmm[0].f[Tensor::get_index_3D(
                                        d, ax, ay, output.shape)] +=
                                        kernel_weights.get_element4D(d, ind, fx,
                                                                     fy);
                                }
                            }
                        }
                    }
                    x += stride_x;
                }
                y += stride_y;
            }
        }

        for (int i = 0; i < input.size; i++) {
            for (int s = 0; s < weights[i].xmm_size; s++) {
                auto val = weights[i].xmm[s];
                bool is_non_zero = val.f[0] != 0.0f or val.f[1] != 0.0f or
                                   val.f[2] != 0.0f or val.f[3] != 0.0f or
                                   val.f[4] != 0.0f or val.f[5] != 0.0f or
                                   val.f[6] != 0.0f or val.f[7] != 0.0f;

                if (is_non_zero) {
                    weight_val[i].emplace_back(val);
                    weight_idx[i].emplace_back(s);
                }
            }
        }
    }

    void forward() override {
        assert(input_layer != nullptr and
               "Layer must be linked to input layer.");

        const auto& input = input_layer->get_output();

        for (size_t i = 0; i < output.xmm_size; i++) {
            output.xmm[i].v = bias.xmm[i].v;
        }

        for (size_t j = 0; j < input.size; j++) {
            const float val = input.xmm[0].f[j];

            if (val == 0.0f) {
                continue;
            } else if (val == 1.0f) {
                // std::cerr << "J:" << j << "val:" << val << "\n";
                for (size_t i = 0; i < weight_idx[j].size(); i++) {
                    auto idx = weight_idx[j][i];
                    // std::cerr << "IDX:" << idx << "\n";

                    output.xmm[idx].v =
                        _mm256_add_ps(weight_val[j][i].v, output.xmm[idx].v);
                }
                // std::cerr << output << "\n";
            } else {
                const __m256 fm = _mm256_set1_ps(val);

                for (size_t i = 0; i < weight_idx[j].size(); i++) {
                    // a * b + c
                    auto idx = weight_idx[j][i];
                    output.xmm[idx].v = _mm256_fmadd_ps(fm, weight_val[j][i].v,
                                                        output.xmm[idx].v);
                }
            }
        }
    }

    virtual void save(std::ostream& os) override {
        kernel_weights.save(os);
        kernel_bias.save(os);
    }

    virtual void load(std::istream& is) override {
        kernel_weights.load(is);
        kernel_bias.load(is);
        // std::cerr << "W:" << kernel_weights << "\n";
        // std::cerr << "B:" << kernel_bias << "\n";

        precompute();
    }

    virtual size_t count_params() const override {
        return kernel_weights.size + kernel_bias.size;
    }

    virtual void fill(const float value) override {
        // TODO: implement
        abort();
    }

    virtual void fill_random(const float min_value, const float max_value) {
        // TODO: implement
        abort();
    }

    size_t out_channels;
    size_t kernel_x;
    size_t kernel_y;
    size_t stride_x;
    size_t stride_y;
    size_t padding_x;
    size_t padding_y;
    Tensor kernel;
    Tensor kernel_weights;
    Tensor kernel_bias;
    Tensor bias;

    __attribute__((aligned(32))) std::vector<std::vector<__m256_f>> weight_val;
    std::vector<std::vector<int>> weight_idx;
};

}  // namespace nn_avx_fast
