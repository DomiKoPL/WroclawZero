#include "activation_layers.hpp"

namespace nn_avx_fast {
void activationNONE(Tensor &output){};

void activationRELU(Tensor &output) {
    const __m256 zero = _mm256_setzero_ps();

    for (size_t i = 0; i < output.xmm_size; ++i) {
        output.xmm[i].v = _mm256_max_ps(output.xmm[i].v, zero);
    }
};

void activationSIGMOID(Tensor &output) {
    const __m256 one = _mm256_set1_ps(1.0f);

    for (size_t i = 0; i < output.xmm_size; ++i) {
        output.xmm[i].v = exp256_ps(output.xmm[i].v);
        auto divisor = _mm256_add_ps(output.xmm[i].v, one);
        output.xmm[i].v = _mm256_div_ps(output.xmm[i].v, divisor);
    }
};

void activationSOFTMAX(Tensor &output) {
    const int rem = (8 - (output.size % 8)) % 8;

    if (rem != 0) {
        for (int i = 0; i < rem; ++i) {
            output.set_element(output.xmm_size * 8 - i - 1, -99999999.99f);
        }
    }

    float sum = 0.0f;

    for (uint32_t i = 0; i < output.xmm_size; ++i) {
        output.xmm[i].v = exp256_ps(output.xmm[i].v);
        sum += output.hsums(output.xmm[i].v)[0];
    }

    if (sum != 0) {
        output *= 1.f / sum;
    }
};

void activationTANH(Tensor &output) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 minus_one = _mm256_set1_ps(-1.0f);

    for (size_t i = 0; i < output.xmm_size; ++i) {
        auto ex = exp256_ps(output.xmm[i].v);
        auto emx = exp256_ps(_mm256_mul_ps(output.xmm[i].v, minus_one));

        output.xmm[i].v = _mm256_sub_ps(ex, emx);
        auto divisor = _mm256_add_ps(ex, emx);
        output.xmm[i].v = _mm256_div_ps(output.xmm[i].v, divisor);
    }
};

}  // namespace nn_avx_fast