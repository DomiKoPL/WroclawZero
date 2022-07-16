#pragma once

#include <cmath>
#include <functional>

#include "tensor.hpp"

namespace nn_avx_fast {

// AVX2 Exponential functions
#define MUL _mm256_mul_ps
#define FMA _mm256_fmadd_ps
#define SET _mm256_set1_ps
inline __m256 exp256_ps(const __m256 &V) {
    const __m256 exp_hi = SET(88.3762626647949f);
    const __m256 exp_lo = SET(-88.3762626647949f);
    const __m256 cLOG2EF = SET(1.44269504088896341f);
    const __m256 cexp_C1 = SET(0.693359375f);
    const __m256 cexp_C2 = SET(-2.12194440e-4f);
    const __m256 cexp_p0 = SET(1.9875691500E-4f);
    const __m256 cexp_p1 = SET(1.3981999507E-3f);
    const __m256 cexp_p2 = SET(8.3334519073E-3f);
    const __m256 cexp_p3 = SET(4.1665795894E-2f);
    const __m256 cexp_p4 = SET(1.6666665459E-1f);
    const __m256 cexp_p5 = SET(5.0000001201E-1f);

    __m256 x = V;
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = SET(1.0f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);
    fx = MUL(x, cLOG2EF);
    fx = _mm256_add_ps(fx, SET(0.5f));
    tmp = _mm256_floor_ps(fx);
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);
    tmp = MUL(fx, cexp_C1);
    __m256 z = MUL(fx, cexp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);
    z = MUL(x, x);
    __m256 y = cexp_p0;
    y = FMA(y, x, cexp_p1);
    y = FMA(y, x, cexp_p2);
    y = FMA(y, x, cexp_p3);
    y = FMA(y, x, cexp_p4);
    y = FMA(y, x, cexp_p5);
    y = FMA(y, z, x);
    y = _mm256_add_ps(y, one);
    imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = MUL(y, pow2n);
    return y;
}
#undef SET
#undef MUL
#undef FMA

using Activation = std::function<void(Tensor &)>;

void activationNONE(Tensor &);
void activationRELU(Tensor &);
void activationSIGMOID(Tensor &);
void activationSOFTMAX(Tensor &);
void activationTANH(Tensor &);

}  // namespace nn_avx_fast