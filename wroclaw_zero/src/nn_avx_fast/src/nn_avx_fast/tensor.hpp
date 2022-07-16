#pragma once

#include <immintrin.h>

#include <cassert>
#include <iostream>
#include <random>
#include <utility>  // move
#include <vector>

namespace nn_avx_fast {

union __m256_f {
    __m256 v;
    float f[8];
};

// https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned/8456491#8456491
// Since C++17 std::vector will work

using aligned_vector = std::vector<__m256_f>;

class Tensor {
   public:
    explicit Tensor(const std::vector<size_t> &shape) : shape(shape) {
        size = 1;

        for (size_t i : shape) {
            size *= i;
        }

        xmm_size = static_cast<size_t>(std::ceil(size / 8.f));
        xmm.resize(xmm_size);

        const auto zero = _mm256_setzero_ps();
        for (size_t i = 0; i < xmm_size; i++) {
            xmm[i].v = zero;
        }
    }

    explicit Tensor(aligned_vector data, const std::vector<size_t> &shape)
        : shape(shape), xmm(std::move(data)) {
        size = 1;

        for (size_t i : shape) {
            size *= i;
        }

        xmm_size = static_cast<size_t>(std::ceil(size / 8.f));
    }

    explicit Tensor() : shape({0}), size(0) {}

    inline static size_t get_index_3D(size_t x, size_t y, size_t z,
                                      const std::vector<size_t> &shape) {
        return z + (y + x * shape[1]) * shape[2];
    }

    inline static size_t get_index_4D(size_t x, size_t y, size_t z, size_t d,
                                      const std::vector<size_t> &shape) {
        return d + (z + (y + x * shape[1]) * shape[2]) * shape[3];
    }

    inline float get_element(size_t idx) const {
        return xmm[idx >> 3].f[idx & 7];
    }

    inline void set_element(size_t idx, float val) {
        xmm[idx >> 3].f[idx & 7] = val;
    }

    inline void set_chunk(size_t idx, const __m256 &chunk) {
        xmm[idx].v = chunk;
    }

    inline __m256 get_chunk(size_t idx) const { return xmm[idx].v; }

    inline float get_element2D(size_t x, size_t y) const {
        const size_t idx = y + x * shape[1];
        return xmm[idx >> 3].f[idx & 7];
    }

    inline float get_element3D(size_t x, size_t y, size_t z) const {
        return xmm[0].f[get_index_3D(x, y, z, shape)];
    }

    inline float get_element4D(size_t x, size_t y, size_t z, size_t d) const {
        return xmm[0].f[get_index_4D(x, y, z, d, shape)];
    }

    inline void set_element2D(size_t x, size_t y, float val) {
        const size_t idx = y + x * shape[1];
        xmm[idx >> 3].f[idx & 7] = val;
    }

    inline void set_element3D(size_t x, size_t y, size_t z, float val) {
        xmm[0].f[get_index_3D(x, y, z, shape)] = val;
    }

    inline void set_element4D(size_t x, size_t y, size_t z, size_t d,
                              float val) {
        xmm[0].f[get_index_4D(x, y, z, d, shape)] = val;
    }

    inline __m256 get_chunk2D(size_t x, size_t y) const {
        const size_t idx = y + x * shape[1];
        return xmm[idx].v;
    }

    Tensor &operator+=(const Tensor &other) {
        assert(size == other.size);

        for (size_t i = 0; i < xmm_size; i++) {
            set_chunk(i, _mm256_add_ps(xmm[i].v, other.xmm[i].v));
        }

        return *this;
    }

    Tensor &operator-=(const Tensor &other) {
        assert(size == other.size);

        for (size_t i = 0; i < xmm_size; i++) {
            set_chunk(i, _mm256_sub_ps(xmm[i].v, other.xmm[i].v));
        }

        return *this;
    }

    Tensor &operator+=(float val) {
        const __m256 chunk = _mm256_set1_ps(val);

        for (auto &i : xmm) {
            i.v = _mm256_add_ps(i.v, chunk);
        }

        return *this;
    }

    Tensor &operator-=(float val) {
        const __m256 chunk = _mm256_set1_ps(val);

        for (auto &i : xmm) {
            i.v = _mm256_sub_ps(i.v, chunk);
        }

        return *this;
    }

    Tensor &operator*=(float val) {
        const __m256 chunk = _mm256_set1_ps(val);

        for (auto &i : xmm) {
            i.v = _mm256_mul_ps(i.v, chunk);
        }

        return *this;
    }

    Tensor &operator/=(float val) {
        const __m256 chunk = _mm256_set1_ps(val);

        for (auto &i : xmm) {
            i.v = _mm256_div_ps(i.v, chunk);
        }

        return *this;
    }

    void reshape(const std::vector<size_t> &new_shape) {
        size_t new_size = 1;
        for (size_t i : new_shape) {
            new_size *= i;
        }

        assert(new_size == size);
        shape = new_shape;
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
        os << "[ ";

        for (size_t i = 0, idx = 0; i < tensor.xmm_size; i++) {
            for (size_t j = 0; j < 8; j++) {
                if (idx >= tensor.size) {
                    break;
                }

                os << tensor.xmm[i].f[j];

                idx++;
                if (idx < tensor.size) {
                    os << ", ";
                }
            }
        }

        return os << " ]";
    }

    std::string shape_str() const {
        std::string shape_str = "(";

        for (size_t i = 0; i < shape.size(); i++) {
            shape_str += std::to_string(shape[i]);
            if (i + 1 < shape.size()) shape_str += ", ";
        }

        shape_str += ")";
        return shape_str;
    }

    void load(std::istream &is) {
        xmm.resize(xmm_size);
        // last chunk could be partially loaded
        xmm.back().v = _mm256_setzero_ps();
        is.read(reinterpret_cast<char *>(xmm.data()), size * sizeof(float));
    }

    void save(std::ostream &os) const {
        os.write(reinterpret_cast<const char *>(xmm.data()),
                 size * sizeof(float));
    }

    // Does horizontal sum of a chunk v
    // Only works if v is __m256, __m128 requires less instructions
    inline __m256 hsums(__m256 const &v) {
        auto x = _mm256_permute2f128_ps(v, v, 1);
        auto y = _mm256_add_ps(v, x);
        x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
        x = _mm256_add_ps(x, y);
        y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_ps(x, y);
    };

    void fill(const float value) {
        // TODO: last chunk should be partially filled
        const auto val = _mm256_set1_ps(value);
        for (auto &i : xmm) {
            i.v = val;
        }
    }

    void fill_random(const float min_value, const float max_value) {
        static std::random_device rd;
        static std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist(min_value, max_value);

        for (size_t i = 0, idx = 0; i < xmm_size; i++) {
            for (size_t j = 0; j < 8; j++) {
                if (idx >= size) {
                    break;
                }

                xmm[i].f[j] = dist(mt);
                idx++;
            }
        }
    }

    std::vector<size_t> shape;
    aligned_vector xmm;
    size_t xmm_size;
    size_t size;
};

}  // namespace nn_avx_fast