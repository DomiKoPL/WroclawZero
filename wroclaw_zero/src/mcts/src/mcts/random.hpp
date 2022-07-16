#pragma once

#include <cstdint>
#include <limits>
#include <random>

class Random {
   public:
    Random(int SEED) {
        std::random_device rd;
        std::mt19937 e2(SEED);
        setSeed(e2);
    }

    Random() {
        std::random_device rd;
        std::seed_seq seedseq1{rd(), rd(), rd()};
        std::mt19937 e2(rd());
        setSeed(e2);
    }

    static Random &instance() {
        static Random random;
        return random;
    }

    void setSeed(std::mt19937 &e2) {
        std::uniform_int_distribution<int32_t> dist(
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max());
        m_seed = dist(e2);
    }

    inline uint32_t xrandom() {
        m_seed = m_seed * K_m + K_a;
        return (uint32_t)(m_seed >> (29 - (m_seed >> 61)));
    }

    inline bool next_bool() { return (xrandom() & 4) == 4; }

    inline uint32_t next_int(const uint32_t &range) {
        return xrandom() % range;
    }

    inline int32_t next_int(const int32_t &a, const int32_t &b) {
        return (int32_t)next_int((uint32_t)(b - a + 1)) + a;
    }

    inline float next_float() {
        uint32_t xr = xrandom();
        if (xr == 0U) return 0.0f;
        union {
            float f;
            uint32_t i;
        } pun = {(float)xr};
        pun.i -= 0x10000000U;

        return pun.f;
    }

    inline float next_float(const float &a, const float &b) {
        return next_float() * (b - a) + a;
    }

   private:
    uint64_t m_seed;
    static const uint64_t K_m = 0x9b60933458e17d7d;
    static const uint64_t K_a = 0xd737232eeccdf7ed;
};
