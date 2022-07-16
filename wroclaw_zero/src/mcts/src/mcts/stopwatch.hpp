#pragma once

#include <chrono>
#include <cstdint>

#define NOW() std::chrono::high_resolution_clock::now()

class Stopwatch {
   public:
    Stopwatch(uint64_t ms) { start(ms); }

    void start(uint64_t ms) {
        c_time = NOW();
        c_timeout = c_time + std::chrono::milliseconds(ms);
    }

    void set_timeout(uint64_t ms) {
        c_timeout = c_time + std::chrono::milliseconds(ms);
    }

    inline bool timeout() { return NOW() > c_timeout; }

    long long elapsed_milliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(NOW() -
                                                                     c_time)
            .count();
    }

   private:
    std::chrono::high_resolution_clock::time_point c_time, c_timeout;
};