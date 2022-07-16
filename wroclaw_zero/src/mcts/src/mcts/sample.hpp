#pragma once

#include <iostream>
#include <model/model.hpp>
#include <vector>

struct Sample {
    Tensor input;
    std::vector<int> legal_moves;
    std::vector<float> policy;
    float score;

    Sample() {}

    Sample(const std::vector<size_t>& input_shape, size_t policy_shape)
        : input(input_shape),
          legal_moves(policy_shape, 0),
          policy(policy_shape, 0) {}
};