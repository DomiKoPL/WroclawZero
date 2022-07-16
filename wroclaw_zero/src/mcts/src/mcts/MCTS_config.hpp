#pragma once

struct MCTSConfig {
    float cpuct_init = 1.0f;

    float dirichlet_noise_epsilon = 0.25f;
    float dirichlet_noise_alpha = 0.1f;  // Heuristic: α = 10 / n, where n =
                                         // maximum number of possible moves
    int number_of_iterations_per_turn = 1600;
    int temperature_turns = 30;  // use temperate in first 30 turns
    float temperature_max = 1.75;
    float temperature_min = 0.5;
    int init_reserved_nodes = 0;

    MCTSConfig() {}
};
