#pragma once

#include <fstream>
#include <model/model.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "MCTS.hpp"

class SelfPlayWorker {
   public:
    SelfPlayWorker(const Game& game, const ModelFactory& factory1,
                   MCTSConfig config1, const ModelFactory& factory2,
                   MCTSConfig config2, int games, int threads_number,
                   bool verbose = false) {
        games_to_play = games;
        games_played_ = 0;
        games_length = 0;
        verbose_ = verbose;
        first_moves_vis.resize(game->get_maximum_number_of_moves());

        const int max_samples =
            games_to_play * game->get_maximum_number_of_turns();
        const auto input_shape = game->get_input_shape();
        const auto policy_shape = game->get_maximum_number_of_moves();
        game_samples.resize(max_samples, Sample(input_shape, policy_shape));
        game_samples_count = 0;

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads(threads_number);
        for (auto& thread : threads) {
            thread = std::thread(&SelfPlayWorker::work, this, game, factory1,
                                 config1, factory2, config2);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (verbose) {
            display_progress_bar();

            std::cerr << std::endl;
            
            // std::cerr << "Samples count " << game_samples_count << "\n";

            // std::cerr << "First moves stats\n";
            // for (int i = 0; i < first_moves_vis.size(); i++) {
            //     int cnt = first_moves_vis[i];
            //     if (cnt > 0) {
            //         std::cerr << "Move[" << i << "] cnt= " << cnt << "\n";
            //     }
            // }
        }
    }

    float get_average_game_length() const {
        return (float)games_length / (float)games_to_play;
    }

    int games_won_ = 0;
    int games_lost_ = 0;
    int draws_ = 0;
    int game_samples_count;
    int games_to_play;
    int games_length;
    std::vector<int> first_moves_vis;
    std::vector<Sample> game_samples;

   private:
    void work(const Game& game, const ModelFactory& factory1,
              MCTSConfig config1, const ModelFactory& factory2,
              MCTSConfig config2) {
        Model model1 = factory1();
        Model model2 = factory2();

        const int max_samples = game->get_maximum_number_of_turns();
        const auto input_shape = game->get_input_shape();
        const auto policy_shape = game->get_maximum_number_of_moves();
        std::vector<Sample> samples(max_samples,
                                    Sample(input_shape, policy_shape));

        MCTS player1(game, config1);
        MCTS player2(game, config2);

        while (true) {
            bool player1_starts = false;

            {
                std::lock_guard<std::mutex> guard(mutex_);

                if (games_played_ >= games_to_play) {
                    break;
                }

                player1_starts = (games_played_ % 2);

                if (verbose_) {
                    display_progress_bar();
                }

                games_played_++;
            }

            player1.reset(game);
            player2.reset(game);

            MCTS* p0 = (player1_starts ? &player1 : &player2);
            MCTS* p1 = (player1_starts ? &player2 : &player1);
            Model* m0 = (player1_starts ? &model1 : &model2);
            Model* m1 = (player1_starts ? &model2 : &model1);
            size_t game_length = 0;

            int first_move;

            while (true) {
                // Player 0 move
                p0->search(*m0);
                const int p0_move = p0->get_best();

                p0->get_sample(samples[game_length++]);

                p0->restore_root(p0_move, *m0);
                p1->restore_root(p0_move, *m1);

                if (game_length == 1) {
                    first_move = p0_move;
                }

                if (p0->get_root_state()->is_terminal()) {
                    break;
                }

                // Player 1 move
                p1->search(*m1);
                const int p1_move = p1->get_best();

                p1->get_sample(samples[game_length++]);

                p0->restore_root(p1_move, *m0);
                p1->restore_root(p1_move, *m1);

                if (p0->get_root_state()->is_terminal()) {
                    break;
                }
            }

            // player1.debug_stats();
            
            auto result = -p0->get_root_state()->get_game_result();
            auto score = -p0->get_root_state()->get_scaled_game_result();

            const float k_min = 0.7;
            float decay = 0;

            if (game_length > 1) {
                decay = (1 - k_min) / (float)(game_length - 1);
            }

            for (int i = (int)game_length - 1; i >= 0; i--) {
                float k = k_min + decay * i;
                samples[i].score = k * result + (1 - k) * samples[i].score;

                score = -score;
                result = -result;
            }

            {
                std::lock_guard<std::mutex> guard(mutex_);
                games_length += game_length;

                if (result == 1)
                    games_won_ += 1;
                else if (result == -1)
                    games_lost_ += 1;
                else
                    draws_ += 1;

                for (size_t i = 0; i < game_length; i++) {
                    game_samples[game_samples_count++] = samples[i];
                }

                first_moves_vis[first_move]++;
            }
        }

        // {
        //     std::lock_guard<std::mutex> guard(mutex_);
        //     games_length += thread_game_length;

        //     for (const auto& sample : thread_samples) {
        //         game_samples_.push_back(sample);
        //     }

        //     for (int i = 0; i < first_moves_vis.size(); i++) {
        //         first_moves_vis[i] += thread_first_moves_vis[i];
        //     }
        // }
    }

    void display_progress_bar() {
        if (games_played_ % 100 != 0 and games_played_ != games_to_play) {
            return;
        }

        float progress = (float)games_played_ / (float)games_to_play;

        int barWidth = 70;

        std::cerr << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos)
                std::cerr << "=";
            else if (i == pos)
                std::cerr << ">";
            else
                std::cerr << " ";
        }
        std::cerr << "] " << int(progress * 100.0) << " %\r";
        std::cerr.flush();
    }

    int games_played_;
    bool verbose_;

    std::mutex mutex_;
};