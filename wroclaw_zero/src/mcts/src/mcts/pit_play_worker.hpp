#pragma once

#include <fstream>
#include <model/model.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "MCTS.hpp"

class PitPlayWorker {
   public:
    PitPlayWorker(Game game, const ModelFactory& factory1, MCTSConfig config1,
                  const ModelFactory& factory2, MCTSConfig config2, int games,
                  int threads_number, bool verbose = false, int ms = -1) {
        games_to_play_ = games;
        games_played_ = 0;
        verbose_ = verbose;
        games_length_ = 0;
        p2_wins_ = p1_wins_ = draws_ = 0;
        ms_ = ms;

        std::vector<std::thread> threads(threads_number);
        for (auto& thread : threads) {
            thread = std::thread(&PitPlayWorker::work, this, game, factory1,
                                 config1, factory2, config2);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (verbose) {
            display_progress_bar();

            const float all_games = (p1_wins_ + p2_wins_ + draws_);
            const float p1_wr =
                ((float)p1_wins_ + (float)draws_ * 0.5f) * 100 / all_games;
            const float p2_wr =
                ((float)p2_wins_ + (float)draws_ * 0.5f) * 100 / all_games;

            std::cerr << std::endl;
            std::cerr << "Model1 stats: WR: " << p1_wr << " wins: " << p1_wins_
                      << " draws: " << draws_ << "\n";
            std::cerr << "Model2 stats: WR: " << p2_wr << " wins: " << p2_wins_
                      << " draws: " << draws_ << "\n";
        }
    }

    float get_first_player_winrate() {
        const float all_games = (p1_wins_ + p2_wins_ + draws_);
        return ((float)p1_wins_ + (float)draws_ * 0.5f) * 100 / all_games;
    }

    float get_average_game_length() const {
        return (float)games_length_ / (float)games_to_play_;
    }

   private:
    void work(Game game, const ModelFactory& factory1, MCTSConfig config1,
              const ModelFactory& factory2, MCTSConfig config2) {
        Model model1 = factory1();
        Model model2 = factory2();

        while (true) {
            bool player1_starts = false;

            {
                std::lock_guard<std::mutex> guard(mutex_);

                if (games_played_ >= games_to_play_) {
                    break;
                }

                player1_starts = (games_played_ % 2);

                if (verbose_) {
                    display_progress_bar();
                }

                games_played_++;
            }

            MCTS player1(game, config1);
            MCTS player2(game, config2);

            MCTS* p0 = (player1_starts ? &player1 : &player2);
            MCTS* p1 = (player1_starts ? &player2 : &player1);
            Model* m0 = (player1_starts ? &model1 : &model2);
            Model* m1 = (player1_starts ? &model2 : &model1);

            Game state = game->clone();
            int game_length = 0;

            while (true) {
                (ms_ != -1 ? p0->search(*m0, ms_) : p0->search(*m0));
                const int p0_move = p0->get_best();
                p0->restore_root(p0_move, *m0);
                p1->restore_root(p0_move, *m1);
                state->make_move(p0_move);
                game_length++;

                if (state->is_terminal()) {
                    auto result = -state->get_game_result();

                    std::lock_guard<std::mutex> guard(mutex_);

                    games_length_ += game_length;

                    if (result == 1) {
                        (player1_starts ? p1_wins_ : p2_wins_)++;
                    } else if (result == -1) {
                        (player1_starts ? p2_wins_ : p1_wins_)++;
                    } else {
                        draws_++;
                    }

                    break;
                }

                (ms_ != -1 ? p1->search(*m1, ms_) : p1->search(*m1));
                const int p1_move = p1->get_best();
                p0->restore_root(p1_move, *m0);
                p1->restore_root(p1_move, *m1);
                state->make_move(p1_move);
                game_length++;

                if (state->is_terminal()) {
                    auto result = -state->get_game_result();

                    std::lock_guard<std::mutex> guard(mutex_);

                    games_length_ += game_length;

                    if (result == 1) {
                        (player1_starts ? p2_wins_ : p1_wins_)++;
                    } else if (result == -1) {
                        (player1_starts ? p1_wins_ : p2_wins_)++;
                    } else {
                        draws_++;
                    }

                    break;
                }
            }
        }
    }

    void display_progress_bar() {
        if (games_played_ % 100 != 0 and games_played_ != games_to_play_) {
            return;
        }

        float progress = (float)games_played_ / (float)games_to_play_;

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
        std::cerr << "] " << int(progress * 100.0) << " % ";

        if (games_played_ > 0) {
            const float all_games = (p1_wins_ + p2_wins_ + draws_);
            const float p1_wr =
                ((float)p1_wins_ + (float)draws_ * 0.5f) * 100 / all_games;
            const float p2_wr =
                ((float)p2_wins_ + (float)draws_ * 0.5f) * 100 / all_games;
            std::cerr << p1_wr << "\\";
            std::cerr << p2_wr;
        }

        std::cerr << "\r";

        std::cerr.flush();
    }

    int draws_;
    int p1_wins_;
    int p2_wins_;
    int games_to_play_;
    int games_played_;
    int games_length_;
    int ms_;
    bool verbose_;
    std::mutex mutex_;
};