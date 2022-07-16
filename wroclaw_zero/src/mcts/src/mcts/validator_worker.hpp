#pragma once

#include <algorithm>
#include <fstream>
#include <model/model.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "MCTS.hpp"
#include "random.hpp"

class Agent {
   public:
    Agent() {}

    virtual int get_best(Game& game) = 0;
};

class Depth1Agent : public Agent {
   public:
    Depth1Agent() {}

    int get_best(Game& game) {
        game->calc_legal_moves();
        std::vector<float> val;

        for (int i = 0; i < game->legal_moves_cnt; i++) {
            const int move = game->legal_moves[i];
            auto temp = game->clone();
            temp->make_move(move);
            val.push_back(-temp->eval());
        }

        int best = std::max_element(val.begin(), val.end()) - val.begin();
        int move = game->legal_moves[best];
        return move;
    }
};

class RandomAgent : public Agent {
   public:
    RandomAgent() {}

    int get_best(Game& game) {
        game->calc_legal_moves();
        int idx = Random::instance().next_int(game->legal_moves_cnt);
        return game->legal_moves[idx];
    }
};

template <class Agent>
class ValidatorWorker {
   public:
    ValidatorWorker(const Game& game, const ModelFactory& factory,
                    MCTSConfig config, int games, int threads_number,
                    bool verbose = false) {
        games_to_play_ = games;
        games_played_ = 0;
        verbose_ = verbose;
        p2_wins_ = p1_wins_ = draws_ = 0;
        games_length_ = 0;

        std::vector<std::thread> threads(threads_number);
        for (auto& thread : threads) {
            thread = std::thread(&ValidatorWorker::work, this, game, factory,
                                 config);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (verbose) {
            display_progress_bar();
            std::cerr << std::endl;

            const float all_games = (p1_wins_ + p2_wins_ + draws_);
            const float p1_wr =
                ((float)p1_wins_ + (float)draws_ * 0.5f) * 100 / all_games;
            const float p2_wr =
                ((float)p2_wins_ + (float)draws_ * 0.5f) * 100 / all_games;

            std::cerr << "Validation result\n";
            std::cerr << "Model stats: WR: " << p1_wr << " wins: " << p1_wins_
                      << " draws: " << draws_ << "\n";
            std::cerr << "Bot   stats: WR: " << p2_wr << " wins: " << p2_wins_
                      << " draws: " << draws_ << "\n";
            std::cerr.flush();
        }
    }

    float get_win_rate() const {
        const float all_games = (p1_wins_ + p2_wins_ + draws_);
        const float p1_wr =
            ((float)p1_wins_ + (float)draws_ * 0.5f) * 100 / all_games;

        return p1_wr;
    }

    float get_average_game_length() const {
        return (float)games_length_ / (float)games_to_play_;
    }

   private:
    void work(const Game& game, const ModelFactory& factory,
              MCTSConfig config) {
        Model model = factory();

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

            MCTS player1(game, config);
            Agent player2;

            Game state = game->clone();

            int game_length = 0;
            while (true) {
                if (player1_starts) {
                    player1.search(model);
                    const int p0_move = player1.get_best();
                    player1.restore_root(p0_move, model);
                    state->make_move(p0_move);
                } else {
                    const int p0_move = player2.get_best(state);
                    player1.restore_root(p0_move, model);
                    state->make_move(p0_move);
                }

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

                if (player1_starts) {
                    const int p0_move = player2.get_best(state);
                    player1.restore_root(p0_move, model);
                    state->make_move(p0_move);
                } else {
                    player1.search(model);
                    const int p0_move = player1.get_best();
                    player1.restore_root(p0_move, model);
                    state->make_move(p0_move);
                }
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
    bool verbose_;
    std::mutex mutex_;
};