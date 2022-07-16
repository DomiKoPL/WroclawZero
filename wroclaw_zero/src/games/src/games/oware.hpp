#pragma once

#include <cassert>
#include <iostream>

#include "abstract_game.hpp"

template <class Tensor>
class OwareGame : public AbstractGame<Tensor> {
   public:
    OwareGame() { reset(); }

    void reset() {
        state_[0] = state_[1] = 0;
        score0_ = score1_ = 0;

        for (int col = 0; col < 6; col++) {
            cell0_[col] = cell1_[col] = 4;
        }

        free_bits_ = 0;
        turn_ = -1;
    }

    void read() {
        turn_++;
        id_to_play_ = 0;

        int in_game_seeds = 0;
        int cg_id = 0;

        for (int player = 0; player < 2; player++) {
            for (int col = 0; col < 6; col++) {
                int seed;
                std::cin >> seed;

                in_game_seeds += seed;

                if (turn_ == 0 and seed != 4) {
                    cg_id = 1;
                }

                (player == 0 ? cell0_ : cell1_)[col] = seed;
            }
        }

        score1_ = 48 - in_game_seeds - score0_;

        if (turn_ == 0 and cg_id == 1) {
            turn_++;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const OwareGame& game) {
        // const uint8_t *my_cells, *enemy_cells;
        // uint8_t my_score = (game.id_to_play_ == 0 ? game.score0_ :
        // game.score1_); uint8_t enemy_score = (game.id_to_play_ == 0 ?
        // game.score1_ : game.score0_);

        // if (game.id_to_play_ == 0) {
        //     my_cells = &game.cell0_[0];
        //     enemy_cells = &game.cell1_[0];
        // } else {
        //     my_cells = &game.cell1_[0];
        //     enemy_cells = &game.cell0_[0];
        // }

        for (int i = 0; i < 6; i++) {
            os << int(game.cell1_[5 - i]) << " ";
        }
        os << "\n";

        for (int i = 0; i < 6; i++) {
            os << int(game.cell0_[i]) << " ";
        }
        os << "\n";

        os << int(game.score0_) << " " << int(game.score1_) << " "
           << int(game.turn_) << "\n";

        return os;
    }

    void calc_legal_moves() override {
        OwareGame::legal_moves_cnt = 0;
        // legal_moves_cnt = 0;

        const auto opponent_seeds = (state_[1 - id_to_play_] >> 16);
        const bool opponent_can_play = (opponent_seeds != 0);

        uint8_t* seeds = (id_to_play_ == 0 ? &cell0_[0] : &cell1_[0]);

        if (opponent_can_play) {
            for (int i = 0; i < 6; i++) {
                if (seeds[i] > 0) {
                    OwareGame::add_legal_move(i);
                }
            }
        } else {
            for (int i = 0; i < 6; i++) {
                if (seeds[i] >= 6 - i) {
                    OwareGame::add_legal_move(i);
                }
            }
        }
    }

    void make_move(int move_id) override {
        uint8_t *my_cells, *enemy_cells;
        uint8_t& my_score = (id_to_play_ == 0 ? score0_ : score1_);
        uint8_t& enemy_score = (id_to_play_ == 0 ? score1_ : score0_);

        if (id_to_play_ == 0) {
            my_cells = &cell0_[0];
            enemy_cells = &cell1_[0];
        } else {
            my_cells = &cell1_[0];
            enemy_cells = &cell0_[0];
        }

        bool can_capture = false;
        int sow_pos = move_id;

        {
            // sowing
            int seeds_to_place = my_cells[move_id];
            int full_sow = 0;
            if (seeds_to_place > 11) {
                int full_sow = (seeds_to_place - 1) / 11;

                for (int i = 0; i < 6; i++) {
                    // fast sow
                    cell0_[i] += full_sow;
                    cell1_[i] += full_sow;
                }

                seeds_to_place %= 11;
                if (seeds_to_place == 0) {
                    seeds_to_place = 11;
                }
            }

            uint8_t *sow_cell = my_cells, *swap_cell = enemy_cells;
            while (seeds_to_place--) {
                ++sow_pos;
                if (sow_pos >= 6) {
                    sow_pos = 0;
                    std::swap(sow_cell, swap_cell);
                }

                ++sow_cell[sow_pos];
            }

            can_capture = (sow_cell == enemy_cells);
        }

        my_cells[move_id] = 0;

        if (can_capture and
            (enemy_cells[sow_pos] == 2 or enemy_cells[sow_pos] == 3)) {
            const uint64_t mask_4 = 0xFCFCFCFCFCFCUL;  // what is this???
            const auto opponent_seeds = (state_[1 - id_to_play_] >> 16);
            bool can_capture = true;

            bool untouched_has_seeds =
                (opponent_seeds >> (8 * (1 + sow_pos))) != 0;

            bool touched_bigger_than_3 = (opponent_seeds & mask_4) != 0;

            if (not untouched_has_seeds and not touched_bigger_than_3) {
                int total = 0;
                int capture = 0;
                for (int i = 5; i >= 0; i--) {
                    total += enemy_cells[i];

                    if (i <= sow_pos) {
                        if (enemy_cells[i] != 2 and enemy_cells[i] != 3) {
                            break;
                        }

                        capture += enemy_cells[i];
                    }

                    if (total != capture) {
                        break;
                    }
                }

                can_capture = (total != capture);
            }

            if (can_capture) {
                for (int i = sow_pos; i >= 0; i--) {
                    if (enemy_cells[i] != 2 and enemy_cells[i] != 3) {
                        break;
                    }

                    my_score += enemy_cells[i];
                    enemy_cells[i] = 0;
                }
            }
        }

        // swap_players();
        id_to_play_ = 1 - id_to_play_;
        turn_++;

        if (turn_ == 200) {
            game_ended_ = true;
            return;
        }

        if (not have_any_legal_moves()) {
            for (int i = 0; i < 6; i++) {
                my_score += my_cells[i];
                enemy_score += enemy_cells[i];
                enemy_cells[i] = 0;
                my_cells[i] = 0;
            }

            game_ended_ = 1;
            return;
        }

        if (score0_ > 24 or score1_ > 24) {
            game_ended_ = 1;
            return;
        }
    }

    int get_maximum_number_of_turns() const override { return 200; }

    int get_turn_number() const override { return (int)turn_; }

    bool is_terminal() const override { return game_ended_; }

    int get_maximum_number_of_moves() const override { return 6; }

    int get_game_result() const override {
        assert(game_ended_);

        uint8_t my_score = (id_to_play_ == 0 ? score0_ : score1_);
        uint8_t enemy_score = (id_to_play_ == 0 ? score1_ : score0_);

        // std::cerr << "GET RESULT: id" << int(id_to_play_) << " " <<
        // int(my_score) << " " << int(enemy_score) << "\n";

        if (my_score < enemy_score) return -1;
        if (my_score > enemy_score) return 1;
        return 0;
    }

    float get_scaled_game_result() const {
        assert(game_ended_);

        uint8_t my_score = (id_to_play_ == 0 ? score0_ : score1_);
        uint8_t enemy_score = (id_to_play_ == 0 ? score1_ : score0_);

        const float turn_diff = 200 - (int)turn_;
        // const float temp = 200 * 0.0007f + 0.85;

        if (my_score < enemy_score) {
            return -0.85f - turn_diff * 0.0007f;
        }

        if (my_score > enemy_score) {
            return 0.85f + turn_diff * 0.0007f;
        }

        return 0;
    }

    /*  
        TODO: dodac parzystosc liczby tur do konca do inputu
            parzystosc raczej nic nie da
            byc moze dobrze bedzie dodac turn / 200
    */
   
    void get_input_for_network(Tensor& input) const override {
        // assert(input.shape[0] == 24 * 12 + 2 * 27);
        input.fill(0);
        // Tensor input(std::vector<size_t>{24 * 12 + 2 * 27});

        const uint8_t *my_cells, *enemy_cells;
        uint8_t my_score = (id_to_play_ == 0 ? score0_ : score1_);
        uint8_t enemy_score = (id_to_play_ == 0 ? score1_ : score0_);

        // 434 - Robo input 
        const int X = 24 * 12 + 2 * 27;

        if (id_to_play_ == 0) {
            my_cells = &cell0_[0];
            enemy_cells = &cell1_[0];
        } else {
            my_cells = &cell1_[0];
            enemy_cells = &cell0_[0];
        }

        for (int i = 0; i < 6; i++) {
            const int val0 = (my_cells[i] > 23 ? 23 : (int)my_cells[i]);
            input.set_element(24 * i + val0, 1);

            const int val1 = (enemy_cells[i] > 23 ? 23 : (int)enemy_cells[i]);
            input.set_element(24 * (i + 6) + val1, 1);
        }

        const int offset = 24 * 12;
        const int val0 = (my_score > 26 ? 26 : (int)my_score);
        input.set_element(offset + val0, 1);

        const int val1 = (enemy_score > 26 ? 26 : (int)enemy_score);
        input.set_element(offset + 26 + val1, 1);

        // return input;
    }

    std::vector<size_t> get_input_shape() const override {
        return std::vector<size_t>{24 * 12 + 2 * 27};
    }

    std::shared_ptr<AbstractGame<Tensor>> clone() const override {
        std::shared_ptr<AbstractGame<Tensor>> it =
            std::make_shared<OwareGame<Tensor>>(*this);
        return it;
    }

    void debug() const { std::cerr << *this << "\n"; }

    float eval() const {
        float score = 0;

        const uint8_t *my_cells, *enemy_cells;
        uint8_t my_score = (id_to_play_ == 0 ? score0_ : score1_);
        uint8_t enemy_score = (id_to_play_ == 0 ? score1_ : score0_);

        if (id_to_play_ == 0) {
            my_cells = &cell0_[0];
            enemy_cells = &cell1_[0];
        } else {
            my_cells = &cell1_[0];
            enemy_cells = &cell0_[0];
        }

        score += my_score * 2;
        score -= enemy_score * 2;

        for (int i = 0; i < 6; i++) {
            if (enemy_cells[i] == 0) {
                score += 4;
            } else if (enemy_cells[i] == 2 or enemy_cells[i] == 3) {
                score += 3;
            } else if (enemy_cells[i] >= 12) {
                score += 2;
            }

            if (my_cells[i] == 0) {
                score -= 4;
            } else if (my_cells[i] == 2 or my_cells[i] == 3) {
                score -= 3;
            } else if (my_cells[i] >= 12) {
                score -= 2;
            }
        }

        return score;
    }

    uint64_t calc_hash() const {
        // remove turn number
        uint64_t C0 = state_[0] & 0xFFFFFFFFFFFFFF00ULL;

        uint64_t lower_hash = splittable64(C0);
        uint64_t upper_hash = splittable64(state_[1]);

        uint64_t rotated_upper = upper_hash << 31 | upper_hash >> 33;
        return lower_hash ^ rotated_upper;
    }

    bool equal(
        const std::shared_ptr<AbstractGame<Tensor>>& other) const override {
        if (OwareGame<Tensor>* ptr =
                dynamic_cast<OwareGame<Tensor>*>(other.get())) {
            return state_[0] == ptr->state_[0] and state_[1] == ptr->state_[1];
        }

        return false;
    }

   private:
    inline uint64_t splittable64(uint64_t x) const {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    bool have_any_legal_moves() const {
        const auto opponent_seeds = (state_[1 - id_to_play_] >> 16);
        const bool opponent_can_play = (opponent_seeds != 0);

        const uint8_t* seeds = (id_to_play_ == 0 ? &cell0_[0] : &cell1_[0]);

        if (opponent_can_play) {
            for (int i = 0; i < 6; i++) {
                if (seeds[i] > 0) {
                    return true;
                }
            }
        } else {
            for (int i = 0; i < 6; i++) {
                if (seeds[i] >= 6 - i) {
                    return true;
                }
            }
        }

        return false;
    }

    // void swap_players() {
    //     for (int col = 0; col < 6; col++) {
    //         std::swap(cell0_[col], cell1_[col]);
    //     }

    //     std::swap(score0_, score1_);
    //     id_to_play_ = 1 - id_to_play_;
    // }

    union {
        uint64_t state_[2];

        struct {
            uint8_t turn_;
            uint8_t score0_;
            uint8_t cell0_[6];

            uint8_t game_ended_ : 1;
            uint8_t id_to_play_ : 1;
            uint8_t last_move_ : 3;
            uint8_t free_bits_ : 3;
            uint8_t score1_;
            uint8_t cell1_[6];
        };
    };
};