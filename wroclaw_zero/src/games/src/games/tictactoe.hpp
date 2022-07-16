#pragma once

#include <array>
#include <cassert>
#include <iostream>

#include "abstract_game.hpp"

template <class Tensor>
class TicTacToeGame : public AbstractGame<Tensor> {
   public:
    TicTacToeGame()
        : mask({0, 0}), current_player(0), status(-1), moves_cnt(0) {}

    void calc_legal_moves() override {
        TicTacToeGame::legal_moves_cnt = 0;
        const int taken = mask[0] | mask[1];

        for (int i = 0; i < 9; i++) {
            if ((taken & (1 << i)) == 0) {
                TicTacToeGame::add_legal_move(i);
            }
        }
    }

    void make_move(int move_id) override {
        assert(0 <= move_id and move_id < 9);
        assert((mask[0] & (1 << move_id)) == 0);
        assert((mask[1] & (1 << move_id)) == 0);

        mask[current_player] ^= (1 << move_id);

        for (auto m : winning_masks) {
            if ((mask[current_player] & m) == m) {
                status = current_player;
                return;
            }
        }

        moves_cnt++;

        if (moves_cnt == 9) {
            status = 2;
        }

        current_player ^= 1;
    }

    void get_input_for_network(Tensor& input) const override {
        // Tensor input(std::vector<size_t>{18});
        input.fill(0);

        for (int i = 0; i < 9; i++) {
            if (mask[current_player] & (1 << i)) {
                input.set_element(i, 1);
            }

            if (mask[1 ^ current_player] & (1 << i)) {
                input.set_element(9 + i, 1);
            }
        }

        // return input;
    }

    std::vector<size_t> get_input_shape() const override {
        return std::vector<size_t>{18};
    }

    int get_maximum_number_of_turns() const override { return 9; }

    int get_turn_number() const override { return moves_cnt; }

    bool is_terminal() const override { return status != -1; }

    int get_game_result() const override {
        if (status == 2) return 0;
        if (status != -1) return -1;
        return -1;
    }

    float get_scaled_game_result() const { return get_game_result(); }

    int get_maximum_number_of_moves() const override { return 9; }

    std::shared_ptr<AbstractGame<Tensor>> clone() const override {
        std::shared_ptr<AbstractGame<Tensor>> it =
            std::make_shared<TicTacToeGame<Tensor>>(*this);
        return it;
    }

    bool equal(
        const std::shared_ptr<AbstractGame<Tensor>>& other) const override {
        if (TicTacToeGame<Tensor>* ptr =
                dynamic_cast<TicTacToeGame<Tensor>*>(other.get())) {
            return mask == ptr->mask and current_player == ptr->current_player;
        }

        return false;
    }

    friend std::ostream& operator<<(std::ostream& os,
                                    const TicTacToeGame& game) {
        for (int i = 0; i < 9; i++) {
            if (game.mask[0] & (1 << i)) {
                os << 'O';
            } else if (game.mask[1] & (1 << i)) {
                os << 'X';
            } else {
                os << ' ';
            }

            if (i < 7 and i % 3 == 2) {
                os << '\n';
            }
        }
        return os;
    }

    void debug() const { std::cerr << *this << "\n"; }

    float eval() const { return 0; }

    uint64_t calc_hash() const { return 0; }

   public:
    std::array<int, 2> mask;
    int current_player;
    int status;
    int moves_cnt;
    static inline const int full_mask = 0b111'111'111;

    static inline const std::array<int, 8> winning_masks = {
        0b111'000'000, 0b000'111'000, 0b000'000'111, 0b100'100'100,
        0b010'010'010, 0b001'001'001, 0b100'010'001, 0b001'010'100};
};
