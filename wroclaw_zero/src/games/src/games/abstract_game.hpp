#pragma once

#include <cstdint>
#include <memory>
#include <vector>

template <class Tensor>
class AbstractGame {
   public:
    virtual void calc_legal_moves() = 0;
    virtual void make_move(int move_id) = 0;
    virtual void get_input_for_network(Tensor& input) const = 0;
    virtual std::vector<size_t> get_input_shape() const = 0;
    virtual int get_maximum_number_of_turns() const = 0;
    virtual int get_turn_number() const = 0;
    virtual bool is_terminal() const = 0;
    virtual int get_game_result() const = 0;
    virtual float get_scaled_game_result() const = 0;
    virtual int get_maximum_number_of_moves() const = 0;
    virtual std::shared_ptr<AbstractGame<Tensor>> clone() const = 0;
    virtual void debug() const = 0;
    virtual float eval() const = 0;
    virtual uint64_t calc_hash() const = 0;
    virtual bool equal(
        const std::shared_ptr<AbstractGame<Tensor>>& other) const = 0;

    thread_local static inline std::vector<int> legal_moves;
    thread_local static inline int legal_moves_cnt;

    virtual void add_legal_move(int move) {
        if (legal_moves.size() == legal_moves_cnt) {
            legal_moves.push_back(move);
            legal_moves_cnt++;
        } else {
            legal_moves[legal_moves_cnt++] = move;
        }
    }
};