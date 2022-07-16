#pragma once

#include <x86intrin.h>

#include <cassert>
#include <iostream>

#include "abstract_game.hpp"

template <class Tensor>
class Connect4Game : public AbstractGame<Tensor> {
   public:
    static inline constexpr int WIDTH = 9;
    static inline constexpr int HEIGHT = 7;
    static inline constexpr uint64_t COLUMN_MASK = (1 << HEIGHT) - 1;

    static inline constexpr uint64_t ROW_MASK =
        0b0000001'0000001'0000001'0000001'0000001'0000001'0000001'0000001'0000001;

    static inline constexpr uint64_t VERTICAL =
        0b0001111'0001111'0001111'0001111'0001111'0001111'0001111'0001111'0001111;

    static inline constexpr uint64_t DIAGONAL1 =
        0b0000000'0000000'0000000'0001111'0001111'0001111'0001111'0001111'0001111;

    static inline constexpr uint64_t DIAGONAL2 =
        0b0001111'0001111'0001111'0001111'0001111'0001111'0000000'0000000'0000000;

    static inline constexpr uint64_t WITHOUT_TOP =
        0b0111111'0111111'0111111'0111111'0111111'0111111'0111111'0111111'0111111;

    Connect4Game()
        : my_mask_(0),
          opp_mask_(0),
          cells_empty_(WIDTH * HEIGHT),
          turn_(0),
          game_ended_(0) {}

    void read() {
        int turn;
        std::cin >> turn;
        std::cin.ignore();
        turn_ = turn;
        my_mask_ = opp_mask_ = 0;

        for (int y = HEIGHT - 1; y >= 0; y--) {
            std::string row;  // one row of the board (from top to bottom)
            std::cin >> row;
            std::cin.ignore();

            for (int x = 0; x < WIDTH; x++) {
                const uint64_t mask = get_cell_mask(x, y);

                if (row[x] == '0') {
                    my_mask_ |= mask;
                } else if (row[x] == '1') {
                    opp_mask_ |= mask;
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& os,
                                    const Connect4Game& game) {
        for (int y = HEIGHT - 1; y >= 0; y--) {
            for (int x = 0; x < WIDTH; x++) {
                const uint64_t mask = game.get_cell_mask(x, y);

                if (game.my_mask_ & mask) {
                    os << '0';
                } else if (game.opp_mask_ & mask) {
                    os << '1';
                } else {
                    os << '.';
                }
            }

            os << '\n';
        }

        return os;
    }

    void calc_legal_moves() override {
        Connect4Game::legal_moves_cnt = 0;

        uint64_t possible_mask = possible();
        uint64_t winning_mask = compute_my_winning() & possible_mask;

        if (winning_mask > 0) {
            for (int x = 0; x < WIDTH; x++) {
                if ((winning_mask >> (x * HEIGHT)) & COLUMN_MASK) {
                    Connect4Game::add_legal_move(x);
                    return;
                }
            }

            assert(0);
        }

        if (turn_ == 1 and cells_empty_ == 62) {
            Connect4Game::add_legal_move(WIDTH);
        }

        uint64_t opponent_win = compute_opponent_winning();
        uint64_t forced_moves = possible_mask & opponent_win;

        if (forced_moves) {
            for (int x = 0; x < WIDTH; x++) {
                if ((forced_moves >> (x * HEIGHT)) & COLUMN_MASK) {
                    Connect4Game::add_legal_move(x);
                    // if there is more than one forced move, we lost
                    return;
                }
            }
        }

        // avoid to play below an opponent winning spot
        uint64_t safe_mask = possible_mask & ~((opponent_win & ~ROW_MASK) >> 1);

        if (safe_mask) {
            for (int x = 0; x < WIDTH; x++) {
                if ((safe_mask >> (x * HEIGHT)) & COLUMN_MASK) {
                    Connect4Game::add_legal_move(x);
                }
            }

            return;
        }

        for (int x = 0; x < WIDTH; x++) {
            if ((possible_mask >> (x * HEIGHT)) & COLUMN_MASK) {
                Connect4Game::add_legal_move(x);
            }
        }
    }

    void make_move(int move) override {
        if (move != WIDTH) [[likely]] {
            const uint64_t mask =
                get_cell_mask(move, _mm_popcnt_u64(get_column_mask(move)));

            my_mask_ ^= mask;

            if (won()) {
                game_ended_ = true;
            }

            std::swap(my_mask_, opp_mask_);
            cells_empty_--;
        }

        turn_++;
    }

    int get_maximum_number_of_turns() const override {
        return WIDTH * HEIGHT + 1;
    }

    int get_turn_number() const override { return (int)turn_; }

    bool is_terminal() const override {
        return game_ended_ or (int) cells_empty_ == 0;
    }

    int get_maximum_number_of_moves() const override { return WIDTH + 1; }

    int get_game_result() const override {
        if (game_ended_) {
            return -1;
        }

        return 0;
    }

    float get_scaled_game_result() const {
        if (game_ended_) {
            const float turn_diff = WIDTH * HEIGHT - (int)turn_;
            // const float MAX = WIDTH * HEIGHT * 0.0023f + 0.85;
            return -0.85 - turn_diff * 0.0023f;
        }

        return 0;
    }

    void get_input_for_network(Tensor& input) const override {
        input.fill(0);

        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                const uint64_t mask = get_cell_mask(x, y);
                if (my_mask_ & mask) {
                    input.set_element3D(0, x, y, 1);
                } else if (opp_mask_ & mask) {
                    input.set_element3D(1, x, y, 1);
                }
            }
        }
    }

    std::vector<size_t> get_input_shape() const override {
        return std::vector<size_t>{2, WIDTH, HEIGHT};
    }

    std::shared_ptr<AbstractGame<Tensor>> clone() const override {
        std::shared_ptr<AbstractGame<Tensor>> it =
            std::make_shared<Connect4Game<Tensor>>(*this);
        return it;
    }

    void debug() const { std::cerr << *this << "\n"; }

    float eval() const {
        abort();
        return 0;
    }

    uint64_t calc_hash() const {
        abort();
        return 0;
    }

    bool equal(
        const std::shared_ptr<AbstractGame<Tensor>>& other) const override {
        if (Connect4Game<Tensor>* ptr =
                dynamic_cast<Connect4Game<Tensor>*>(other.get())) {
            return my_mask_ == ptr->my_mask_ and opp_mask_ == ptr->opp_mask_ and
                   cells_empty_ == ptr->cells_empty_ and turn_ == ptr->turn_ and
                   game_ended_ == ptr->game_ended_;
        }

        return false;
    }

   public:
    inline uint64_t get_cell_id(int x, int y) const { return x * HEIGHT + y; }

    inline uint64_t get_cell_mask(int x, int y) const {
        return 1ULL << get_cell_id(x, y);
    }

    inline uint64_t get_column_mask(int x) const {
        return ((my_mask_ | opp_mask_) >> get_cell_id(x, 0)) & COLUMN_MASK;
    }

    bool is_legal(int x) const {
        return not(((my_mask_ | opp_mask_) >> get_cell_id(x, HEIGHT - 1)) & 1);
    }

    inline uint64_t possible() const {
        const uint64_t mask = my_mask_ | opp_mask_;
        return (((mask & WITHOUT_TOP) << 1) | ROW_MASK) & ~mask;
    }

    inline uint64_t compute_winning(uint64_t mask) const {
        uint64_t result = 0;

        // vertical
        result = (mask << 1) & (mask << 2) & (mask << 3) & (VERTICAL << 3);

        // horizontal
        uint64_t temp = (mask << HEIGHT) & (mask << (HEIGHT * 2));
        result |= temp & (mask << (HEIGHT * 3));  // _111
        result |= temp & (mask >> HEIGHT);        // 1_11
        temp = (mask >> HEIGHT) & (mask >> (HEIGHT * 2));
        result |= temp & (mask >> (HEIGHT * 3));  // 111_
        result |= temp & (mask << HEIGHT);        // 11_1

        // diagonal1
        temp = (mask >> (HEIGHT + 1)) & (mask >> (2 * (HEIGHT + 1)));
        result |= temp & (mask >> (3 * (HEIGHT + 1))) & DIAGONAL1;  // _111
        result |= temp & (mask << (HEIGHT + 1)) &
                  (DIAGONAL1 << (HEIGHT + 1));  // 1_11

        temp = (mask << (HEIGHT + 1)) & (mask << (2 * (HEIGHT + 1)));
        result |= temp & (mask << (3 * (HEIGHT + 1))) &
                  (DIAGONAL1 << (3 * (HEIGHT + 1)));  // 111_
        result |= temp & (mask >> (HEIGHT + 1)) &
                  (DIAGONAL1 << (2 * (HEIGHT + 1)));  // 111_

        // diagonal2
        temp = (mask << (HEIGHT - 1)) & (mask << (2 * (HEIGHT - 1)));
        result |= temp & (mask << (3 * (HEIGHT - 1))) & DIAGONAL2;  // 111_
        result |= temp & (mask >> (HEIGHT - 1)) &
                  (DIAGONAL2 >> (HEIGHT - 1));  // 11_1

        temp = (mask >> (HEIGHT - 1)) & (mask >> (2 * (HEIGHT - 1)));
        result |= temp & (mask >> (3 * (HEIGHT - 1))) &
                  (DIAGONAL2 >> (3 * (HEIGHT - 1)));  // _111

        result |= temp & (mask << (HEIGHT - 1)) &
                  (DIAGONAL2 >> (2 * (HEIGHT - 1)));  // 1_11

        return result;
    }

    inline uint64_t compute_my_winning() const {
        return compute_winning(my_mask_);
    }

    inline uint64_t compute_opponent_winning() const {
        return compute_winning(opp_mask_);
    }

    bool won() const {
        uint64_t mask = my_mask_;

        if ((mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & VERTICAL) != 0) {
            return true;
        }

        if ((mask & (mask >> HEIGHT) & (mask >> (HEIGHT * 2)) &
             (mask >> (HEIGHT * 3))) != 0) {
            return true;
        }

        if ((mask & (mask >> (HEIGHT + 1)) & (mask >> (2 * (HEIGHT + 1))) &
             (mask >> (3 * (HEIGHT + 1))) & DIAGONAL1) != 0) {
            return true;
        }

        if ((mask & (mask << (HEIGHT - 1)) & (mask << (2 * (HEIGHT - 1))) &
             (mask << (3 * (HEIGHT - 1))) & DIAGONAL2) != 0) {
            return true;
        }

        return false;
    }

    uint64_t my_mask_, opp_mask_;
    uint32_t cells_empty_;
    uint16_t turn_;
    uint16_t game_ended_;
};