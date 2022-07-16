#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <games/abstract_game.hpp>
#include <iomanip>
#include <memory>
#include <model/model.hpp>
#include <random>
#include <set>
#include <vector>

#include "MCTS_config.hpp"
#include "MCTS_node.hpp"
#include "random.hpp"
#include "sample.hpp"
#include "stopwatch.hpp"

// TODO: move this to some utils or sth
inline float fastlogf(const float &x) {
    union {
        float f;
        uint32_t i;
    } vx = {x};
#pragma warning(push)
#pragma warning(disable : 4244)
    float y = vx.i;
#pragma warning(pop)
    y *= 8.2629582881927490e-8f;
    return (y - 87.989971088f);
}

// TODO: move this to some utils or sth
inline float fastinv(const int &x) {
    return _mm_cvtss_f32(_mm_rcp_ps(_mm_set_ss((float)x)));
}

// TODO: move this to some utils or sth
inline float fastsqrtf(const float &x) {
    union {
        int i;
        float x;
    } u;
    u.x = x;
    u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
    return (u.x);
}

using Game = std::shared_ptr<AbstractGame<Tensor>>;

class MCTS {
   public:
    MCTS(const Game &gamestate, MCTSConfig config) : config_(config) {
        root_gamestate_ = gamestate->clone();

        nodes_.resize(1);
        if (config.init_reserved_nodes > 0) {
            nodes_.reserve(config.init_reserved_nodes);
        }
        root_idx_ = 0;
        nodes_count_ = 1;
    }

    void reset(const Game &gamestate) {
        root_gamestate_ = gamestate->clone();
        nodes_[0] = MCTSNode();
        root_idx_ = 0;
        nodes_count_ = 1;
    }

    void search(Model &model) {
        if (nodes_[root_idx_].is_solved()) {
            return;
        }

        for (int i = 0; i < config_.number_of_iterations_per_turn; i++) {
            update(model);
        }
    }

    void search(Model &model, int ms) {
        if (nodes_[root_idx_].is_solved()) {
            return;
        }

        Stopwatch watch(ms);

        for (int i = 0; i % 10 or not watch.timeout(); i++) {
            update(model);
        }
    }

    void restore_root(int move, Model &model) {
        auto &root = nodes_[root_idx_];

        for (int i = 0; i < root.child_count; i++) {
            const int child_idx = root.child_index + i;
            auto &child = nodes_[child_idx];

            if (child.move == move) {
                root_idx_ = child_idx;
                root_gamestate_->make_move(move);
                return;
            }
        }

        root_gamestate_->make_move(move);
        reset(root_gamestate_);
    }

    void restore_root(const Game &gamestate, Model &model) {
        auto &root = nodes_[root_idx_];

        for (int i = 0; i < root.child_count; i++) {
            const int child_idx = root.child_index + i;
            auto &child = nodes_[child_idx];

            auto temp = root_gamestate_->clone();
            temp->make_move(child.move);

            if (temp->equal(gamestate)) {
                root_idx_ = child_idx;
                root_gamestate_->make_move(child.move);
                return;
            }
        }

        expansion(root_idx_, root_gamestate_, model);

        reset(gamestate);
    }

    const Game &get_root_state() const { return root_gamestate_; }

    int get_best(bool debug = false) {
        const auto &root = nodes_[root_idx_];

        assert(root.child_count > 0);

        for (int i = 0; i < root.child_count; i++) {
            const auto &node = nodes_[root.child_index + i];

            if (debug) {
                std::cerr << "move:" << node.move << " vis:" << node.visits
                          << " score:" << std::fixed << std::setprecision(5)
                          << -node.get_value() << " policy: " << node.policy
                          << "\n";
            }
        }
        if (root.is_solved()) {
            float best_score = -2;
            int best_move = -1;

            for (int i = 0; i < root.child_count; i++) {
                const auto &node = nodes_[root.child_index + i];

                if (node.is_solved()) {
                    const float score = -node.get_value();

                    if (score > best_score) {
                        best_score = score;
                        best_move = node.move;
                    }
                }
            }

            assert(best_move != -1);
            return best_move;
        }

        std::vector<float> pi(root.child_count);

        for (int i = 0; i < root.child_count; i++) {
            const auto &node = nodes_[root.child_index + i];

            if (node.status == 1) {
                //* node is solved, and it's lose
                pi[i] = 0;
            } else {
                pi[i] = node.visits;
            }

            // std::cerr << node.move << " " << node.policy << " " << pi[i]
            //           << "\n";
        }

        const int turn = root_gamestate_->get_turn_number();

        if (turn < config_.temperature_turns) {
            float temperature = config_.temperature_max;

            if (turn > 0) {
                const float decay =
                    (config_.temperature_max - config_.temperature_min) /
                    (float)(config_.temperature_turns - 1);

                temperature -= turn * decay;
            }

            temperature = 1 / temperature;

            // if (turn == 0) {
            //     for (auto p : pi) {
            //         std::cerr << p << " ";
            //     }
            //     std::cerr << "\n";
            // }

            float sum = 0;
            for (auto &p : pi) {
                p = std::pow(p, temperature);
                sum += p;
            }

            const float x = Random::instance().next_float(0, sum);
            float cur = 0;

            // if (turn == 0) {
            //     std::cerr << temperature << " " << x << '\n';
            //     for (auto p : pi) {
            //         std::cerr << p << " ";
            //     }
            //     std::cerr << "\n";
            // }

            for (int i = 0; i < root.child_count; i++) {
                cur += pi[i];

                if (x <= cur) {
                    return nodes_[root.child_index + i].move;
                }
            }

            return nodes_[root.child_index + root.child_count - 1].move;
        }

        //* temperature = 0
        const int best_child = root.child_index +
                               std::max_element(pi.begin(), pi.end()) -
                               pi.begin();

        return nodes_[best_child].move;
    }

    std::pair<int, std::string> get_cg_best() {
        const int best_move = get_best(true);

        const auto &root = nodes_[root_idx_];

        for (int i = 0; i < root.child_count; i++) {
            const auto &node = nodes_[root.child_index + i];

            if (node.move != best_move) {
                continue;
            }

            if (node.status == -1) {
                return {best_move, "WIN"};
            } else if (node.status == 0) {
                return {best_move, "DRAW"};
            } else if (node.status == 1) {
                return {best_move, "LOSE"};
            } else {
                const float score = -node.get_value();
                return {best_move, std::to_string(score)};
            }
        }

        assert(0);
    }

    void get_sample(Sample &sample) {
        // sample.input = Tensor(root_gamestate_->get_input_shape());
        root_gamestate_->get_input_for_network(sample.input);

        //* sample score = <final result> * K + <avg MCTS score> * (1 - K)
        sample.score = nodes_[root_idx_].get_value();

        for (auto &i : sample.legal_moves) {
            i = 0;
        }

        const auto &root = nodes_[root_idx_];
        for (int i = 0; i < root.child_count; i++) {
            const auto &node = nodes_[root.child_index + i];
            sample.legal_moves[node.move] = 1;
        }

        // sample.policy.resize(root_gamestate_->get_maximum_number_of_moves());

        for (auto &i : sample.policy) {
            i = 0;
        }

        if (root.is_solved()) {
            float best_score = -2;
            int best_move = -1;

            for (int i = 0; i < root.child_count; i++) {
                const auto &node = nodes_[root.child_index + i];

                if (node.is_solved()) {
                    const float score = -node.get_value();

                    if (score > best_score) {
                        best_score = score;
                        best_move = node.move;
                    }
                }
            }

            assert(best_move != -1);
            sample.policy[best_move] = 1;
        } else {
            float visits_sum = 0;

            for (int i = 0; i < root.child_count; i++) {
                const auto &node = nodes_[root.child_index + i];
                if (node.status != 1) {
                    visits_sum += node.visits;
                }
            }

            if (visits_sum == 0) {
                for (int i = 0; i < root.child_count; i++) {
                    auto &node = nodes_[root.child_index + i];
                    if (node.status != 1) {
                        node.visits = 1;
                        visits_sum += node.visits;
                    }
                }
            }

            assert(visits_sum > 0);

            for (int i = 0; i < root.child_count; i++) {
                const auto &node = nodes_[root.child_index + i];

                if (node.status != 1) {
                    sample.policy[node.move] = (float)node.visits / visits_sum;
                }
            }
        }
    }

    void debug_stats() {
        std::cerr << "MCTS STATS\n";
        std::cerr << "Nodes cnt:" << nodes_count_ << "/" << nodes_.size()
                  << "\n";
        std::cerr << "ROOT IDX: " << root_idx_ << "\n";

        auto &root = nodes_[root_idx_];

        for (int i = 0; i < root.child_count; i++) {
            const auto &node = nodes_[root.child_index + i];

            const float score = -node.get_value();

            std::cerr << root.child_index + i << " ";
            std::cerr << "move:" << node.move << " vis:" << node.visits
                      << " score:" << std::fixed << std::setprecision(5)
                      << score << " policy: " << node.policy << " ";

            std::cerr << "status:" << node.status;
            std::cerr << "\n";
        }

        std::cerr.flush();
    }

    void debug_tree(int max_depth) {
        debug_tree(root_idx_, root_gamestate_->clone(), max_depth);
    }

    void debug_select() {
        std::cerr << "DEBUG SELECT\n";

        auto &node = nodes_[root_idx_];
        const float cpuct = config_.cpuct_init;

        std::cerr << "cpuct:" << cpuct << "\n";

        const float parent_value =
            (node.visits <= 1 ? cpuct : cpuct * fastsqrtf(node.visits));

        std::cerr << "parent_value: " << parent_value << "\n";

        float best_U = -1e9f;
        int best_child = -1;

        assert(node.child_count > 0);

        for (int i = 0; i < node.child_count; i++) {
            auto &child = nodes_[node.child_index + i];

            if (child.status == 1) {
                // skip solved lose nodes
                continue;
            }

            const float Q = -child.get_value();
            const float P =
                parent_value * child.policy * fastinv(1 + child.visits);

            const float U = Q + P;
            std::cerr << "Q=" << Q << " P=" << P << "\n";

            if (U > best_U) {
                best_U = U;
                best_child = i;
            }
        }

        // if (best_child == -1) {
        //     std::cerr << node_idx << " " << root_idx_ << "\n";

        //     for (int i = 0; i < node.child_count; i++) {
        //         auto &child = nodes_[node.child_index + i];
        //         std::cerr << child.status << "\n";

        //         const float Q = -child.get_value();
        //         std::cerr << Q << "\n";
        //         const float P =
        //             parent_value * child.policy * fastinv(1 +
        //             child.visits);

        //         std::cerr << node.child_index + i << "\n";
        //         std::cerr << parent_value << " " << child.policy << "\n";
        //         const float U = Q + P;
        //         std::cerr << U << "\n";
        //     }
        // }
    }

   private:
    void debug_tree(int node_idx, Game state, int d) {
        auto &root = nodes_[node_idx];

        std::cerr << "Node(" << node_idx << ") val: " << root.nn_value << "\n";
        // std::cerr << root.child_index << " " << root.child_count << "\n";
        state->debug();

        if (d == 0) return;

        for (int i = 0; i < root.child_count; i++) {
            const auto &node = nodes_[root.child_index + i];

            auto temp = state->clone();
            temp->make_move(node.move);

            debug_tree(root.child_index + i, temp, d - 1);
        }
    }

    void update(Model &model) {
        const int node_idx = select();

        if (nodes_[node_idx].is_solved()) {
            backpropagation();
            return;
        }

        Game gamestate = root_gamestate_->clone();

        for (int i = 1; i < selected_nodes_cnt_; i++) {
            gamestate->make_move(nodes_[selected_nodes_[i]].move);
        }

        expansion(node_idx, gamestate, model);

        backpropagation();
    }

    void expansion(uint32_t node_idx, Game &current_gamestate, Model &model) {
        if (current_gamestate->is_terminal()) {
            auto &node = nodes_[node_idx];
            node.nn_value = current_gamestate->get_scaled_game_result();
            node.status = current_gamestate->get_game_result();
            node.child_count = 0;
            node.child_index = 0;
        } else {
            /*
                !calculate legal moves before forward!
                Model will clear policy of not legal moves
            */
            current_gamestate->calc_legal_moves();
            // std::cerr << "LEGAL: " << current_gamestate->legal_moves_cnt <<
            // "\n";

            current_gamestate->get_input_for_network(
                model.input_layer->get_output());

            // std::cerr << *model.input_layer->output << "\n";
            model.forward(current_gamestate);

            // std::cerr << "MCTS OUT:" << *model.output << "\n";

            auto &node = nodes_[node_idx];
            node.nn_value = model.get_value();

            node.child_count = current_gamestate->legal_moves_cnt;
            node.child_index = nodes_count_;

            const int cnt = node.child_count;
            // std::cerr << "EXPAND\n";
            // std::cerr << *model.output << "\n";

            for (int i = 0; i < cnt; i++) {
                const int move_idx = current_gamestate->legal_moves[i];

                if (nodes_count_ == nodes_.size()) {
                    // std::cerr << "[CPP] Increase reserved nodes in MCTS!\n";
                    nodes_.emplace_back(move_idx, model.get_policy(move_idx));
                } else {
                    nodes_[nodes_count_] =
                        MCTSNode(move_idx, model.get_policy(move_idx));
                }

                // std::cerr << nodes_[nodes_count_].policy << "\n";

                nodes_count_++;
            }
        }

        if (node_idx == root_idx_) {
            add_dirichlet_noise(node_idx, config_.dirichlet_noise_epsilon,
                                config_.dirichlet_noise_alpha);
        }
    }

    void add_dirichlet_noise(uint32_t node_idx, float epsilon, float alpha) {
        if (epsilon < 0.001f) {
            return;
        }

        if (alpha < 0.001f) {
            return;
        }

        auto &node = nodes_[node_idx];

        if (node.child_count == 0) {
            return;
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::vector<float> dirichlet_vector(node.child_count);
        std::gamma_distribution<float> gamma(alpha, 1.0f);

        float factor_dirich = 0;
        for (auto &i : dirichlet_vector) {
            i = gamma(gen);
            factor_dirich += i;
        }

        if (factor_dirich < std::numeric_limits<float>::min()) {
            return;
        }

        factor_dirich = epsilon / factor_dirich;

        for (int i = 0; i < node.child_count; i++) {
            auto &child = nodes_[node.child_index + i];

            // std::cerr << "before " << child.policy << " ";
            child.policy = child.policy * (1.f - epsilon) +
                           dirichlet_vector[i] * factor_dirich;
            // std::cerr << "after " << child.policy << "\n";
        }
    }

    float max_solved_child_nn_value(int node_idx) {
        const auto &node = nodes_[node_idx];

        float value = -1;

        for (int i = 0; i < node.child_count; i++) {
            const auto &child = nodes_[node.child_index + i];

            if (not child.is_solved()) {
                continue;
            }

            value = std::max(value, -child.nn_value);
        }

        return value;
    }

    void backpropagation() {
        const int node_idx = selected_nodes_[selected_nodes_cnt_ - 1];

        if (not nodes_[node_idx].is_solved()) {
            float score = nodes_[node_idx].nn_value;

            for (int i = selected_nodes_cnt_ - 1; i >= 0; i--) {
                auto &node = nodes_[selected_nodes_[i]];
                node.visits += 1;
                node.score_sum += score;

                score = -score;
            }

            return;
        }

        //* node is solved
        auto &solved_node = nodes_[node_idx];
        solved_node.visits++;
        float score = -solved_node.nn_value;
        int status = -solved_node.status;

        for (int i = selected_nodes_cnt_ - 2; i >= 0; i--) {
            const int cur_idx = selected_nodes_[i];
            auto &node = nodes_[cur_idx];
            node.visits++;

            if (status == 2) {
                node.score_sum += score;
                score = -score;
                continue;
            }

            if (status == 1) {
                node.status = 1;
                node.nn_value = score;
                status = -1;
                score = -score;
                continue;
            }

            if (status == 0) {
                node.children_draw = 1;
            }

            // calc number of solved children
            int solved = 0;
            for (int i = 0; i < node.child_count; i++) {
                const auto &child = nodes_[node.child_index + i];

                solved += child.is_solved();
            }

            if (solved == node.child_count) {
                node.status = status;
                node.nn_value = max_solved_child_nn_value(cur_idx);
                score = -node.nn_value;
                status = -status;
            } else {
                //* node isn't solved yet
                node.score_sum += score;
                status = 2;
                score = -score;
            }
        }
    }

    int select() {
        int node_idx = root_idx_;
        selected_nodes_cnt_ = 0;

        while (true) {
            if (selected_nodes_.size() == selected_nodes_cnt_) {
                selected_nodes_.emplace_back(0);
            }

            selected_nodes_[selected_nodes_cnt_++] = node_idx;

            auto &node = nodes_[node_idx];

            if (node.is_solved()) {
                return node_idx;
            }

            // std::cerr << "node_idx: " << node_idx;
            // std::cerr << " child_count:" << node.child_count;
            // std::cerr << " child_index:" << node.child_index;
            // std::cerr << " visits:" << node.visits;
            // std::cerr << "\n";

            if (node.child_count == 0) {
                return node_idx;
            }

            if (node.child_count == 1) {
                node_idx = node.child_index;
                continue;
            }

            const float cpuct = config_.cpuct_init;

            // std::cerr << cpuct << "\n";

            const float parent_value =
                (node.visits <= 1 ? cpuct : cpuct * fastsqrtf(node.visits));

            // std::cerr << "parent_value: " << parent_value << "\n";

            float best_U = -1e9f;
            int best_child = -1;

            assert(node.child_count > 0);

            for (int i = 0; i < node.child_count; i++) {
                auto &child = nodes_[node.child_index + i];

                if (child.status == 1) {
                    // skip solved lose nodes
                    continue;
                }

                const float Q = -child.get_value();
                const float P =
                    parent_value * child.policy * fastinv(1 + child.visits);

                const float U = Q + P;

                if (U > best_U) {
                    best_U = U;
                    best_child = i;
                }
            }

            // if (best_child == -1) {
            //     std::cerr << node_idx << " " << root_idx_ << "\n";

            //     for (int i = 0; i < node.child_count; i++) {
            //         auto &child = nodes_[node.child_index + i];
            //         std::cerr << child.status << "\n";

            //         const float Q = -child.get_value();
            //         std::cerr << Q << "\n";
            //         const float P =
            //             parent_value * child.policy * fastinv(1 +
            //             child.visits);

            //         std::cerr << node.child_index + i << "\n";
            //         std::cerr << parent_value << " " << child.policy << "\n";
            //         const float U = Q + P;
            //         std::cerr << U << "\n";
            //     }
            // }

            assert(best_child != -1);
            node_idx = node.child_index + best_child;
        }

        assert(0);
    }

    MCTSConfig config_;
    std::vector<MCTSNode> nodes_;
    size_t nodes_count_;
    std::vector<uint32_t> selected_nodes_;  // used in backpropagation
    int selected_nodes_cnt_;
    uint32_t root_idx_;
    Game root_gamestate_;
};