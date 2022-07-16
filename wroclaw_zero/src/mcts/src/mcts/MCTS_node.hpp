#pragma once

struct MCTSNode {
    float score_sum;
    float nn_value;
    float policy;
    int visits;
    int move;
    int child_index;
    int child_count;
    int status;  //? for solver, 2 if not solved
    int children_draw;

    MCTSNode()
        : score_sum(0),
          nn_value(0),
          policy(0),
          visits(0),
          child_index(0),
          child_count(0),
          status(2),
          children_draw(0) {}

    MCTSNode(int move_)
        : score_sum(0),
          nn_value(0),
          policy(0),
          visits(0),
          move(move_),
          child_index(0),
          child_count(0),
          status(2),
          children_draw(0) {}

    MCTSNode(int move_, float policy_)
        : score_sum(0),
          nn_value(0),
          policy(policy_),
          visits(0),
          move(move_),
          child_index(0),
          child_count(0),
          status(2),
          children_draw(0) {}

    inline bool is_solved() const { return status != 2; }

    float get_value() const {
        if (status != 2) {
            return nn_value;
        }

        if (visits == 0) return 1;
        if (visits == 1) return score_sum;
        return score_sum / (float)visits;
    }
};