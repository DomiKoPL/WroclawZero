#include <games/oware.hpp>
#include <model/model.hpp>



int main() {
    OwareGame<Tensor> game;

    // std::cerr << game.get_input_for_network() << "\n";

    game.read();

    while (not game.is_terminal()) {
        std::cerr << "GAME STATE\n";
        std::cerr << game << "\n";
        
        game.calc_legal_moves();
        std::vector<float> val;

        std::cerr << "legal\n";
        for (int i = 0; i < game.legal_moves_cnt; i++) {
            const int move = game.legal_moves[i];
            auto temp = game;
            temp.make_move(move);
            val.push_back(-temp.eval());
            std::cerr << move << ", ";
        }
        std::cerr << "\n";

        int best = std::max_element(val.begin(), val.end()) - val.begin();
        int move = game.legal_moves[best];
        game.make_move(move);
        
        // std::cerr << game << "\n";
        std::cerr << "CHOSEN MOVE:";
        std::cout << move << "\n";

        static int cnt = 0;
        cnt++;
        if (cnt == 5) break;
    }
    
    std::cerr << game << "\n";
}