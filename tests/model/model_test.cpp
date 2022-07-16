#include <model/model.hpp>

int main() {
    size_t input_size = 10;
    size_t output_size = 100;

    Model model(std::make_shared<InputLayer>(std::vector<size_t>{input_size}),
                std::make_shared<LinearLayer>(80, activationSIGMOID),
                std::make_shared<LinearLayer>(50, activationTANH),
                std::make_shared<LinearLayer>(50, activationRELU),
                std::make_shared<LinearLayer>(output_size));

    model.fill_random(-0.1, 0.1);
    // model.set_input(0, 1);
    model.forward();

    std::cerr << model.get_value() << "\n";
    
    float sum = 0;
    for (int i = 1; i < output_size; i++) {
        std::cerr << model.get_policy(i - 1) << " ";
        sum += model.get_policy(i - 1);
    }

    std::cerr << "\n";
    std::cerr << sum << "\n";
}