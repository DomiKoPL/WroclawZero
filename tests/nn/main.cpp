#include <chrono>
#include <fstream>
#include <games/oware.hpp>
#include <iostream>
#include <nn_avx_fast/common.hpp>
#include <sstream>
#include <string>

using namespace nn_avx_fast;

#define NOW() std::chrono::high_resolution_clock::now()

class Stopwatch {
   public:
    Stopwatch(uint64_t ms) { start(ms); }

    void start(uint64_t ms) {
        c_time = NOW();
        c_timeout = c_time + std::chrono::milliseconds(ms);
    }

    void set_timeout(uint64_t ms) {
        c_timeout = c_time + std::chrono::milliseconds(ms);
    }

    inline bool timeout() { return NOW() > c_timeout; }

    long long elapsed_milliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(NOW() -
                                                                     c_time)
            .count();
    }

   private:
    std::chrono::high_resolution_clock::time_point c_time, c_timeout;
};

int main() {
    std::wstring w = L"No model for you :D";
    float norm_min = -22.026, mult = 1837.68;

    std::string decoded = "";
    for (auto c : w) {
        decoded += (char)(c >> 8);
        decoded += (char)(c & 255);
    }

    std::stringstream ss2(decoded), ss;
    for (int i = 0; i < decoded.size(); i += 2) {
        char s1, s2;
        ss2.get(s1);
        ss2.get(s2);
        uint32_t s = (uint8_t)s2 + (((uint8_t)s1) << 8);
        float val = (s / mult) + norm_min;
        ss.write((char*)&val, 4);
    }

    auto input_layer = std::make_shared<InputLayer>(std::vector<size_t>{342});
    auto layer1 =
        std::make_shared<LinearLayer>(64, input_layer, activationRELU);
    auto layer2 = std::make_shared<LinearLayer>(64, layer1, activationRELU);
    auto layer3 = std::make_shared<LinearLayer>(64, layer2, activationRELU);
    auto layer4 = std::make_shared<LinearLayer>(7, layer3, activationSOFTMAX);

    layer1->load(ss);
    layer2->load(ss);
    layer3->load(ss);
    layer4->load(ss);

    OwareGame<Tensor> game;
    game.get_input_for_network(input_layer->get_output());

    Stopwatch watch(100);

    watch.start(0);
    for (int i = 0; i < 100'000; i++) {
        game.calc_legal_moves();
        game.get_input_for_network(input_layer->get_output());
        // model.forward(game);
        std::cerr << layer4->get_output() << "\n";
    }
    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    watch.start(0);
    for (int i = 0; i < 100'000; i++) {
        layer1->forward();
        layer2->forward();
        layer3->forward();
        layer4->forward();
    }
    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    watch.start(0);
    for (int i = 0; i < 100'000; i++) {
        game.calc_legal_moves();
        game.get_input_for_network(input_layer->get_output());
        layer1->forward();
        layer2->forward();
        layer3->forward();
        layer4->forward();
    }
    std::cerr << "TIME: " << watch.elapsed_milliseconds() << "ms\n";

    // try {
    //     size_t input_size = 100;
    //     size_t output_size = 30;
    //     auto input_layer =
    //         std::make_shared<InputLayer>(std::vector<size_t>{input_size});

    //     auto model = Sequential(
    //         input_layer, std::make_shared<LinearLayer>(80),
    //         std::make_shared<Sigmoid>(), std::make_shared<LinearLayer>(50),
    //         std::make_shared<Tanh>(), std::make_shared<LinearLayer>(50),
    //         std::make_shared<ReLu>(),
    //         std::make_shared<LinearLayer>(output_size),
    //         std::make_shared<Softmax>());

    //     std::ifstream file("models/model.weights", std::ios::binary);
    //     model.load(file);
    //     file.close();

    //     for (size_t i = 0; i < input_size; i++) {
    //         input_layer->set(
    //             i, static_cast<float>(i) / static_cast<float>(input_size));
    //     }

    //     model.forward();

    //     std::cerr << "OUTPUT: " << *model.output << "\n";

    //     std::ifstream out_file("models/model.out", std::ios::binary);
    //     Tensor output({output_size});
    //     output.load(out_file);
    //     out_file.close();

    //     std::cerr << "OUTPUT: " << output << "\n";

    //     auto diff = *model.output;
    //     diff -= output;

    //     float max_diff = 0.f;

    //     for (size_t i = 0; i < output_size; i++) {
    //         max_diff = std::max(max_diff, abs(diff.get_element(i)));
    //     }

    //     std::cerr << "MAX DIFF: " << max_diff << "\n";
    // } catch (const std::exception &e) {
    //     std::cerr << e.what() << "\n";
    // }
}