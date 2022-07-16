#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::string read_file(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

std::vector<float> to_float(std::string s) {
    int size = s.size() / sizeof(float);
    std::vector<float> res(size, 0);
    std::stringstream ss{s};

    ss.read(reinterpret_cast<char*>(res.data()), s.size());

    return res;
}

void compare(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float max_diff = 0;

    for (int i = 0; i < a.size(); i++) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }

    std::cerr << max_diff << "\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid number of arguments!\n";
        std::cerr << "Usage: ./compress_weights model out\n";
        return 1;
    }

    auto w32 = read_file(argv[1]);

    std::cerr << "File32 Size:" << w32.size() << "\n";
    auto a = to_float(w32);

    std::cerr << "weights size:" << a.size() << "\n";

    float max = *std::max_element(a.begin(), a.end());
    float min = *std::min_element(a.begin(), a.end());

    std::cerr << "MAX:" << max << "\n";
    std::cerr << "MIN:" << min << "\n";

    float norm_min = std::floor(min * 1000) / 1000.f;
    float norm_max = std::ceil(max * 1000) / 1000.f;
    std::cerr << "NORM: " << norm_min << " " << norm_max << "\n";

    float mult = 55'000 / (norm_max - norm_min);
    std::cerr << "MULT: " << mult << "\n";

    std::stringstream ss;

    float max_diff = 0;
    for (auto f : a) {
        int32_t s = (int32_t)std::round(mult * (f - norm_min)) + 255;
        char a = (char)(s & 255);
        char b = (char)((s >> 8) & 255);
        ss << b << a;

        float val = ((s - 255) / mult) + norm_min;
        max_diff = std::max(max_diff, std::abs(f - val));
    }

    std::cerr << "max diff=" << max_diff << "\n";

    std::ofstream file(argv[2]);
    file << ss.str();
    file.close();

    std::string torch_test_py = R"AAAAA(
import sys
with open(sys.argv[1], "rb") as file:
    a = file.read()
    print(f"{len(a) = }")
    b = a.decode("utf-16-be")
    print(f"{len(b) = }")

with open(sys.argv[1], "w") as file:
    file.write(b)
    )AAAAA";

    system(
        ("python3 -c '" + torch_test_py + "' " + std::string(argv[2])).c_str());

    std::fstream file2(argv[2], std::ios::in);
    std::stringstream buffer;
    buffer << file2.rdbuf();
    file2.close();
    file2.open(argv[2], std::ios::out);
    std::cerr << "Data size:" << buffer.str().size() << "\n";
    file2 << "std::wstring w = L\"" << buffer.rdbuf() << "\";";
    file2 << "float norm_min=" << norm_min << ", mult=" << mult << ";\n";
    file2 << R"AAA(
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
    int32_t s = (uint8_t)s2 + (((uint8_t)s1) << 8) - 255;
    float val = (s / mult) + norm_min;
    ss.write((char*)&val, 4);
}
)AAA";

    return 0;
}