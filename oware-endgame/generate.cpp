// MSmits code
// https://www.codingame.com/playgrounds/58572/endgame-books-in-oware-abapa

#pragma GCC optimize("Ofast", "unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC target("rdrnd", "popcnt", "avx", "bmi2")

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

using namespace std;

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_ctz ctz
#define __builtin_ctzl ctzl
#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64

inline uint32_t ctz(uint32_t x) {
    unsigned long r = 0;
    _BitScanForward(&r, x);
    return r;
}

inline uint64_t ctzl(uint64_t x) {
    unsigned long r = 0;
    _BitScanForward64(&r, x);
    return r;
}

#endif

const uint64_t START_BOARD = 0x210842108421084;
const int END_GAME_SEEDS = 9;
const int SEED_9_STATECOUNT = 293929;
const int PATTERN_LIMIT_9 = 146;
auto start = std::chrono::high_resolution_clock::now();
uint64_t sowing[384] = {0};
uint32_t capturing[384] = {0};
uint8_t nextHouse[144] = {0};
uint64_t stateCounts[END_GAME_SEEDS + 1][END_GAME_SEEDS + 1][12] = {0};

uint64_t FlipBoard(uint64_t board) {
    uint64_t p1 = board & 0x3FFFFFFF;
    uint64_t p2 = (board >> 30) & 0x3FFFFFFF;
    uint64_t extra = (board >> 60);
    return extra << 60 | p1 << 30 | p2;
}

int SeedsOnBoard(uint64_t board) {
    int total = board >> 60;

    for (int i = 0; i < 60; i = i + 5) total += (board >> i) & 31;

    return total;
}

uint64_t StateCounter(int64_t pits, int64_t seeds) {
    int64_t top = 1;
    int64_t bottom = 1;
    if (pits > seeds) {
        for (int64_t i = seeds + 1; i <= seeds + pits - 1; i++) top *= i;

        for (int64_t i = 2; i <= (pits - 1); i++) bottom *= i;
    } else {
        for (int64_t i = pits; i <= seeds + pits - 1; i++) top *= i;

        for (int64_t i = 2; i <= seeds; i++) bottom *= i;
    }

    return top /= bottom;
}

uint64_t IndexFunction(
    uint64_t state,
    int total)  // assume state has pits with 31 seeds max, spaced as 5 bit
{
    uint64_t index = 0;
    int left = total;

    for (int house = 0; house < 11; house++) {
        int seeds = 31 & (state >> (house * 5));
        index += stateCounts[left][seeds][house];
        left -= seeds;
    }
    return index;
}

void FillStateCountLookups() {
    for (int left = 0; left <= END_GAME_SEEDS; left++) {
        for (int seeds = 0; seeds <= min(left, END_GAME_SEEDS); seeds++) {
            for (int house = 0; house < 12; house++) {
                uint64_t index = 0;
                for (int j = 0; j < seeds; j++) {
                    uint64_t stateCount = StateCounter(11 - house, left - j);
                    index += stateCount;
                }

                stateCounts[left][seeds][house] = index;
            }
        }
    }
}

void SowingArray() {
    for (int i = 0; i < 384; i++) {
        uint64_t changes[12] = {0};
        int houseOrigin = i % 12;
        int seeds = i / 12;
        int house = houseOrigin;
        while (seeds > 0) {
            house++;
            if (house == 12) house = 0;
            if (house == houseOrigin) continue;
            changes[house]++;
            seeds--;
        }
        for (int j = 0; j < 12; j++) sowing[i] += changes[j] << (j * 5);
        if (houseOrigin < 6 && house < 6 || houseOrigin > 5 && house > 5)
            sowing[i] |= 6ULL << 60;
        else
            sowing[i] |= (uint64_t)(house % 6) << 60;
    }
}

void CaptureArray() {
    for (int i = 0; i < 384; i++) {
        int house = i >> 6;
        uint8_t code = 0;
        for (int j = house; j >= 0; j--) {
            if (i & (1 << j))
                code |= 1 << j;
            else
                break;
        }
        capturing[i] = code;
    }
}

void NextHouseArray() {
    for (int i = 0; i < 144; i++) {
        int origin = i / 12;
        int current = i % 12;

        current++;
        if (current > 11) current = 0;

        if (origin == current) {
            current++;
            if (current > 11) current = 0;
        }
        nextHouse[i] = current;
    }
}

inline uint64_t BoardFromHouseArray(int houses[]) {
    uint64_t board = 0;
    for (int i = 0; i < 12; i++) {
        if (houses[i] > 31) {
            board |= 31ULL << (5 * i);
            board |= ((uint64_t)(houses[i] - 31)) << 60;
        } else
            board |= ((uint64_t)houses[i]) << (5 * i);
    }

    return board;
}

inline int ApplyNoCheck(int move, uint64_t& board, int player) {
    // careful this function does not work if a pit has > 31 seeds. That's never
    // the case for endgame books
    int opponent = player ^ 1;
    int shift = 5 * move + 30 * player;
    int originSeeds = (board >> shift) & 31;
    uint64_t sowingLookup = sowing[12 * originSeeds + move + 6 * player];
    board += sowingLookup & 0xFFFFFFFFFFFFFFF;
    board &= ~(31ULL << shift);
    int lastHouse = sowingLookup >> 60;
    if (lastHouse == 6) return 0;

    int opponentShift = 30 * opponent;

    uint32_t oppBoard = board >> opponentShift;
    uint32_t bit2 = oppBoard & 0x4210842;
    uint32_t bit3 = (oppBoard & 0x8421084) >> 1;
    uint32_t bit4 = (oppBoard & 0x10842108) >> 2;
    uint32_t bit5 = (oppBoard & 0x21084210) >> 3;
    uint32_t combined = bit2 & ~bit3 & ~bit4 & ~bit5;

    if (combined == 0) return 0;

    uint32_t captureKey = lastHouse << 6 | _pext_u32(combined, 0x4210842);
    uint8_t captureLookup = capturing[captureKey];
    uint32_t selectedBits = _pdep_u32(captureLookup, 0x4210842);
    uint32_t selectedExtended = selectedBits | selectedBits >> 1;
    uint32_t allCapturedBits = selectedExtended & oppBoard;

    uint32_t tempBoard =
        oppBoard & ~allCapturedBits;  // check if the board doesn't become empty

    if (tempBoard == 0)
        return 0;
    else {
        board &= ~(0x3FFFFFFFULL << opponentShift);
        board |= (uint64_t)tempBoard << opponentShift;
        int bit2Count = __builtin_popcount(selectedBits);
        int bit12Count = __builtin_popcount(allCapturedBits);
        return bit2Count + bit12Count;
    }
}

inline bool PlayerHasSeeds(int player, uint64_t board) {
    if (((board >> (30 * player)) & 0x3FFFFFFF) == 0)
        return false;
    else
        return true;
}

inline bool HouseHasSeeds(int houseIndex, int playerIndex, uint64_t board) {
    int houseShift = 5 * (6 * playerIndex + houseIndex);
    return (board & (31ULL << houseShift)) != 0;
}

inline int GetPlayerSeeds(int player, uint64_t board) {
    uint32_t playerBoard = board >> (30 * player);
    int playerSeeds = 0;
    for (int i = 0; i < 6; i++) playerSeeds += (playerBoard >> (5 * i)) & 31;

    return playerSeeds;
}

struct SimCache {
    uint64_t board;
    int childIndexBuffer[6];
    int8_t capturedBuffer[6];
    int8_t moveCount;
    int8_t currentSeeds;
    int8_t moveHouse[6];

    SimCache(){};
};

SimCache buffers[SEED_9_STATECOUNT];
int stateCount = 0;
int8_t endSeeds[SEED_9_STATECOUNT * PATTERN_LIMIT_9];
int arrayStarts[END_GAME_SEEDS + 1];

inline int GetSeedScore(uint64_t childBoard, int turnsLeft, int seedsOnBoard) {
    if (turnsLeft > 146) turnsLeft = 146;
    int stateIndex =
        arrayStarts[seedsOnBoard] + IndexFunction(childBoard, seedsOnBoard);
    int index = (turnsLeft - 1) * stateCount + stateIndex;

    return endSeeds[index];
}

inline void SetSeedScore(int stateIndex, int turnsLeft, int seeds) {
    endSeeds[(turnsLeft - 1) * stateCount + stateIndex] = seeds;
}

void GenerateStates(int seedsLeft, uint64_t board, int house, int current) {
    if (house < 11) {
        for (int i = 0; i <= seedsLeft; i++)
            GenerateStates(seedsLeft - i, board | (uint64_t)i << (house * 5),
                           house + 1, current);
    } else {
        board |= (uint64_t)seedsLeft << 55;
        buffers[stateCount].currentSeeds = current;
        buffers[stateCount++].board = board;
    }
}

struct Sample {
    std::vector<float> input;
    std::vector<float> policy;
    float value;
};

std::ofstream file;

void print_sample(int s, int turn, int bestScore) {
    Sample sample;
    sample.input.resize(24 * 12 + 2 * 27);

    auto board = buffers[s].board;

    for (int i = 0; i < 12; i++) {
        int seeds = (board >> (i * 5)) & 0b11111;
        sample.input[seeds + 24 * i] = 1;
    }

    sample.policy.resize(6, 0);

    int sum = 0;
    for (int m = 0; m < buffers[s].moveCount; m++) {
        int captured = buffers[s].capturedBuffer[m];
        int childIndex = buffers[s].childIndexBuffer[m];
        int score = captured - endSeeds[(turn - 2) * stateCount + childIndex];
        if (bestScore == score) {
            sample.policy[buffers[s].moveHouse[m]] = 1;
            sum += 1;
        }
    }

    if (sum > 1) {
        for (auto& i : sample.policy) {
            i /= sum;
        }
    }

    if (bestScore > 0) {
        sample.value = 1;
    } else if (bestScore == 0) {
        sample.value = 0;
    } else {
        sample.value = -1;
    }

    file.write(reinterpret_cast<const char*>(sample.input.data()),
                    sizeof(float) * sample.input.size());
    file.write(reinterpret_cast<const char*>(sample.policy.data()),
                    sizeof(float) * sample.policy.size());
    file.write(reinterpret_cast<const char*>(&sample.value),
                    sizeof(float));
}

void GenerateBook() {
    stateCount = 0;

    for (int currentSeeds = 1; currentSeeds <= END_GAME_SEEDS; currentSeeds++) {
        arrayStarts[currentSeeds] = stateCount;
        GenerateStates(currentSeeds, 0, 0, currentSeeds);
    }

    for (int s = 0; s < stateCount; s++) {
        uint64_t board = buffers[s].board;
        int moveCount = 0;
        int bestScore = -100;

        for (int house = 0; house < 6; house++) {
            if (!HouseHasSeeds(house, 0, board)) continue;

            uint64_t childBoard = board;
            int captured = ApplyNoCheck(house, childBoard, 0);
            bestScore = max(captured, bestScore);

            if (!PlayerHasSeeds(1, childBoard))
                continue;
            else {
                int seedsLeft = buffers[s].currentSeeds - captured;
                buffers[s].capturedBuffer[moveCount] = captured;
                buffers[s].childIndexBuffer[moveCount] =
                    arrayStarts[seedsLeft] +
                    IndexFunction(FlipBoard(childBoard), seedsLeft);
                buffers[s].moveHouse[moveCount] = house;
                moveCount++;
            }
        }
        buffers[s].moveCount = moveCount;
        if (moveCount == 0) bestScore = GetPlayerSeeds(0, board);

        SetSeedScore(s, 1, bestScore);
    }

    int cnt = stateCount;
    file.open("samples", std::ios::binary);
    std::cerr << "CNT: " << cnt << "\n";
    file.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));

    std::vector<std::pair<int, int>> scores(stateCount);

    for (int turn = 2; turn <= PATTERN_LIMIT_9; turn++) {
        for (int s = 0; s < stateCount; s++) {
            int bestScore = -100;

            if (buffers[s].moveCount == 0)
                bestScore = GetPlayerSeeds(0, buffers[s].board);
            else {
                for (int m = 0; m < buffers[s].moveCount; m++) {
                    int captured = buffers[s].capturedBuffer[m];
                    int childIndex = buffers[s].childIndexBuffer[m];
                    int score = captured -
                                endSeeds[(turn - 2) * stateCount + childIndex];
                    bestScore = max(score, bestScore);
                }
            }

            if (turn == PATTERN_LIMIT_9) {
                scores[s].first = bestScore;
                // print_sample(s, turn, bestScore);
            } else if (turn == 50) {
                scores[s].second = bestScore;
            }

            SetSeedScore(s, turn, bestScore);
        }
    }

    int temp = 0;
    for (auto [a, b] : scores) {
        if (a != b) {
            temp++;
            // std::cerr << a << " " << b << "\n";
        }
    }

    std::cerr << 1.0 * temp / scores.size() << "\n";

    auto now = std::chrono::high_resolution_clock::now();
    auto calcTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
            .count();

    std::cerr << "End games done. Time: " << calcTime
              << " ms. States: " << stateCount << endl;
}

int main() {
    SowingArray();
    CaptureArray();
    NextHouseArray();
    FillStateCountLookups();
    GenerateBook();
}
