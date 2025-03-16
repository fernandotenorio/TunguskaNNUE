#pragma once
#include <vector>
#include <string>
#include <cstdint> // For int32_t
#include <cassert>
#include <sstream>
#include <iostream>
#include <utility> // For std::pair
#include "Engine/Board.h"
#include "Engine/defs.h"
#include "Constants.h"

struct FeatureChanges {
    int add_white_count;
    int rem_white_count;
    std::vector<int> add_white;
    std::vector<int> rem_white;
    int add_black_count;
    int rem_black_count;
    std::vector<int> add_black;
    std::vector<int> rem_black;

    FeatureChanges() :
        add_white_count(0),
        rem_white_count(0),
        add_white(4),
        rem_white(4),
        add_black_count(0),
        rem_black_count(0),
        add_black(4),
        rem_black(4) {}

    void add_white_feat(int idx) {
        add_white[add_white_count++] = idx;
    }
    void rem_white_feat(int idx) {
        rem_white[rem_white_count++] = idx;
    }

    void add_black_feat(int idx) {
        add_black[add_black_count++] = idx;
    }
    void rem_black_feat(int idx) {
        rem_black[rem_black_count++] = idx;
    }

    void print() {
        std::cout << "Add white features idx" << std::endl;
        for (size_t i = 0; i < add_white_count; i++) {
            std::cout << add_white[i] << ", ";
        }
        std::cout << "\nRem white features idx" << std::endl;
        for (size_t i = 0; i < rem_white_count; i++) {
            std::cout << rem_white[i] << ", ";
        }
        std::cout << "Add black features idx" << std::endl;
        for (size_t i = 0; i < add_black_count; i++) {
            std::cout << add_black[i] << ", ";
        }
        std::cout << "\nRem black features idx" << std::endl;
        for (size_t i = 0; i < rem_black_count; i++) {
            std::cout << rem_black[i] << ", ";
        }
    }
};

class FeatureExtractor {
public:
    static std::pair<std::vector<int>, std::vector<int>> extractFeatures(const Board& board);
    static std::pair<std::vector<int>, std::vector<int>> extractFeatures(const std::string& fen);
    static FeatureChanges moveDiffFeatures(const Board& board, int move);

private:
    // No member variables needed
    static void addFeature(std::vector<int>& features, int feature_index) {
        features[feature_index] = 1;
    }
};