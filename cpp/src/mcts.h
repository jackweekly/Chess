#pragma once
#include "chess.h"
#include "inference.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

struct MCTSNode {
    float prior;
    int visit = 0;
    float value_sum = 0.f;
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    std::mutex node_mutex;
};

class ParallelMCTS {
public:
    ParallelMCTS(InferenceQueue& inf, int sims);
    std::vector<std::vector<float>> search(const std::vector<ChessBoard>& boards);

private:
    InferenceQueue& inf_;
    int sims_;
    // TODO: action encoder/decoder mapping here
};
