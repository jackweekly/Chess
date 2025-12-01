#include "mcts.h"
#include <torch/torch.h>

ParallelMCTS::ParallelMCTS(InferenceQueue& inf, int sims) : inf_(inf), sims_(sims) {}

std::vector<std::vector<float>> ParallelMCTS::search(const std::vector<ChessBoard>& boards) {
    // Placeholder: returns uniform policy until full implementation.
    std::vector<std::vector<float>> out;
    out.reserve(boards.size());
    const int action_size = 4672;
    for (size_t i = 0; i < boards.size(); ++i) {
        std::vector<float> pi(action_size, 0.f);
        if (action_size > 0) pi[0] = 1.f;
        out.push_back(std::move(pi));
    }
    return out;
}
