#include "chess.h"
#include "mcts.h"
#include "inference.h"
#include <torch/script.h>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <alphazero_traced.pt>\\n";
        return 1;
    }
    std::string model_path = argv[1];
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.msg() << "\\n";
        return 1;
    }
    InferenceQueue inf(model, /*batch_size=*/4096);
    ParallelMCTS mcts(inf, /*sims=*/800);

    // Placeholder single game
    std::vector<ChessBoard> boards;
    boards.emplace_back();
    auto policies = mcts.search(boards);
    std::cout << "Ran placeholder search. Policy[0] first entry: " << policies[0][0] << "\\n";
    return 0;
}
