#pragma once
#include <torch/script.h>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <vector>
#include <atomic>
#include <optional>

struct InferenceRequest {
    torch::Tensor input; // (119,8,8) float
    std::promise<std::pair<torch::Tensor, torch::Tensor>> promise;
};

class InferenceQueue {
public:
    InferenceQueue(torch::jit::script::Module model, size_t batch_size);
    ~InferenceQueue();

    std::future<std::pair<torch::Tensor, torch::Tensor>> enqueue(const torch::Tensor& input);

private:
    void worker();

    torch::jit::script::Module model_;
    size_t batch_size_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<InferenceRequest> queue_;
    std::thread worker_thread_;
    std::atomic<bool> stop_{false};
};
