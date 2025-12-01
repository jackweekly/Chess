#include "inference.h"
#include <torch/torch.h>

InferenceQueue::InferenceQueue(torch::jit::script::Module model, size_t batch_size)
    : model_(std::move(model)), batch_size_(batch_size) {
    model_.to(torch::kCUDA);
    worker_thread_ = std::thread(&InferenceQueue::worker, this);
}

InferenceQueue::~InferenceQueue() {
    stop_.store(true);
    cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

std::future<std::pair<torch::Tensor, torch::Tensor>> InferenceQueue::enqueue(const torch::Tensor& input) {
    InferenceRequest req;
    req.input = input;
    auto fut = req.promise.get_future();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push(std::move(req));
    }
    cv_.notify_one();
    return fut;
}

void InferenceQueue::worker() {
    while (!stop_.load()) {
        std::vector<InferenceRequest> batch;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return stop_.load() || !queue_.empty(); });
            if (stop_.load()) break;
            while (!queue_.empty() && batch.size() < batch_size_) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop();
            }
        }
        if (batch.empty()) continue;
        std::vector<torch::Tensor> inputs;
        inputs.reserve(batch.size());
        for (auto& r : batch) inputs.push_back(r.input.unsqueeze(0));
        auto input_batch = torch::cat(inputs, 0).to(torch::kCUDA);
        auto outputs = model_.forward({input_batch}).toTuple();
        auto pi = outputs->elements()[0].toTensor().to(torch::kCPU);
        auto v = outputs->elements()[1].toTensor().to(torch::kCPU);
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].promise.set_value({pi[i].clone(), v[i].clone()});
        }
    }
}
